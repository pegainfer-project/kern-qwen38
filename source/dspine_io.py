# SPDX-License-Identifier: Apache-2.0
"""DSpine context cache, rotary preparation and causal block attention."""

import triton
import triton.language as tl


@triton.jit
def rms_norm(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + r * H + d, d < H, 0).to(tl.float32)
    y = (x * tl.rsqrt(tl.sum(x * x, 0) / H + EPS)).to(Y.dtype.element_ty)
    w = tl.load(W + d, d < H, 0).to(tl.float32)
    tl.store(Y + r * H + d, (y.to(tl.float32) * w).to(Y.dtype.element_ty), d < H)


@triton.jit
def copy_columns(X, Y, M, N: tl.constexpr, STRIDE: tl.constexpr, OFFSET: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + r * N + d, (r < M) & (d < N), 0)
    tl.store(Y + r * STRIDE + OFFSET + d, x, (r < M) & (d < N))


@triton.jit
def block_ids(Input, Anchor, Ids, T: tl.constexpr, MASK: tl.constexpr):
    s = tl.program_id(0)
    r = tl.arange(0, T)
    a = tl.load(Input + s * T)
    tl.store(Anchor + s, a)
    tl.store(Ids + s * T + r, tl.where(r == 0, a, MASK))


@triton.jit
def proposals(Anchor, Draft, Ids, T: tl.constexpr):
    s = tl.program_id(0)
    r = tl.arange(0, T)
    a = tl.load(Anchor + s)
    d = tl.load(Draft + s * (T - 1) + tl.maximum(r - 1, 0))
    tl.store(Ids + s * T + r, tl.where(r == 0, a, d))


@triton.jit
def skip_anchors(X, Y, H: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    source = r // (T - 1) * T + 1 + r % (T - 1)
    tl.store(Y + r * H + d, tl.load(X + source * H + d, d < H, 0), d < H)


@triton.jit
def qkv_rope(X, QW, KW, Positions, Q, K, V, CONTEXT: tl.constexpr, EPS: tl.constexpr, THETA: tl.constexpr):
    r = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    stride = 2048 if CONTEXT else 6144
    qheads = 0 if CONTEXT else 32
    kvhead = head - qheads
    is_q = head < qheads
    offset = tl.where(is_q, head * 128, qheads * 128 + kvhead * 128)
    x = tl.load(X + r * stride + offset + d).to(tl.float32)
    w = tl.load(tl.where(is_q, QW, KW) + d).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x * x, 0) / 128 + EPS)).to(tl.bfloat16)
    x = (x.to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    other = tl.gather(x, (d + 64) % 128, 0)
    angle = tl.load(Positions + r).to(tl.float32) * tl.exp(-tl.log(THETA) * (d % 64) / 64)
    y = x * tl.cos(angle) + tl.where(d < 64, -other, other) * tl.sin(angle)
    if CONTEXT:
        tl.store(K + r * 1024 + head * 128 + d, y.to(tl.bfloat16))
        tl.store(V + r * 1024 + head * 128 + d, tl.load(X + r * 2048 + 1024 + head * 128 + d))
    else:
        if head < 32:
            tl.store(Q + r * 4096 + head * 128 + d, y.to(tl.bfloat16))
        else:
            tl.store(K + r * 1024 + kvhead * 128 + d, y.to(tl.bfloat16))
            tl.store(V + r * 1024 + kvhead * 128 + d, tl.load(X + r * 6144 + 5120 + kvhead * 128 + d))


@triton.jit
def context_store(
    K,
    V,
    Positions,
    Lines,
    Cache,
    TOKENS,
    PREFILL: tl.constexpr,
    LAYER: tl.constexpr,
    RING: tl.constexpr,
    T: tl.constexpr,
):
    r = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    seq = 0 if PREFILL else r // T
    line = tl.load(Lines + seq).to(tl.int64)
    pos = tl.load(Positions + r).to(tl.int64) % RING
    base = ((line * 5 + LAYER) * RING + pos) * 2048
    tl.store(Cache + base + d, tl.load(K + r * 1024 + d), (line > 0) & (r < TOKENS))
    tl.store(Cache + base + 1024 + d, tl.load(V + r * 1024 + d), (line > 0) & (r < TOKENS))


@triton.jit
def block_attention(
    Q,
    K,
    V,
    Cache,
    Lines,
    Pages,
    Lengths,
    Output,
    D: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    T: tl.constexpr,
    DRAFT: tl.constexpr,
    LAYER: tl.constexpr,
    RING: tl.constexpr,
    TABLE,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BK: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    rr = tl.arange(0, T)
    dd = tl.arange(0, D)
    kk = tl.arange(0, BK)
    end = tl.load(Lengths + seq)
    prefix = end - T
    qpos = prefix + rr
    q = tl.load(Q + (seq * T + rr[:, None]) * QH * D + head * D + dd[None, :])
    khead = head // (QH // KH)
    first = tl.maximum(0, prefix - 8192) if DRAFT else 0
    line = tl.load(Lines + seq).to(tl.int64) if DRAFT else tl.full((), 0, tl.int64)
    acc = tl.zeros((T, D), tl.float32)
    maximum = tl.full((T,), float("-inf"), tl.float32)
    denom = tl.zeros((T,), tl.float32)
    for start in range(first, end, BK):
        pos = start + kk
        if DRAFT:
            at = ((line * 5 + LAYER) * RING + pos.to(tl.int64) % RING) * 2048 + khead * D
            oldk = tl.load(Cache + at[:, None] + dd[None, :], (pos[:, None] < prefix), 0)
            oldv = tl.load(Cache + at[:, None] + 1024 + dd[None, :], (pos[:, None] < prefix), 0)
            live = tl.maximum(pos - prefix, 0)
            newk = tl.load(
                K + (seq * T + live[:, None]) * KH * D + khead * D + dd[None, :],
                (pos[:, None] >= prefix) & (pos[:, None] < end),
                0,
            )
            newv = tl.load(
                V + (seq * T + live[:, None]) * KH * D + khead * D + dd[None, :],
                (pos[:, None] >= prefix) & (pos[:, None] < end),
                0,
            )
            k = tl.where(pos[:, None] < prefix, oldk, newk)
            v = tl.where(pos[:, None] < prefix, oldv, newv)
        else:
            page = tl.load(Pages + seq * TABLE + pos // 64, pos < end, 0).to(tl.int64)
            at = page * (16 * 2 * 4 * 64 * D) + LAYER * (2 * 4 * 64 * D) + khead * 64 * D + (pos % 64) * D
            k = tl.load(Cache + at[:, None] + dd[None, :], pos[:, None] < end, 0.0).to(tl.bfloat16)
            v = tl.load(Cache + at[:, None] + 4 * 64 * D + dd[None, :], pos[:, None] < end, 0.0).to(
                tl.bfloat16
            )
        score = tl.dot(q, tl.trans(k)) * ((D**-0.5) * KS)
        valid = (pos[None, :] <= qpos[:, None]) & (pos[None, :] < end)
        if DRAFT:
            valid = valid & (pos[None, :] >= qpos[:, None] - 8192 + 1)
        score = tl.where(valid, score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        prob = tl.exp(score - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), v)
        denom = denom * alpha + tl.sum(prob, 1)
        maximum = next_max
    result = acc / denom[:, None] * VS
    tl.store(Output + (seq * T + rr[:, None]) * QH * D + head * D + dd[None, :], result.to(tl.bfloat16))
