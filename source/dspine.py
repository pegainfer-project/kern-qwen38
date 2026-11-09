# SPDX-License-Identifier: Apache-2.0
# Adapted from Tencent/DSpine, sglang_dspine/kernels/dspine.py (321b0fc).
"""DSpine greedy bridge and refinement kernels, compiled AOT for SM120."""

import triton
import triton.language as tl


@triton.jit
def prepare_read(
    X,
    Residual,
    Raw,
    Normalized,
    RMS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def normalize_codes(
    Projected,
    Codebook,
    Anchors,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    ANCHOR_ROW: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if ANCHOR_ROW and row % T == 0:
        token = tl.load(Anchors + row // T)
        z = tl.load(Codebook + token * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = tl.sqrt(float(C)) * z / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def feature_message(Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid(query + sender + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def feature_finish(
    Raw,
    Delta,
    RMS,
    Valid,
    Weight,
    Output,
    Normed,
    H: tl.constexpr,
    T: tl.constexpr,
    WRITE: tl.constexpr,
    NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    previous = tl.maximum(row - 1, 0)
    edge = (row % T != 0) & tl.load(Valid + row) & tl.load(Valid + previous)
    update = rms * delta * tl.cast(edge & WRITE, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    if NORM:
        x = result.to(tl.float32)
        variance = tl.sum(x * x, 0) / H
        y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
        weight = tl.load(Weight + k, k < H, 0)
        y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
        tl.store(Normed + row * H + k, y, k < H)


@triton.jit
def _chunk_max(X, stride_x, V, CM, NCH, BLOCK: tl.constexpr, CH: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    m = tl.max(tl.reshape(x, (BLOCK // CH, CH)), axis=1)
    j = blk * (BLOCK // CH) + tl.arange(0, BLOCK // CH)
    tl.store(CM + row * NCH + j, m, mask=j < NCH)


@triton.jit
def _select_topk(
    X,
    stride_x,
    V,
    CM,
    NCH,
    OUT,
    K: tl.constexpr,
    CH: tl.constexpr,
    NCHPAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NCHPAD)
    cm = tl.load(CM + row * NCH + offs, mask=offs < NCH, other=float("-inf"))
    kk = tl.arange(0, K)
    chunks = tl.zeros([K], dtype=tl.int32)
    for j in tl.static_range(K):
        (_m, i) = tl.max(cm, axis=0, return_indices=True, return_indices_tie_break_left=True)
        chunks = tl.where(kk == j, i, chunks)
        cm = tl.where(offs == i, float("-inf"), cm)
    chunks = tl.sort(chunks)
    idx = tl.reshape(chunks[:, None] * CH + tl.arange(0, CH)[None, :], (K * CH,))
    vals = tl.load(X + row * stride_x + idx, mask=idx < V, other=float("-inf")).to(tl.float32)
    lane = tl.arange(0, K * CH)
    res = tl.zeros([K], dtype=tl.int32)
    for j in tl.static_range(K):
        (_m, i) = tl.max(vals, axis=0, return_indices=True, return_indices_tie_break_left=True)
        sel = tl.sum(tl.where(lane == i, idx, 0), axis=0)
        res = tl.where(kk == j, sel, res)
        vals = tl.where(lane == i, float("-inf"), vals)
    res = tl.sort(res)
    tl.store(OUT + row * K + kk, res.to(tl.int64))


@triton.jit
def _serial_table(
    Raw,
    stride_raw,
    NormW,
    Cand,
    InputIds,
    LmW,
    stride_w,
    TG,
    TSV,
    Query,
    stride_q,
    Rms,
    GateB,
    Out,
    Confidence,
    H: tl.constexpr,
    RK: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TM1: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    n = tl.program_id(0)
    b = n // TM1
    r = n % TM1
    row = b * (TM1 + 1) + 1 + r
    kp = tl.arange(0, KP)
    live = kp < K
    ids = tl.load(Cand + n * K + kp, mask=live, other=0)
    prev = tl.load(Cand + tl.maximum(n - 1, 0) * K + kp, mask=live, other=0)
    anchor = tl.load(InputIds + b * (TM1 + 1)).to(tl.int64)
    sid = tl.where(r == 0, anchor, prev)
    base = tl.zeros([KP], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BH):
        off = h0 + tl.arange(0, BH)
        u = tl.load(Raw + row * stride_raw + off).to(tl.float32) * tl.load(NormW + off).to(tl.float32)
        w = tl.load(LmW + ids[:, None] * stride_w + off[None, :], mask=live[:, None], other=0.0).to(
            tl.float32
        )
        base += tl.sum(w * u[None, :], axis=1)
    s = tl.zeros([KP, KP], dtype=tl.float32)
    for r0 in tl.static_range(0, RK, BR):
        rr = r0 + tl.arange(0, BR)
        gate = tl.load(TSV + sid[:, None] * (2 * RK) + rr[None, :], mask=live[:, None], other=0.0).to(
            tl.float32
        )
        val = tl.load(
            TSV + sid[:, None] * (2 * RK) + RK + rr[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(Query + row * stride_q + rr).to(tl.float32)
        gb = tl.load(GateB + rr).to(tl.float32)
        m = tl.sigmoid(q[None, :] + gate + gb[None, :]) * val
        g = tl.load(TG + ids[:, None] * RK + rr[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        s += tl.dot(m, tl.trans(g), input_precision="ieee")
    base = tl.where(live, base, -1e30)
    s = base[None, :] + tl.load(Rms + row).to(tl.float32) * s
    (best_value, best) = tl.max(s, axis=1, return_indices=True, return_indices_tie_break_left=True)
    tl.store(Out + n * K + kp, best, mask=live)
    if WITH_CONFIDENCE:
        scale = tl.maximum(tl.load(Rms + row).to(tl.float32), 1e-06)
        mass = tl.sum(tl.exp((s - best_value[:, None]) / scale), 1)
        tl.store(Confidence + n * K + kp, 1.0 / mass, mask=live)


@triton.jit
def _serial_chain(
    Idx,
    Cand,
    Out,
    Confidence,
    ConfidenceOut,
    K: tl.constexpr,
    TM1: tl.constexpr,
    RPAD: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    b = tl.program_id(0)
    rr = tl.arange(0, RPAD)
    kk = tl.arange(0, K)
    inside = rr[:, None] < TM1
    at = (b * TM1 + rr[:, None]) * K + kk[None, :]
    tab = tl.load(Idx + at, mask=inside, other=0)
    cnd = tl.load(Cand + at, mask=inside, other=0)
    if WITH_CONFIDENCE:
        probs = tl.load(Confidence + at, mask=inside, other=0.0)
    j = tl.sum(tl.where((rr[:, None] == 0) & (kk[None, :] == 0), tab, 0))
    previous = tl.full((), 0, tl.int64)
    for r in tl.static_range(TM1):
        if WITH_CONFIDENCE:
            probability = tl.sum(tl.where((rr[:, None] == r) & (kk[None, :] == previous), probs, 0.0))
            tl.store(ConfidenceOut + b * TM1 + r, probability)
        tl.store(
            Out + b * TM1 + r,
            tl.sum(tl.where((rr[:, None] == r) & (kk[None, :] == j), cnd, 0)),
        )
        previous = j
        j = tl.sum(tl.where((rr[:, None] == r + 1) & (kk[None, :] == j), tab, 0))
