# SPDX-License-Identifier: Apache-2.0
"""Tiled FP8-weight GEMM with checkpoint block128 scales."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def fp8_block_gemm(
    A, W, S, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, A8: tl.constexpr
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        kk = tile * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0).to(tl.float32)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0.0)
        sw = tl.load(S + (n // 128) * tl.cdiv(K, 128) + (tile * BK) // 128, n < N, 0).to(tl.float32)
        if A8:
            sa = tl.maximum(tl.max(tl.abs(a), axis=1), 1e-12) / 448.0
            aq = (a / sa[:, None]).to(tl.float8e4nv)
            acc += tl.dot(aq, w) * sa[:, None] * sw[None, :]
        else:
            # Decode keeps BF16 activations; expand only the current weight tile.
            wb = (w.to(tl.float32) * sw[None, :]).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), wb, acc)
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))


@triton.jit(do_not_specialize=["M"])
def fp8_act_quant(A, AQ, SA, M, K, BM: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    kb = tl.program_id(1)
    k = kb * 128 + tl.arange(0, 128)
    mask = (m[:, None] < M) & (k[None, :] < K)
    a = tl.load(A + m[:, None] * K + k[None, :], mask, 0.0).to(tl.float32)
    sa = tl.maximum(tl.max(tl.abs(a), axis=1), 1e-12) / 448.0
    aq = (a / sa[:, None]).to(tl.float8e4nv)
    tl.store(AQ + m[:, None] * K + k[None, :], aq, mask)
    tl.store(SA + m * tl.cdiv(K, 128) + kb, sa, m < M)


@triton.jit(do_not_specialize=["M"])
def fp8_w8a8_gemm(AQ, SA, W, S, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    k_blocks = tl.cdiv(K, 128)
    for tile in range(tl.cdiv(K, BK)):
        kk = tile * BK + k
        kb = (tile * BK) // 128
        aq = tl.load(AQ + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0.0)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0.0)
        sa = tl.load(SA + m * k_blocks + kb, m < M, 0.0)
        sw = tl.load(S + (n // 128) * k_blocks + kb, n < N, 0).to(tl.float32)
        acc += tl.dot(aq, w) * sa[:, None] * sw[None, :]
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))
