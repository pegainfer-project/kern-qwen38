# SPDX-License-Identifier: Apache-2.0
"""Native FP8 Tensor Core prefill GEMM with checkpoint block128 scales."""

import triton
import triton.language as tl


@triton.jit
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
            acc += tl.dot(a.to(tl.bfloat16), wb)
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))
