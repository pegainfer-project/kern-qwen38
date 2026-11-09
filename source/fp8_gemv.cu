// SPDX-License-Identifier: Apache-2.0
// Adapted from pegainfer's streaming row-dot GEMV. Batch-1/small-batch candidate:
// W[N,K] E4M3, dequantization scales[ceil(N/128),K/128] BF16, x[B,K] BF16.
// K must be divisible by 128. Each warp owns an output row, with FP32 accumulation.
// This W8A16 path avoids activation quantization for bandwidth-bound decode.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void kern_gemv_fp8_block128(
    const __nv_fp8_e4m3* __restrict__ weights, const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int rows, int cols) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows) return;
  const int batch = blockIdx.y;
  float acc = 0.f;
  for (int col = lane * 16; col < cols; col += 512) {
    const uint4 packed = __ldcs(reinterpret_cast<const uint4*>(weights + (long long)row * cols + col));
    const auto* w = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
    const float s = float(scales[(long long)(row / 128) * (cols / 128) + col / 128]);
    const auto* input = x + (long long)batch * cols + col;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc = fmaf(float(__float2bfloat16_rn(float(w[i]) * s)), float(input[i]), acc);
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) acc += __shfl_down_sync(0xffffffff, acc, offset);
  if (lane == 0) y[(long long)batch * rows + row] = __float2bfloat16_rn(acc);
}
