// SPDX-License-Identifier: Apache-2.0
// Native FP8 prefill with per-query-head scales and calibrated V output.
#define KERN_HEAD_DIM 256
#include "attention.cuh"
#include <cstddef>

using Q8 = __nv_fp8_e4m3;
using Params8 = BatchPrefillPagedParams<Q8, KV, B, int>;
struct Prefill8 : Params8 {
  float v_scale;
  const float* q_scales;
};
struct ScaledAttention : Variant {
  float q_scale_low, q_scale_high;
  template<class Params>
  __device__ __host__ ScaledAttention(const Params& params, uint32_t batch, uint8_t* smem)
      : Variant(params, batch, smem) {
#ifdef __CUDA_ARCH__
    // Each lane's score fragments cover these same two query heads for every KV tile.
    const int first = params.q_indptr[batch];
    const int low = params.qo_tile_indices[blockIdx.x] * 16 + threadIdx.x / 4;
    const int high = low + 8;
    q_scale_low = low < this->qo_len * 6
        ? params.q_scales[((first + low / 6) * 24) + blockIdx.z * 6 + low % 6] : 0.f;
    q_scale_high = high < this->qo_len * 6
        ? params.q_scales[((first + high / 6) * 24) + blockIdx.z * 6 + high % 6] : 0.f;
#endif
  }
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits * ((qo_idx * 6 + qo_head_idx % 6) % 16 < 8 ? q_scale_low : q_scale_high);
  })
};
using NativeTraits = KernelTraits<MaskMode::kCausal, 16, 1, 1, 16, 16, 1, 4,
    PosEncodingMode::kNone, Q8, KV, B, float, int, ScaledAttention>;
static_assert(!NativeTraits::IsInvalid());
static_assert(offsetof(Prefill8, v_scale) == offsetof(Prefill, v_scale));

extern "C" __global__ void quantize_q(const B* q, Q8* out, float* scales, int rows) {
  const int head = blockIdx.x * 4 + threadIdx.x / 32;
  if (head >= rows * 24) return;
  const int lane = threadIdx.x % 32;
  float values[8], maximum = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = float(q[head * 256 + lane + j * 32]);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  const float scale = fmaxf(maximum, 1.e-12f) / 448.f;
  if (lane == 0) scales[head] = scale;
#pragma unroll
  for (int j = 0; j < 8; ++j) out[head * 256 + lane + j * 32] = Q8(values[j] / scale);
}

extern "C" __global__ void prefill_plan_fp8(int* request, int* tile, int* zero, int rows) {
  for (int j = threadIdx.x; j < (rows * 6 + 15) / 16; j += blockDim.x) {
    request[j] = 0;
    tile[j] = j;
    zero[j] = 0;
  }
}

extern "C" __global__ __launch_bounds__(128)
void prefill_fp8(const __grid_constant__ Prefill8 p, int rows) {
  if (p.block_valid_mask && !p.block_valid_mask[blockIdx.x]) return;
  extern __shared__ uint8_t shared[];
  auto& storage = *reinterpret_cast<NativeTraits::SharedStoragePaged*>(shared);
  BatchPrefillWithPagedKVCacheDevice<NativeTraits>(p, storage);
  // The pinned VO-split writer bypasses the variant's V-scale transform.
  __syncthreads();
  const int request = p.request_indices[blockIdx.x];
  const int query_tile = p.qo_tile_indices[blockIdx.x];
  const int query_rows = p.get_qo_len(request);
  const int splits = p.partition_kv
      ? (p.get_kv_len(request) + *p.kv_chunk_size_ptr - 1) / *p.kv_chunk_size_ptr : 1;
  const int split = p.partition_kv ? p.kv_tile_indices[blockIdx.x] : 0;
  const int thread = threadIdx.x + 32 * (threadIdx.y + blockDim.y * threadIdx.z);
  for (int j = thread; j < 16 * 256; j += 128) {
    const int packed = query_tile * 16 + j / 256;
    if (packed < query_rows * 6 && p.q_indptr[request] + packed / 6 < rows) {
      const int output_row = p.o_indptr[request] + packed / 6 * splits + split;
      const int at = (output_row * 24 + blockIdx.z * 6 + packed % 6) * 256 + j % 256;
      p.o[at] = B(float(p.o[at]) * p.v_scale);
    }
  }
}

extern "C" __global__ void verify_plan_fp8(
    const int* split_indptr, const int* request, const int* kv_tile,
    const unsigned char* valid, int* verify_request, int* verify_kv_tile,
    int* qo_tile, unsigned char* verify_valid, int* merge, int* output_indptr,
    int groups, int max_splits) {
  for (int j = threadIdx.x; j < groups * max_splits * 6; j += blockDim.x) {
    verify_request[j] = request[j / 6];
    verify_kv_tile[j] = kv_tile[j / 6];
    qo_tile[j] = j % 6;
    verify_valid[j] = valid[j / 6];
  }
  for (int j = threadIdx.x; j <= groups * 16; j += blockDim.x) {
    const int seq = j / 16, row = j % 16;
    merge[j] = split_indptr[seq] * 16;
    if (seq < groups) merge[j] += row * (split_indptr[seq + 1] - split_indptr[seq]);
  }
  for (int seq = threadIdx.x; seq <= groups; seq += blockDim.x)
    output_indptr[seq] = split_indptr[seq] * 16;
}
