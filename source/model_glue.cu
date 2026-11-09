// SPDX-License-Identifier: Apache-2.0
// Single-token bring-up glue. Projection weights remain quantized in device memory.
#include <cuda_bf16.h>
#include <stdint.h>
using B = __nv_bfloat16;

extern "C" __global__ void residual_add(B* out, const B* x, const B* y, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = B(float(x[j]) + float(y[j]));
}

// Compute the model's 64-dimensional partial rotary embedding at runtime positions.
extern "C" __global__ void rope_gather(B* cosine, B* sine, const int64_t* positions, float theta) {
  int j = threadIdx.x;
  if (j >= 32) return;
  float angle = float(positions[blockIdx.x]) / powf(theta, float(j) / 32.f);
  cosine[blockIdx.x * 32 + j] = B(cosf(angle));
  sine[blockIdx.x * 32 + j] = B(sinf(angle));
}

// Single request, unpartitioned paged decode plan. CSR pages are filled by the caller.
extern "C" __global__ void decode_plan(const int* pages, const int* lengths, int* indices,
                                        int* indptr, int* last, int* request, int* tile, int* chunk) {
  int n = (lengths[0] + 63) / 64;
  for (int p = threadIdx.x; p < n; p += blockDim.x) indices[p] = pages[p];
  if (!threadIdx.x) {
    indptr[0] = 0; indptr[1] = n; last[0] = (lengths[0] - 1) % 64 + 1;
    request[0] = 0; tile[0] = 0; chunk[0] = lengths[0];
  }
}

// Parallel convolution reads a stable history; commit history in a separate launch.
extern "C" __global__ void conv_prefill(const B* qkvz, const B* w, const B* state,
    const int* line, B* out, int tokens, long long stride) {
  int c = blockIdx.y * blockDim.x + threadIdx.x, t = blockIdx.x;
  if (c >= 10240) return;
  const B* history = reinterpret_cast<const B*>(reinterpret_cast<const char*>(state) + (long long)line[0] * stride);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int pos = t + j - 3;
    float x = pos < 0 ? float(history[(pos + 3) * 10240 + c]) : float(qkvz[(long long)pos * 16384 + c]);
    acc += float(B(x * float(w[c * 4 + j])));
  }
  out[(long long)t * 10240 + c] = B(acc / (1.f + expf(-acc)));
}
extern "C" __global__ void conv_commit(const B* qkvz, B* state, const int* line, int tokens, long long stride) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 10240) return;
  B* h = reinterpret_cast<B*>(reinterpret_cast<char*>(state) + (long long)line[0] * stride);
  B v[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) { int t = tokens - 3 + j; v[j] = t < 0 ? h[(t + 3) * 10240 + c] : qkvz[(long long)t * 16384 + c]; }
#pragma unroll
  for (int j = 0; j < 3; ++j) h[j * 10240 + c] = v[j];
}

// Chunk kernels use [head,K,V]; decode uses [head,V,K]. Persistent state is BF16.
extern "C" __global__ void state_gather(const B* state, const int* line, float* out, long long stride) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= 48 * 128 * 128) return;
  int h = j / 16384, k = (j / 128) % 128, v = j % 128;
  const B* src = reinterpret_cast<const B*>(reinterpret_cast<const char*>(state) + (long long)line[0] * stride);
  out[j] = float(src[h * 16384 + v * 128 + k]);
}
extern "C" __global__ void state_scatter(B* state, const int* line, const float* in, long long stride) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= 48 * 128 * 128) return;
  int h = j / 16384, v = (j / 128) % 128, k = j % 128;
  B* dst = reinterpret_cast<B*>(reinterpret_cast<char*>(state) + (long long)line[0] * stride);
  dst[j] = B(in[h * 16384 + k * 128 + v]);
}
extern "C" __global__ void gdn_norm_gate(B* out, const B* in, const B* qkvz, const B* w, float eps) {
  int h = blockIdx.y, t = blockIdx.x, lane = threadIdx.x;
  float x[4], z[4], sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {int d = lane * 4 + j; x[j] = float(in[(long long)t * 6144 + h * 128 + d]);z[j] = float(qkvz[(long long)t * 16384 + 10240 + h * 128 + d]);sum += x[j] * x[j];}
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, d);
  float inv = rsqrtf(sum / 128.f + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {int d = lane * 4 + j;out[(long long)t * 6144 + h * 128 + d] = B(x[j] * inv * float(w[d]) * z[j] / (1.f + expf(-z[j])));}
}
extern "C" __global__ void prefill_plan(int* request, int* tile, int* zero, int tokens) {
  int n = (tokens * 6 + 63) / 64;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {request[j] = 0;tile[j] = j;zero[j] = 0;}
}
