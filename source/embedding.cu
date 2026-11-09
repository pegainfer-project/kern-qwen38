// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>
using B = __nv_bfloat16;

extern "C" __global__ void embedding(const int64_t* ids, const B* table,
                                     B* out, int hidden, int vocab) {
  int row = blockIdx.x;
  int64_t id = ids[row];
  if (id < 0 || id >= vocab) __trap();
  for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    out[int64_t(row) * hidden + j] = table[id * hidden + j];
}
