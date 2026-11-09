// SPDX-License-Identifier: Apache-2.0
// Benchmark adapter for the pinned FlashInfer SM120 CUTLASS implementation.
#include <flashinfer/gemm/gemm_groupwise_sm120.cuh>
extern "C" int run_fp8(void* workspace,unsigned long long bytes,void* a,void* w,void* sa,void* sw,void* out,int m,int n,int k,void* stream) {
 return flashinfer::gemm::CutlassGroupwiseScaledGEMMSM120<1,128,128,true,cutlass::float_e4m3_t,cutlass::bfloat16_t>(workspace,bytes,(cutlass::float_e4m3_t*)a,(cutlass::float_e4m3_t*)w,(float*)sa,(float*)sw,(cutlass::bfloat16_t*)out,m,n,k,1,(cudaStream_t)stream);
}
