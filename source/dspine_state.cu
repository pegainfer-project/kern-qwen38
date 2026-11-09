// SPDX-License-Identifier: Apache-2.0
// Speculative GDN writes live in a scratch state until the accepted prefix is known.
#include <cuda_bf16.h>
#include <stdint.h>
using B=__nv_bfloat16;
extern "C" __global__ void spec_conv_commit(const B* raw,B* state,const int* lines,const int* count,
    long long stride) {
  int seq=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x;
  if(c>=10240||lines[seq]<=0)return;
  B* history=reinterpret_cast<B*>(reinterpret_cast<char*>(state)+(long long)lines[seq]*stride);
  int n=count[seq];
  B v[3];
#pragma unroll
  for(int j=0;j<3;++j){int row=n-3+j;v[j]=row<0?history[(row+3)*10240+c]:raw[((long long)seq*16+row)*16384+c];}
#pragma unroll
  for(int j=0;j<3;++j)history[j*10240+c]=v[j];
}
extern "C" __global__ void spec_count(const long long* draft,const long long* verified,int* count) {
  int seq=blockIdx.x;
  if(threadIdx.x)return;
  int n=0;
  while(n<15&&draft[seq*15+n]==verified[seq*16+n])++n;
  count[seq]=n+1;
}

extern "C" __global__ void spec_conv(const B* raw,const B* w,const B* state,const int* lines,
    B* out,long long stride) {
  int seq=blockIdx.z,row=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x;
  if(c>=10240)return;
  long long line=lines[seq];
  if(line<=0){out[((long long)seq*16+row)*10240+c]=B(0.f);return;}
  const B* history=reinterpret_cast<const B*>(reinterpret_cast<const char*>(state)+line*stride);
  float acc=0.f;
#pragma unroll
  for(int j=0;j<4;++j){int pos=row+j-3;float x=pos<0?float(history[(pos+3)*10240+c]):float(raw[((long long)seq*16+pos)*16384+c]);acc+=float(B(x*float(w[c*4+j])));}
  out[((long long)seq*16+row)*10240+c]=B(acc/(1.f+expf(-acc)));
}
extern "C" __global__ void spec_constants(int* zero,int* one,long long* indices,int n) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j<n){zero[j]=0;one[j]=1;indices[j]=j;}
}
extern "C" __global__ void spec_copy(const B* src,B* dst,int elements) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<elements)dst[j]=src[j];
}
extern "C" __global__ void spec_tap(const B* src,B* dst,int tap) {
 int row=blockIdx.x,d=blockIdx.y*blockDim.x+threadIdx.x;
 if(d<5120)dst[(long long)row*25600+tap*5120+d]=src[(long long)row*5120+d];
}

// Every verification row samples from the target distribution. Prefix matching
// consumes only the draws before the first mismatch; no rejection correction.
extern "C" __global__ void spec_sampling(
    const float* temperature, const int* top_k, const float* top_p, const float* min_p,
    const long long* seed, const int* lengths,
    float* row_temperature, int* row_top_k, float* row_top_p, float* row_min_p,
    long long* row_seed, int* row_lengths) {
  int seq=blockIdx.x, j=threadIdx.x;
  if(j>=16)return;
  int row=seq*16+j;
  row_temperature[row]=temperature[seq];
  row_top_k[row]=top_k[seq];
  row_top_p[row]=top_p[seq];
  row_min_p[row]=min_p[seq];
  row_seed[row]=seed[seq];
  // The runtime seq_len includes all 16 verification inputs.
  row_lengths[row]=lengths[seq]-15+j;
}
