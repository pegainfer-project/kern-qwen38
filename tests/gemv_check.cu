// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../source/fp8_gemv.cu"
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} } while(0)
template<class T> T* device(const std::vector<T>& x){T* p;CUDA(cudaMalloc(&p,x.size()*sizeof(T)));CUDA(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 for(int batch:{1,8}){
  constexpr int N=259,K=384;std::vector<__nv_fp8_e4m3> w(N*K);std::vector<__nv_bfloat16> s(3*3),x(batch*K),y(batch*N);
  for(int i=0;i<N*K;++i)w[i]=__nv_fp8_e4m3(4.f*sinf(float(i)*.013f));
  for(int i=0;i<9;++i)s[i]=__nv_bfloat16(.03f+.01f*i);
  for(size_t i=0;i<x.size();++i)x[i]=__nv_bfloat16(.4f*cosf(float(i)*.017f));
  auto dw=device(w);auto ds=device(s);auto dx=device(x);auto dy=device(y);
  kern_gemv_fp8_block128<<<dim3((N+7)/8,batch),256>>>(dw,ds,dx,dy,N,K);CUDA(cudaGetLastError());CUDA(cudaMemcpy(y.data(),dy,y.size()*sizeof(y[0]),cudaMemcpyDeviceToHost));
  float error=0;
  for(int b=0;b<batch;++b)for(int n=0;n<N;++n){double sum=0;for(int k=0;k<K;++k)sum+=double(float(__nv_bfloat16(float(w[n*K+k])*float(s[(n/128)*3+k/128]))))*float(x[b*K+k]);float diff=fabsf(float(y[b*N+n])-float(__nv_bfloat16(float(sum))));if(!std::isfinite(diff))return 1;error=fmaxf(error,diff);}
  if(error>.032f){fprintf(stderr,"GEMV failed %g\n",error);return 1;}printf("{\"op\":\"fp8_block128_gemv\",\"batch\":%d,\"rows\":259,\"cols\":384,\"max_error\":%.8g}\n",batch,error);
  for(void* p:{(void*)dw,(void*)ds,(void*)dx,(void*)dy})CUDA(cudaFree(p));
 }
}
