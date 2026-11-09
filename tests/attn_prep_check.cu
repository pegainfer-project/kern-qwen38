// SPDX-License-Identifier: Apache-2.0
// Check fused norm/RoPE/cache writes, non-unit scales, page boundaries and masked slots.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../source/attn_prep.cu"
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} } while(0)
using B=__nv_bfloat16;
template<class T> T* device(const std::vector<T>& x){T* p;CUDA(cudaMalloc(&p,x.size()*sizeof(T)));CUDA(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 constexpr int tokens=3,width=14336,cache_size=2*4*64*256;
 std::vector<B> qkv(tokens*width),cosine(tokens*32),sine(tokens*32),q(tokens*24*256),k(tokens*4*256);
 std::vector<float> qw(256),kw(256),ks{.02f},vs{.04f};
 std::vector<long long> slots{0,65,-1};std::vector<Cache> kc(cache_size,Cache(0.f)),vc=kc;
 for(size_t i=0;i<qkv.size();++i)qkv[i]=B(.7f*sinf(float(i)*.03f));
 for(int i=0;i<256;++i){qw[i]=1.f+.1f*cosf(float(i));kw[i]=1.f+.1f*sinf(float(i));}
 for(size_t i=0;i<cosine.size();++i){cosine[i]=B(cosf(float(i)*.1f));sine[i]=B(sinf(float(i)*.1f));}
 auto dx=device(qkv);auto dq=device(q);auto dk=device(k);auto dqw=device(qw);auto dkw=device(kw);
 auto dc=device(cosine);auto ds=device(sine);auto dkc=device(kc);auto dvc=device(vc);auto dslots=device(slots);auto dks=device(ks);auto dvs=device(vs);
 ATTN_ENTRY<<<tokens,256>>>(dx,dqw,dkw,dc,ds,dq,dk,dkc,dvc,dslots,tokens,1e-6f,width,4*64*256,256,64*256
#if KERN_KV_FP8
 ,dks,dvs
#endif
 );
 CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
 CUDA(cudaMemcpy(q.data(),dq,q.size()*sizeof(B),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(k.data(),dk,k.size()*sizeof(B),cudaMemcpyDeviceToHost));
 CUDA(cudaMemcpy(kc.data(),dkc,kc.size()*sizeof(Cache),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(vc.data(),dvc,vc.size()*sizeof(Cache),cudaMemcpyDeviceToHost));
 float max_error=0;
 for(int t=0;t<tokens;++t)for(int r=0;r<28;++r){bool isq=r<24;int h=isq?r:r-24;int base=t*width+(isq?h*512:24*512+h*256);float values[256],sum=0;
  for(int i=0;i<256;++i){float x=float(qkv[base+i]);sum+=x*x;}
  float inv=1.f/sqrtf(sum/256.f+1e-6f);
  for(int i=0;i<256;++i)values[i]=float(B(float(qkv[base+i])*inv*(isq?qw[i]:kw[i])));
  for(int i=0;i<32;++i){float a=values[i],b=values[i+32],c=float(cosine[t*32+i]),s=float(sine[t*32+i]);values[i]=float(B(fmaf(a,c,-float(B(b*s)))));values[i+32]=float(B(fmaf(a,s,float(B(b*c)))));}
  const B* actual=(isq?q.data()+t*24*256:k.data()+t*4*256)+h*256;
  for(int i=0;i<256;++i){float error=fabsf(float(actual[i])-values[i]);if(!std::isfinite(error))return 1;max_error=fmaxf(max_error,error);}
 }
 if(max_error>.032f){fprintf(stderr,"norm/rope error %g\n",max_error);return 1;}
 std::vector<Cache> expected_k(cache_size,Cache(0.f)),expected_v=expected_k;
 for(int t=0;t<tokens;++t)if(slots[t]>=0)for(int h=0;h<4;++h)for(int i=0;i<256;++i){int at=(slots[t]/64)*4*64*256+(slots[t]%64)*256+h*64*256+i;
#if KERN_KV_FP8
 expected_k[at]=Cache(float(k[(t*4+h)*256+i])/ks[0]);expected_v[at]=Cache(float(qkv[t*width+24*512+4*256+h*256+i])/vs[0]);
#else
 expected_k[at]=k[(t*4+h)*256+i];expected_v[at]=qkv[t*width+24*512+4*256+h*256+i];
#endif
 }
 for(int i=0;i<cache_size;++i)if(float(kc[i])!=float(expected_k[i])||float(vc[i])!=float(expected_v[i])){fprintf(stderr,"cache mismatch at %d\n",i);return 1;}
 printf("{\"kv\":\"%s\",\"tokens\":3,\"max_norm_rope_error\":%.8g,\"cache_exact\":true}\n",KERN_KV_FP8?"fp8":"bf16",max_error);
 for(void* p:{(void*)dx,(void*)dq,(void*)dk,(void*)dqw,(void*)dkw,(void*)dc,(void*)ds,(void*)dkc,(void*)dvc,(void*)dslots,(void*)dks,(void*)dvs})CUDA(cudaFree(p));
}
