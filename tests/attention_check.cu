// SPDX-License-Identifier: Apache-2.0
// FlashInfer GQA=6 paged decode and causal prefill against a CPU softmax oracle.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#define KERN_HEAD_DIM 256
#include "../source/attention.cu"
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} } while(0)
template<class T> T* device(const std::vector<T>& x){T* p;CUDA(cudaMalloc(&p,x.size()*sizeof(T)));CUDA(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 for(int len:{1,67,1025}){
  int pages=(len+63)/64;size_t elements=size_t(pages)*4*64*256;
  float ks=KERN_KV_FP8?.02f:1.f,vs=KERN_KV_FP8?.04f:1.f;
  std::vector<B> q(24*256),out(q.size());std::vector<KV> k(elements),v(elements);
  for(size_t i=0;i<q.size();++i)q[i]=B(.5f*sinf(float(i)*.07f));
  for(size_t i=0;i<elements;++i){k[i]=KV(.5f*cosf(float(i)*.11f)/ks);v[i]=KV(.6f*sinf(float(i)*.09f)/vs);}
  std::vector<int> ids(pages);std::iota(ids.begin(),ids.end(),0);std::reverse(ids.begin(),ids.end());
  auto dq=device(q);auto dout=device(out);auto dk=device(k);auto dv=device(v);auto di=device(ids);
  auto dp=device(std::vector<int>{0,pages});auto last=device(std::vector<int>{(len-1)%64+1});
  auto req=device(std::vector<int>{0});auto tile=device(std::vector<int>{0});auto chunk=device(std::vector<int>{len});
  auto qi=device(std::vector<int>{0,1});auto rows=device(std::vector<unsigned>{1});
  paged_kv_t<KV,int> cache(4,64,256,1,QKVLayout::kHND,dk,dv,di,dp,last);
  std::vector<float> expected(q.size()),logits(len);
  for(int h=0;h<24;++h){float maxval=-INFINITY,sum=0;
   for(int t=0;t<len;++t){int at=(ids[t/64]*4*64+(h/6)*64+t%64)*256;float dot=0;for(int j=0;j<256;++j)dot+=float(q[h*256+j])*float(k[at+j])*ks;logits[t]=dot/16.f;maxval=fmaxf(maxval,logits[t]);}
   for(float& l:logits){l=expf(l-maxval);sum+=l;}
   for(int j=0;j<256;++j){float total=0;for(int t=0;t<len;++t){int at=(ids[t/64]*4*64+(h/6)*64+t%64)*256;total+=logits[t]*float(v[at+j])*vs;}expected[h*256+j]=total/sum;}
  }
  for(int mode=0;mode<2;++mode){
   if(!mode){Decode p;p.q=dq;p.o=dout;p.paged_kv=cache;p.num_qo_heads=24;p.q_stride_n=24*256;p.q_stride_h=256;p.window_left=-1;p.sm_scale=ks/16.f;p.v_scale=vs;p.request_indices=req;p.kv_tile_indices=tile;p.kv_chunk_size_ptr=chunk;p.padded_batch_size=1;
    decode_g6<<<dim3(1,4),dim3(X,6,(X*6>128?X*6:128)/(X*6)),32768>>>(p);
   }else{Prefill p;p.q=dq;p.o=dout;p.paged_kv=cache;p.num_qo_heads=24;p.group_size=uint_fastdiv(6);p.q_stride_n=24*256;p.q_stride_h=256;p.window_left=-1;p.sm_scale=ks/16.f;p.v_scale=vs;p.q_indptr=qi;p.o_indptr=qi;p.request_indices=req;p.qo_tile_indices=tile;p.kv_tile_indices=tile;p.kv_chunk_size_ptr=chunk;p.padded_batch_size=1;p.max_total_num_rows=1;p.total_num_rows=rows;
    CUDA(cudaFuncSetAttribute(prefill,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Traits::SharedStoragePaged)));
    prefill<<<dim3(1,1,4),dim3(32,get_num_warps_q(64),get_num_warps_kv(64)),sizeof(Traits::SharedStoragePaged)>>>(p);
   }
   CUDA(cudaGetLastError());CUDA(cudaMemcpy(out.data(),dout,out.size()*sizeof(B),cudaMemcpyDeviceToHost));
   float error=0;for(size_t i=0;i<out.size();++i){float d=fabsf(float(out[i])-expected[i]);if(!std::isfinite(d))return 1;error=fmaxf(error,d);}
   if(error>.008f){fprintf(stderr,"attention failed: mode=%d len=%d error=%g\n",mode,len,error);return 1;}
   printf("{\"kv\":\"%s\",\"mode\":\"%s\",\"length\":%d,\"gqa\":6,\"max_error\":%.8g}\n",KERN_KV_FP8?"fp8":"bf16",mode?"prefill":"decode",len,error);
  }
  for(void* p:{(void*)dq,(void*)dout,(void*)dk,(void*)dv,(void*)di,(void*)dp,(void*)last,(void*)req,(void*)tile,(void*)chunk,(void*)qi,(void*)rows})CUDA(cudaFree(p));
 }
}
