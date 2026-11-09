// SPDX-License-Identifier: Apache-2.0
// CPU recurrence oracle: validates state writes and gated outputs over repeated steps.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../source/gdn_decode.cu"
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} } while(0)
using B=__nv_bfloat16;
constexpr int SHARED = GDN_STATE_BF16 ? 0 : 65536;
float rb(float x){return float(B(x));}
template<class T> T* device(const std::vector<T>& x){T* p;CUDA(cudaMalloc(&p,x.size()*sizeof(T)));CUDA(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 CUDA(cudaSetDevice(0)); cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=12 || prop.minor!=0){fprintf(stderr,"SM120 required\n");return 1;}
 for(int batch:{1,8}){
  const int heads=48,K=128,V=128,Q=16,width=16384;
  const size_t elems=size_t(heads)*V*K;
  std::vector<B> x(batch*width),ba(batch*96),bias(heads),w(V,B(1.f)),out(batch*heads*V);
  std::vector<float> a(heads,-2.f);std::vector<int> lines(batch);
  std::vector<State> state((batch+1)*elems,State(0.f));
  for(size_t i=0;i<x.size();++i)x[i]=B(.2f*sinf(float(i)*.13f)+.01f);
  for(size_t i=0;i<ba.size();++i)ba[i]=B(.3f*cosf(float(i)*.17f));
  for(int n=0;n<batch;++n)lines[n]=n+1;
  if(batch==8)lines.back()=0;
  for(size_t i=elems;i<state.size();++i)state[i]=State(.01f*sinf(float(i%8191)*.07f));
  auto gold=state;
  auto dx=device(x);auto db=device(ba);auto da=device(a);auto d_bias=device(bias);auto dw=device(w);
  auto ds=device(state);auto dl=device(lines);auto dout=device(out);
  CUDA(cudaFuncSetAttribute(GDN_ENTRY,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
  float max_output=0,max_state=0;double squared_error=0,squared_gold=0;
  for(int step=0;step<16;++step){
   GDN_ENTRY<<<dim3(heads,batch),256,SHARED>>>(dx,db,da,d_bias,dw,dout,ds,dl,1.f/sqrtf(128.f),1e-6f,width,96,elems*sizeof(State));
   CUDA(cudaGetLastError());CUDA(cudaMemcpy(out.data(),dout,out.size()*sizeof(B),cudaMemcpyDeviceToHost));
   for(int n=0;n<batch;++n)for(int h=0;h<heads;++h){
    float expected[128]={};
    if(lines[n]){
     float q[128],k[128],qs=0,ks=0;int kh=h/3;
     for(int j=0;j<K;++j){q[j]=float(x[n*width+kh*K+j]);k[j]=float(x[n*width+Q*K+kh*K+j]);qs+=q[j]*q[j];ks+=k[j]*k[j];}
     for(int j=0;j<K;++j){q[j]/=sqrtf(qs+1e-6f)*sqrtf(128.f);k[j]/=sqrtf(ks+1e-6f);}
     float av=float(ba[n*96+48+h])+float(bias[h]);
     float decay=expf(-expf(a[h])*(av>20.f?av:log1pf(expf(av))));
     float beta=rb(1.f/(1.f+expf(-float(ba[n*96+h]))));
     for(int v=0;v<V;++v){
      size_t base=size_t(lines[n])*elems+size_t(h)*V*K+v*K;float dot=0;
      for(int j=0;j<K;++j)dot+=float(gold[base+j])*decay*k[j];
      float u=(float(x[n*width+2*Q*K+h*V+v])-dot)*beta,o=0;
      for(int j=0;j<K;++j){float next=fmaf(k[j],u,float(gold[base+j])*decay);gold[base+j]=State(next);o+=next*q[j];}
      expected[v]=rb(o);
     }
     float sq=0;for(float val:expected)sq+=val*val;
     float inv=1.f/sqrtf(sq/128.f+1e-6f);
     for(int v=0;v<V;++v){float z=float(x[n*width+2*Q*K+heads*V+h*V+v]);expected[v]=rb(expected[v]*inv*float(w[v])*z/(1.f+expf(-z)));}
    }
    for(int v=0;v<V;++v){float error=float(out[(n*heads+h)*V+v])-expected[v];max_output=std::max(max_output,fabsf(error));squared_error+=error*error;squared_gold+=expected[v]*expected[v];}
   }
  }
  CUDA(cudaMemcpy(state.data(),ds,state.size()*sizeof(State),cudaMemcpyDeviceToHost));
  for(size_t i=0;i<state.size();++i)max_state=std::max(max_state,fabsf(float(state[i])-float(gold[i])));
  float relative=sqrt(squared_error/std::max(squared_gold,1e-20));
  if(!std::isfinite(relative)||max_output>.02f||relative>.01f||max_state>.002f){fprintf(stderr,"GDN failed: out=%g rel=%g state=%g\n",max_output,relative,max_state);return 1;}
  cudaEvent_t begin,end;CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
  // Graph timing removes host submission gaps from the per-layer comparison.
  cudaStream_t stream;CUDA(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
  CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  for(int i=0;i<100;++i)GDN_ENTRY<<<dim3(heads,batch),256,SHARED,stream>>>(dx,db,da,d_bias,dw,dout,ds,dl,1.f/sqrtf(128.f),1e-6f,width,96,elems*sizeof(State));
  CUDA(cudaStreamEndCapture(stream,&graph));CUDA(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  CUDA(cudaGraphLaunch(exec,stream));CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaEventRecord(begin,stream));CUDA(cudaGraphLaunch(exec,stream));CUDA(cudaEventRecord(end,stream));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,begin,end));
  printf("{\"state\":\"%s\",\"batch\":%d,\"steps\":16,\"max_output_error\":%.8g,\"relative_output_error\":%.8g,\"max_state_error\":%.8g,\"graph_us\":%.4f}\n",GDN_STATE_BF16?"bf16":"fp32",batch,max_output,relative,max_state,ms*10);
  for(void* p:{(void*)dx,(void*)db,(void*)da,(void*)d_bias,(void*)dw,(void*)ds,(void*)dl,(void*)dout})CUDA(cudaFree(p));
  CUDA(cudaGraphExecDestroy(exec));CUDA(cudaGraphDestroy(graph));CUDA(cudaStreamDestroy(stream));CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));
 }
}
