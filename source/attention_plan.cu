// SPDX-License-Identifier: Apache-2.0
// Convert kern's rectangular page table to FlashInfer CSR and split-KV plans.
// Runs once per forward, before all layers, on the caller's stream/graph.
extern "C" __global__ void attention_plan(
    const int* pages, const int* lengths, int* indices, int* indptr, int* last,
    int* request, int* tile, int* merge, unsigned char* valid, int* chunk,
    int groups, int width, int split_size, int max_splits, int prefill_tiles, int plan_slots, int page_size) {
  int lane = threadIdx.x;
  __shared__ int selected_split;
  if (lane == 0) {
    int longest = 1;
    for (int j = 0; j < groups; ++j) longest = max(longest, lengths[j]);
    selected_split = split_size ? split_size : max(256, ((longest + max_splits * page_size - 1) / (max_splits * page_size)) * page_size);
  }
  __syncthreads();
  split_size = selected_split;
  if (lane == 0) { *chunk = split_size; indptr[0] = 0; merge[0] = 0; }
  for (int s=lane; s<groups; s+=blockDim.x) {
    int np=0, ns=0;
    for (int j=0;j<=s;j++) { np+=(lengths[j]+page_size-1)/page_size; ns+=(lengths[j]+split_size-1)/split_size; }
    indptr[s+1]=np; merge[s+1]=ns; last[s]=(lengths[s]-1)%page_size+1;
  }
  __syncthreads();
  for (int flat=lane;flat<groups*width;flat+=blockDim.x) {
    int s=flat/width, p=flat%width;
    if (p<(lengths[s]+page_size-1)/page_size) indices[indptr[s]+p]=pages[flat];
  }
  for (int flat=lane;flat<groups*max_splits;flat+=blockDim.x) {
    int s=0;
    while (s<groups && flat>=merge[s+1]) ++s;
    valid[flat]=s<groups;
    request[flat]=s<groups?s:0;
    tile[flat]=s<groups?flat-merge[s]:0;
  }
  // Prefill has one request; its separate tile arrays follow the decode arrays.
  for (int t=lane;t<prefill_tiles;t+=blockDim.x) {
    request[plan_slots+t]=0;
    tile[plan_slots+t]=t;
  }
}

// Expand the shared split plan for 16-query causal verification (GQA=6,
// FlashInfer query tile=64). Partial rows are query-major within each request.
extern "C" __global__ void verify_attention_plan(
    const int* split_indptr, const int* request, const int* kv_tile,
    const unsigned char* valid, int* verify_request, int* verify_kv_tile,
    int* qo_tile, unsigned char* verify_valid, int* merge, int* output_indptr,
    int groups, int max_splits) {
  for(int j=threadIdx.x;j<groups*max_splits*2;j+=blockDim.x) {
    verify_request[j]=request[j/2];
    verify_kv_tile[j]=kv_tile[j/2];
    qo_tile[j]=j%2;
    verify_valid[j]=valid[j/2];
  }
  for(int j=threadIdx.x;j<=groups*16;j+=blockDim.x) {
    int s=j/16, row=j%16;
    merge[j]=split_indptr[s]*16;
    if(s<groups)merge[j]+=row*(split_indptr[s+1]-split_indptr[s]);
  }
  for(int s=threadIdx.x;s<=groups;s+=blockDim.x)output_indptr[s]=split_indptr[s]*16;
}
