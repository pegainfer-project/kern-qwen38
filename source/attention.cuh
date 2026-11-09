// SPDX-License-Identifier: Apache-2.0
// FlashInfer types and launch geometry used by pegainfer's paged_launch.cuh.
#pragma once
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
using namespace flashinfer;
using B = __nv_bfloat16;
#ifndef KERN_KV_FP8
#define KERN_KV_FP8 1
#endif
using KV = std::conditional_t<KERN_KV_FP8, __nv_fp8_e4m3, B>;
// K scale is folded into sm_scale. V scale is applied by AttentionVariantBase.
struct Decode : BatchDecodeParams<B, KV, B, int> { float v_scale = 1.f; };
struct Prefill : BatchPrefillPagedParams<B, KV, B, int> { float v_scale = 1.f; };
using Variant = DefaultAttention<false, false, false, false>;
constexpr int D = KERN_HEAD_DIM;
constexpr int VEC = 16 / sizeof(KV), X = D / VEC;
// A fixed 64-query tile and two KV MMA tiles fit SM80 through SM121.
using Traits = KernelTraits<MaskMode::kCausal, 64, get_num_mma_q(64), 2,
    D / 16, D / 16, get_num_warps_q(64), get_num_warps_kv(64),
    PosEncodingMode::kNone, B, KV, B, float, int, Variant>;
static_assert(!Traits::IsInvalid());
