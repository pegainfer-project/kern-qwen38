# Operator contracts

The native batched bundle targets SM120. `scripts/gen.py` binds the
checkpoint, derived normalization weights, calibrated FP8 KV scales, chunked
prefill and recurrent decode into a runnable kern manifest (16 sequences, context up to the checkpoint limit). CUDA build metadata
and Triton AOT metadata record the compiler versions and cubin identities.

| Module | Implementation | Validation |
| --- | --- | --- |
| `gdn_decode_bf16` | kern fused conv and recurrent update/norm/gate; BF16 state updated in FP32 registers | 16 recurrent steps against CPU, batch 1/8, null slot |
| `gdn_decode_fp32` | kern reference using asynchronous shared-memory state loads | Same recurrence oracle |
| `attn_prep_fp8` / `attn_prep_bf16` | Fused Q/K Gemma norm, partial RoPE, cache append | CPU norm/RoPE, exact cache conversion, non-unit scales, masked slots, page boundary |
| `attention_fp8` / `attention_bf16` | Latest FlashInfer paged causal prefill, decode and merge | GQA=6, length 1/67/1025, permuted physical pages; one query per request |
| `fp8_gemv` | Streaming warp-per-row W8A16 decode candidate | CPU FP64 dot-product oracle, block-scale boundaries and row tails, batch 1/8 |
| `embedding` | kern-qwen3 indexed BF16 gather with vocabulary bounds trap | Whole-model E2E |
| `gemma_rms_norm`, `weight_prep` | kern norms and derived weight tables | Whole-model E2E |
| `sigmoid_mul`, `silu_mul` | kern vectorized gated outputs and MLP activation | Whole-model E2E |
| `argmax` | Two-stage greedy in one op with private scratch | Whole-model E2E |
| `sampling`, `sample_glue` | FlashInfer stochastic sampling adapters | Compilation only; not selected by model generator |
| `model_glue`, `gdn_*` AOT stages | Parallel causal conv, state transpose, FLA chunked recurrence and gated norm | Whole-model E2E and chunk continuity |
| `fp8_block_gemm` | Local native FP8 Tensor Core prefill GEMM | Whole-model E2E |
| `attention_plan`, `copy_rows`, `line_copy` | kern-qwen3 page planning and kern state/row movement | Compilation only |

## GDN

The fused decode input is `[seqs, 16384]` in Q/K/V/Z order, with Q/K heads=16,
V heads=48, K=V=128. Gate input is `[seqs, 96]`, B followed by A.
Norm weights and `dt_bias` are BF16; `A_log` is FP32.

Convolution uses grid `[seqs, ceil(10240/256)]`, block 256. Its state pointer
addresses the start of a sequence line, containing `[3, 10240]` BF16 history.
The recurrent launch uses grid `[48, seqs]`, block 256. Pass its state pointer
already offset past the convolution history; `line_stride_bytes` includes the
whole line. Matrix state is `[48, 128 V, 128 K]`, K contiguous. Slot 0 is null.

BF16 recurrent state requires **zero dynamic shared memory**; the FP32 reference
requires **65536 bytes** and the corresponding CUDA function attribute.
The output uses the FP32 updated state before rounding the persistent write.
The GDN line-table row stride is the declared maximum batch, 16 entries; it
must not be replaced by the live batch size.
The conv entry remains BF16 history in either module. The standalone recurrence
gate does not yet validate the conv entry or chunked prefill continuity.

The BF16 register path avoids expanding a whole head into 64 KiB of shared
memory. Timing is a synthetic, warm-cache graph microbenchmark, not a model
speed claim. Batch 8 includes one null slot (seven active sequences).

## Full attention

Use 64-token HND pages. Strides passed to `attn_prep` count cache **elements**,
so FP8 and BF16 use identical logical strides and different byte sizes.
Negative slots skip both K and V writes. Q/K norm weights are FP32 `1 + weight`
tables produced by `weight_prep`. RoPE touches the first 64 of 256 dimensions;
cos/sin inputs are gathered per token before this prologue.

FP8 uses dequantization factors: `stored = E4M3(value / scale)`. Both factors
must be positive and finite and must remain fixed while cached pages survive.
Calibrate them before serving; silently using 1.0 is not a precision policy.
The attention consumer folds K scale into `sm_scale = K_scale / sqrt(256)`
and sets `v_scale = V_scale`. These scalar values must agree with the pointers
read by the append kernel. BF16 uses both scales equal to 1.

Parameter structs extend upstream structs with `v_scale`; use build-derived
offsets/initialized bytes, never offsets copied from older FlashInfer versions.
Set both K **and V** page/head/token strides (latest FlashInfer has independent
V strides). Prefill grid is `[query_tiles, 1, kv_heads]`; GQA=6 decode grid is
`[plan_slots, kv_heads, 1]`. Decode blocks use
`[256 / (16 / sizeof(KV)), 6, 1]`; prefill geometry comes from build metadata.
`o_indptr` must be bound even for non-partitioned prefill. `attention_plan`
takes an explicit page-size argument; pass 64 to match the append layout.

Multi-query causal prefill is selected by the generated bundle and covered by
E2E. Split-KV decode and merge are selected and tested with shuffled physical pages
and different sequence lengths, including the maximum-context index boundary.
Decode uses adaptive split-KV planning (at most 32 partitions per sequence)
and FlashInfer's variable-length merge. Prefill is scheduled one request at a
time; the running requests join the `decode_batch` program.

## FP8 linear projections

Official checkpoint weights are E4M3 `[N,K]`. Their `weight_scale_inv` tensors
are BF16 `[ceil(N/128), ceil(K/128)]`; despite the suffix these are the factors
used to dequantize the stored weights. Confirmed from downloaded tensor headers.
Ordinary decode and speculative verification use the same autotuned tiled
GEMM interface at every batch size. Activations and decoded weight tiles remain
BF16; weights stay FP8 in device memory. `source/fp8_gemv.cu` is a standalone
historical candidate and is not selected by either model manifest.

Prefill uses the local `fp8_block_gemm` Triton AOT kernel: 32x64x128 tiles,
native E4M3 dot products and per-row/per-128 activation scales applied alongside
checkpoint weight scales. BF16 exclusions use the runtime cuBLASLt extern.
Weights remain quantized on device; there is no full weight expansion at load.
The FlashInfer SM120 CUTLASS groupwise GEMM remains a performance candidate.
kern's per-tensor FP8 extern cannot consume these block128 scales unchanged.

## Prefill and decode state continuity

The seven local Triton AOT stages originate from pegainfer's FLA-derived chunk
pipeline. Local adaptations handle interleaved B/A projection rows, match the
decode normalization epsilon and round beta to BF16. Convolution reads the old
history in parallel before a separate history-commit launch. State gather and
scatter transpose between chunk `[head,K,V]` FP32 workspace and persistent
`[head,V,K]` BF16 layout.

Prefill performs FP32 state computation within each internal 64-token chunk,
then rounds the recurrent state to BF16, independent of aligned external chunk
boundaries; decode rounds persistent state every token. The
225-token continuity prompt generated the same 16 tokens with chunk 64/128
and with eager/graph decode. This is an integration check, not a proof of
long-context numerical equivalence.

## Remaining work

Evaluate further FlashInfer GDN/GEMM candidates, long-context KV calibration
and recurrent drift. The native bundles are text-only and greedy, with at most
16 running sequences. The optional [DSpine bundle](dspine.md) uses ReplaySSM and
autotuned verification; its state policy and measured scope are documented there.
