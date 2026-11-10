# DSpine on SM120

The native bundle combines the FP8 target with
[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16).
It uses five BF16 draft layers, causal blocks of 16 rows (one anchor and 15
proposals), an 8192-token draft sliding window and greedy prefix acceptance.
The target checkpoint's own MTP weights are not used.

Each `round` is one CUDA Graph: draft, serial last-write refinement, target
verify, draft-context projection, prefix acceptance and ReplaySSM state commit.
The server stages anchors and reads accepted target tokens through kern's
existing counted-forward protocol. Fifteen internal padding positions preserve
the public 262144-token context limit for 16-row rounds. No model-specific runtime changes or
serving-time Python/JIT are required. Maximum concurrency remains 16.

Target features after layers 1, 16, 31, 46 and 61 feed the checkpoint's `fc`
projection. Context K/V are materialized separately from the masked draft
forward. The BF16 draft cache is a per-sequence ring of 8208 positions: the
8192-token window plus 16 speculative positions, so rejected writes cannot
overwrite history still visible to the next round. Its payload is about
160.3 MiB per sequence, in addition to the target KV and recurrent state.

ReplaySSM uses the upstream empty-history verify and compact D/K/G commit,
including low-part residuals for stored D/K. Verify leaves committed state
untouched. Acceptance selects the prefix to materialize, then convolution
history is advanced to the same boundary. Persistent target state remains
BF16; verify arithmetic is FP32 and state rounds at the accepted boundary.
That rounding schedule differs from ordinary one-token BF16 decode. This is
an explicitly tested numerical policy, not a guarantee of bitwise identity
for arbitrary prompts.

Target verification supports temperature, top-k, top-p, min-p and seeded sampling.
The 16 target rows independently draw tokens using their absolute sequence lengths
as RNG offsets. Parameters are expanded from each request to its verification rows.
The existing prefix matcher accepts draft tokens only while they equal these target
draws, then emits the first differing target token, or the bonus draw if all 15 match.
There is no draft-probability acceptance test or residual-distribution correction.
The draft lookup remains greedy; accepted prefixes may shorten for sampled requests.
FlashInfer includes the flattened row index in its RNG subsequence, so fixed seeds
replay at fixed batch layouts, without promising identity with ordinary decoding.

## Build

Use the standard CUDA build, then prepare greedy lookup tables from the target
LM head and draft checkpoint. These derived weights are build artifacts; the
original checkpoints remain unchanged. TF32 is disabled during preparation.

```sh
uv run scripts/build.py
uv run scripts/prepare_dspine.py \
  --checkpoint /path/to/Qwen3.8-27B-FP8 \
  --draft /path/to/Qwen3.8-27B-DSpine-b16
uv run scripts/build_dsl.py --dspine --gemm-tuning results/gemm-tuning.json
uv run scripts/gen.py \
  --checkpoint /path/to/Qwen3.8-27B-FP8 \
  --kv-dtype fp8 --kv-scales results/kv-scales.json \
  --dspine /path/to/Qwen3.8-27B-DSpine-b16 \
  --dspine-tables out/dspine-tables --out out/qwen38-dspine
```

The checked-in tuning profile is for RTX PRO 5000 72GB and Triton 3.8.
To retune on that hardware, with Python development headers available:

```sh
uv run scripts/tune_gemm.py --out out/gemm-tuning.json
uv run scripts/build_dsl.py --dspine --gemm-tuning out/gemm-tuning.json
```

`triton.autotune` evaluates 32 configurations per M/N/K over all five target
FP8 projection geometries and M=1,2,4,8,16,32,64,128,256. Selection retains BF16
activations and checkpoint block scales. AOT compilation preserves the JIT's
16-byte pointer alignment and checkpoint N/K divisibility attributes. M remains
dynamic, including single-row ordinary decode.
The manifest chooses the specialization for its live sequence bucket. The
profile contains timings and numerical checks; it is not rerun at startup.

## Serve and trace

```sh
KERN=/path/to/kern/target/release/kern
"$KERN" server --config out/qwen38-dspine/kern.toml qwen38 \
  --port 8000 --max-seqs 16 --served-model-name qwen3.8-27b \
  --trace --trace-port 8001 --trace-dir out/dspine-trace
```

HES timestamps and call labels are available in the trace Parquet files.
`GET :8001/kern/metrics` exposes step counters; `POST :8001/kern/trace/rotate`
closes the current files for analysis. The ordinary `/metrics` endpoint also
reports speculative acceptance counters. The separate ordinary bundle remains available for plain-decode comparisons.

The independent recurrent-state gate uses the public runtime wheel:

```sh
uv run --with /path/to/kern_llm.whl scripts/replayssm_check.py \
  --bundle out/qwen38-dspine
```

It compares AOT verify and accepted-state materialization with a direct FP32
recurrence, checks acceptance lengths 1..16, null slots, and verifies that the
speculative forward does not modify committed state.

## FlashInfer GEMM candidate

FlashInfer's pinned SM120 CUTLASS groupwise GEMM is benchmarked separately.
It requires FP8 activations, whereas the current verify path keeps BF16
activations. It is not selected by this bundle. To reproduce its microbenchmark:

```sh
git -C third_party/flashinfer submodule update --init 3rdparty/cutlass
nvcc -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
  -arch=sm_120a -shared -Xcompiler -fPIC \
  -Ithird_party/flashinfer/include \
  -Ithird_party/flashinfer/3rdparty/cutlass/include \
  -Ithird_party/flashinfer/3rdparty/cutlass/tools/util/include \
  tests/flashinfer_gemm.cu -o build/sm_120/libflashinfer_gemm.so
uv run scripts/flashinfer_gemm_bench.py \
  --library build/sm_120/libflashinfer_gemm.so
```

GEMM timings exclude activation quantization and full-model effects. Random
input error against BF16 activations does not establish model quality.


The small serving sweep builds its prompts with lm-eval and retains every raw
HTTP result, including usage and finish reason. Run it once per server variant:

```sh
uv run --with 'lm-eval[api]==0.4.11' scripts/spec_bench.py \
  --checkpoint /path/to/Qwen3.8-27B-FP8 --out out/gsm8k-dspine
```

It warms the exact eight prompts first, then measures concurrency 1/2/4/8.
Use `--prompts out/gsm8k-dspine/prompts.json` for the other server to replay the
same inputs. `results/gsm8k-dspine.json` records the current comparison.


## FlashInfer target verification attention

Target verification uses the existing FlashInfer causal paged prefill kernel
and variable-length merge, with at most 32 KV partitions per sequence. The
shared page/partition plan is expanded for each request's 16 verification rows:
GQA=6 maps 96 query/head positions to two 64-query tiles per partition. Partial
outputs and LSEs are query-major within each request, and merge pointers use
that exact layout. Planning runs once before the round's layers; the 16 target
attention layers reuse the same workspace and plan. K and V page strides and
calibrated scales are inherited from the ordinary attention ABI.

The draft's 8192-window attention remains unchanged. No model-specific runtime
branch or new attention math kernel is required. This changes reduction order,
so numerical agreement is assessed with output tolerances and end-to-end KL,
without requiring identical generated tokens. See
[validation and performance](../results/verify-attention.md).
