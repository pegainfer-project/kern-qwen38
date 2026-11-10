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

## Verification MLP W8A8 experiment — 2026-11-10

`scripts/verify_w8a8.py` builds an isolated candidate from an existing DSpine
bundle. It autotunes the gate/up and down projections at 16/32/64/128/256 rows,
then compiles the selected W8A8 kernels ahead of time. Activation quantization
uses the existing prefill kernel. The original bundle and generator defaults
are unchanged.

```sh
uv run python scripts/verify_w8a8.py \
  --baseline out/qwen38-dspine --out out/verify-w8a8
uv run python scripts/verify_w8a8_check.py \
  --kern ../kern/target/release/kern --config out/qwen38-dspine/kern.toml \
  --baseline out/qwen38-dspine/model.json --baseline-kernels out/qwen38-dspine/kernels \
  --candidate out/verify-w8a8/model.json --candidate-kernels out/verify-w8a8/kernels \
  --out out/verify-w8a8/numerics
```

On one RTX PRO 5000 72GB, the November 10 service workload's 157 scenarios
(261,672 calls, 16 samples each) cost 26,673.647 GPU seconds on the baseline
and 25,153.260 seconds on the candidate: **5.70% less GPU time**. Round cost
fell 0.305%/1.663%/8.021%/15.476%/19.554% in sequence buckets 1/2/4/8/16;
unchanged prefill cost increased 0.090%.

Batch-1 model probes used one repeated prose prompt at four prefix lengths
(128/1,024/8,192/32,768), four rounds each. All emitted prefixes and accepted
counts matched, 52 output tokens per side. Twelve rounds had identical
proposals, allowing 192 verification-logit rows to be scored: mean KL 0.001537,
maximum row KL 0.011344. Six top-1 changes were beyond the accepted prefix.
The default 0.01 maximum-row KL gate fails; the observed numerical difference
was explicitly accepted for this experiment. The checker retains its original
threshold. Larger-batch model quality and loaded-service acceptance have not
been validated.

The measured outcomes are in `results/verify-w8a8.json`; selected Triton
configurations and synthetic operator errors are in
`results/verify-w8a8-tuning.json`. These measurements support the experiment;
they do not establish a general request-throughput improvement.

## Native FP8 prefill experiment — 2026-11-11

`scripts/prefill_fp8.py` builds a separate bundle with native FP8 attention
from the served BF16-query/FP8-KV bundle:

```sh
uv run python scripts/prefill_fp8.py \
  --baseline out/qwen38-dspine --out out/prefill-fp8
```

The candidate quantizes each query head dynamically, caches each lane's two
scales before the KV loop, and uses the pinned FlashInfer FP8 layout with
16 packed query rows. It generates the corresponding plan and reads the ABI
and shared-memory sizes from a compiled probe. The pinned VO-split output
writer bypasses the V-scale transform; the wrapper applies that scale inside
the same kernel after synchronizing the CTA. Decode and verification remain
unchanged. No third-party headers or generator defaults are modified.

On one RTX PRO 5000 72GB, the November 10 traffic workload (157 scenarios,
261,672 calls, 16 samples each) costs **25,904.147 GPU seconds**, versus
26,673.647 on the deployed baseline: **2.88% less total GPU time**. Weighted
prefill program time falls 8.46%; prefill attention call time, including
online Q quantization and output scaling, falls 21.16%. Unchanged round time
increases 0.34% between runs. These are fixed-work kernel measurements.

A teacher-forced `kern test e2e-logits` diagnostic set of 12 questions and
768 positions passes the mean-KL gate: **0.00150 <= 0.01**. It covers code,
Chinese, mathematics and long contexts through the 32–128k depth bucket.
Seven top-1 flips remain; passing this mean-KL check does not imply identical
greedy generations or identical speculative acceptance. The question set is
small and is not a broad quality benchmark.

Artifacts, recordings and logs are retained in `out/prefill-fp8/`. The first
integrated variant and its incomplete performance run are in
`out/prefill-fp8-v1/`; only the final variant has a complete workload report.
`scripts/prefill_round_check.py` compares free-running speculative outputs
and counters for supplied tokenized prompts. The separate verification-MLP
W8A8 optimization above is not included in these measurements.

Four batch-1 free-running round checks each generated 256 tokens. Baseline
used 339 rounds in total, candidate 340. Code and long-32k improved, Chinese
and long-8k regressed; all four outputs diverged. This small check found no
aggregate acceptance collapse and does not establish unchanged quality or
acceptance under service load. Portable benchmark and per-case evidence is
in `results/prefill-fp8.json`.

## ReplaySSM verify register pressure — 2026-11-11

The SM120 verify specialization uses K slices of 16 elements over eight
iterations, with the original 32-value tile and four warps. This reduces the
live FP32/TF32x3 operand footprint: 219 registers and no stack spill, versus
255 registers and a 640-byte stack with 64-element slices. The kernel source,
FP32 intermediate types, TF32x3 dot policy and BF16 residual-cache storage are
unchanged. K-reduction grouping changes, so outputs are not bitwise identical.

The direct FP32 recurrence and accepted-state oracle passes at batch
1/2/4/8/16, including acceptance lengths 1..16 and a null state slot. Maximum
verify error is 3.05e-5 and maximum committed-state error 9.77e-4, within the
existing atol=0.002 / rtol=0.02. Rebuild the cubins and regenerate the bundle
before checking it; with the local kern Python binding available:

```sh
uv run python scripts/replayssm_check.py \
  --bundle out/qwen38-dspine --out out/replayssm-timed.json --timing
```

The optional timing mode uses 12 CUDA-graph replays per shape, flushing 128 MiB
before each timed launch, and records output/state hashes. Verify timing runs
before commit, preserving the initial state. Isolated synthetic verify time
falls 17–30% over batches 1–16 on one RTX PRO 5000 72GB.

Full-model speculative-round probes at 128/1k/8k/32k prefixes emit identical
accepted prefixes over 16 rounds (52 tokens per side). Fifteen rounds have
identical proposals, allowing 240 distributions to be compared: mean KL
0.000615 and maximum row KL 0.003515, passing the 0.01 per-row gate. Two top-1
flips are beyond the accepted prefix. Whole-model probes cover batch 1;
operator oracles cover the larger batches. Ordinary decode-only logits checks
would not exercise the changed ReplaySSM specialization.

The complete November 10 workload (157 scenarios, 261,672 calls, 16 samples)
costs **26,376.796 GPU seconds**, versus 26,673.647 on the baseline: **1.11%
less total GPU time**. Weighted replay-verify call time falls **22.27%** and
round program time **1.89%**. Unchanged prefill time increases 0.24% between
runs; the total result is reported without normalizing that drift. This
candidate excludes the W8A8-MLP and native FP8-prefill experiments above.
Portable results, resources, operator samples and round evidence are retained
in `results/replayssm-spill.json`; raw artifacts are in `out/replay-spill/`.
