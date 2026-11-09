# SM120 validation and initial performance

Measured on NVIDIA RTX PRO 5000 72GB Blackwell, CUDA 13.5. The compiler,
driver and FlashInfer revision are recorded in `operators.json`.
`operators-memcheck.json` records a separate Compute Sanitizer run: all seven
executables passed, producing 20 numeric cases and no reported memory errors.

Synthetic GDN recurrent update plus output norm/gate, warm CUDA graph timing:

| Batch | FP32 state | BF16 register state |
| --- | ---: | ---: |
| 1 | 4.51 us | 4.34 us |
| 8 | 15.84 us | 9.96 us |

Batch 8 includes one null slot. Both implementations use the same inputs,
16-step CPU recurrence oracle and 100-launch timing graph. The BF16 shared-memory
prototype measured about 20 us at batch 8 and was replaced by register updates.
These are initial microbenchmarks, not end-to-end model performance or a claim
that this implementation beats every FlashInfer backend. Sanitizer timings are
instrumented and must not be used for performance comparisons.

Maximum output error against the CPU oracle was 0.000977 for BF16 state.
FP8 and BF16 attention passed GQA=6 at context lengths 1, 67 and 1025 with
non-contiguous page mappings. FP8 cache conversion matched expected bytes, and
block-FP8 GEMV passed batch 1/8. The gates do not establish whole-model quality,
long-run recurrent stability, chunked prefill correctness or speculative safety.

## Native whole-model bundle

Validated FP8 weights, calibrated E4M3 KV and BF16 recurrent storage on one
RTX PRO 5000 72GB Blackwell. `e2e.json` records the four short prompts:
32/32 generated token IDs match the independent Transformers BF16-dequantized
reference. First-token logits KL(reference || native) ranges from 0.00122 to
0.00674 (limit 0.1). Calibration uses these prompts too; this is a bring-up
smoke gate, not an independent broad quality evaluation.

A separate 225-token prompt crosses prefill boundaries. Chunk 64 and 128,
and eager versus CUDA Graph decode, generate the same 16 tokens.
The native whole-model Compute Sanitizer smoke run reports zero memory errors.
API-error reporting was disabled for this run because kern's parameter-layout
inspection deliberately calls `cuFuncGetParamInfo` until CUDA returns
`INVALID_VALUE` to signal the end; memory instrumentation remains enabled.

Initial single-request measurements, excluding weight loading:

| Workload | Measurement |
| --- | ---: |
| 5-token prompt, chunk 128 prefill | about 268 ms |
| 225-token prompt, chunk 128 prefill | about 740 ms / 304 tokens/s |
| CUDA Graph decode | about 27 ms/token |
| Weight residency | 27.4 GiB |
| Unique decode weight reads, excluding unused embedding rows | 26.92 GB/token |
| Effective weight-read bandwidth at 26.98 ms | 998 GB/s / 74.2% of 1344 GB/s |

The bandwidth calculation counts unique weight reads, not hardware DRAM
transactions; it excludes KV, recurrent state and intermediate-buffer traffic.
These are initial timings on short contexts. Prefill is functional but has
substantial optimization headroom. Long-context quality and speculative execution remain outside the validated scope.

## HTTP serving

The native kern-serve binary passed `/v1/models`, completions, chat completions
(with thinking disabled), SSE equivalence and 16 concurrent requests at
`--max-seqs 16` with automatic shared-pool capacity. Greedy completion matches the CLI output. This historical run used greedy-only sampling. Current sampling validation is
recorded in [sampling.md](sampling.md). See `http.json`; run `uv run scripts/http_check.py`
against the local server to repeat the gate.


## Batch and context validation

`batch.json` compares batch 1, 2, 3, 4, 8, 9 and 16 against singleton execution
for two consecutive steps, restoring identical KV/recurrent state before each
comparison. Physical KV pages are shuffled. Maximum logits KL is below 0.002;
the few top-1 differences have singleton margins of at most 0.0625. Batched and singleton results are not bitwise identical.

`context-boundary.json` checks lengths 262144, 65536 and 4096 using explicitly
zero prior KV/state and shared read-only zero pages. Maximum KL is 0.013991.
This verifies last-position indexing; it does not establish full-context quality.

`long-context.json` runs a real 16384-token prompt. Prefill chunks 1024 and 512
produce identical eight-token continuations after fixing BF16 state writeback
to internal 64-token boundaries. Prefill measured 465 and 442 tokens/s.
This checks aligned chunk continuity on one prompt, not broad long-context quality.


## Native DSpine and autotuned verification

The DSpine round is a native CUDA Graph using a five-layer BF16 draft, the
upstream code bridge/serial refinement, FP8-weight/BF16-activation target GEMM,
and ReplaySSM compact verify/commit. See [build and serving](../docs/dspine.md).
`replayssm.json` checks an independent direct FP32 recurrence, accepted prefixes
1..16, null slots and verify preserving committed state. Maximum verify output
error is 0.000031 and committed BF16 state error is 0.000977. A whole-model
Compute Sanitizer smoke run reports zero memory errors.

`gemm-tuning.json` records Triton autotune over 32 configurations and 45 M/N/K
shapes on RTX PRO 5000 72GB. Selected configurations were independently checked
against a true FP32 GEMM of BF16 activations and BF16-decoded weights. The AOT
build carries the pointer/dimension divisibility attributes inferred by JIT.
BF16 activations remain unchanged; no activation quantization was added.

At batch 1, the FP8 projection GEMMs perform 778.5 GFLOP per 16-row verify.
Unique weight/scale traffic is 24.33 GB: the 1344 GB/s bandwidth roofline is
18.1 ms / 43.0 effective TFLOP/s. Initial HES timing was 107.5 ms; after tuning,
`dspine-hes.json` reports about 22.5 ms for these GEMMs, about 80% of this ideal
weight-read roofline. This is useful-work accounting, not DRAM counters.
The complete round averages about 37.9 ms on the short GSM8K contexts; draft
and state materialization take about 6.5 ms and 0.4 ms. `dspine-native.json`
also records an identical 64-token continuation against plain decode.

### Small GSM8K serving sweep

The earlier comparison against per-sequence GEMV is withdrawn. Both arms now
use autotuned Tensor Core GEMM at every batch size, retaining BF16 activations.
`gsm8k-dspine.json` records the replacement run: lm-eval 0.4.11 first eight
GSM8K test questions, 8-shot chat, thinking, greedy, max_tokens=32768 and
identical warmed prompts. All phases reuse the same eight questions. Results
are total completion tokens divided by end-to-end HTTP wall time:

| Concurrency | Plain tokens/s | DSpine tokens/s | Throughput ratio | Plain strict | DSpine strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.6 | 136.0 | 3.52x | 7/8 | 8/8 |
| 2 | 65.8 | 212.5 | 3.23x | 6/8 | 8/8 |
| 4 | 106.4 | 266.1 | 2.50x | 8/8 | 8/8 |
| 8 | 99.8 | 292.6 | 2.93x | 8/8 | 8/8 |

No response was truncated. Standard flexible extraction scores are 7/8, 7/8,
8/8 and 8/8 for plain decode, and 8/8 for every DSpine phase. The failures
calculate the correct numeric result but omit the `####` marker; the 160-minute
answer also ends in an hours/minutes representation, misleading the flexible
last-number extractor. These eight questions cannot establish broad accuracy.

The finite batch drains as requests finish. At concurrency 8, the plain arm's
longest response has 923 tokens, so the tail becomes single-sequence execution.
`decode-hes.json` separately reports ordinary GPU decode at about 39.6, 74.6,
139.8 and 281.3 tokens/s for live batches 1, 2, 4 and 8. Mean step time is
about 25.2-28.6 ms: weight reads are shared across rows. HTTP throughput above
includes partial batches, prefix restoration and drain time. Responses can
have different wording/lengths across arms; complete times and token counts
are retained rather than assuming identical generation lengths.

`flashinfer-gemm.json` separately measures the pinned FlashInfer CUTLASS W8A8
candidate at M=16/32/64/128, N=5120, K=17408. GEMM takes about 119 us, excluding
activation quantization. Random-data RMS error against BF16 inputs is about
2.6%; this candidate is not enabled in the serving bundle.
