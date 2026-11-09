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
(with thinking disabled), SSE equivalence and 128 concurrent requests at
`--max-seqs 128` with automatic shared-pool capacity. Greedy completion matches the CLI output. Unsupported stochastic
sampling returns HTTP 400. See `http.json`; run `uv run scripts/http_check.py`
against the local server to repeat the gate.


## Batch and context validation

`batch.json` compares batch 1, 3, 8, 9, 17 and 128 against singleton execution
for two consecutive steps, restoring identical KV/recurrent state before each
comparison. Physical KV pages are shuffled. Maximum logits KL is 0.004903;
batch 128 top-1 agreement is 124/128 and 126/128. Mismatches have singleton
top-two margins of at most 0.0625. Batched and singleton results are not bitwise identical.

`context-boundary.json` checks lengths 262144, 65536 and 4096 using explicitly
zero prior KV/state and shared read-only zero pages. Maximum KL is 0.013991.
This verifies last-position indexing; it does not establish full-context quality.

`long-context.json` runs a real 16384-token prompt. Prefill chunks 1024 and 512
produce identical eight-token continuations after fixing BF16 state writeback
to internal 64-token boundaries. Prefill measured 465 and 442 tokens/s.
This checks aligned chunk continuity on one prompt, not broad long-context quality.
