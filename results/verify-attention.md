# FlashInfer split-KV target verification

Measured on 2026-11-09 with one RTX PRO 5000 72GB (SM120), using the pinned
FlashInfer headers and existing AOT prefill and merge cubins. The implementation
adds only plan expansion and manifest wiring. Draft attention is unchanged.

## Operator checks

The generated AOT operator was compared with the previous Triton implementation
using shuffled physical pages, different per-request lengths, non-unit K/V
scales and CUDA Graph replay. All 11 cases passed `atol=0.002, rtol=0.02` output
comparison and finite-output checks. These are isolated single-layer timings,
including the merge, excluding the once-per-round planner.

| Batch | KV tokens | Previous ms | FlashInfer ms | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 0.0677 | 0.0221 | 3.06x |
| 1 | 8192 | 0.5191 | 0.0572 | 9.07x |
| 1 | 32768 | 2.1767 | 0.1815 | 11.99x |
| 1 | 131072 | 10.9414 | 0.6906 | 15.84x |
| 2 | 32768 | 2.0893 | 0.2988 | 6.99x |
| 8 | 8193 | 0.8012 | 0.3233 | 2.48x |
| 16 | 4097 | 0.7906 | 0.3414 | 2.32x |

At length 16, the extra merge makes the operator 1.8us slower. Maximum absolute
error is 0.00390625 at that short case (within relative tolerance); at 32K it is
0.0000610352. See [all operator cases](verify-attention-flashinfer.json).

## Complete-model gates

`kern verify` passed. `kern test` A/B gates passed with 512- and 8192-token
prefills, chunk 1024, 16 requested decode steps and KL limit 0.01:

- 512-token case: maximum end-to-end KL 0.000500, two argmax flips within limit.
- 8192-token case: maximum KL 0.000170, three argmax flips within limit.
- At 8192, captured round time fell from 48.912ms to 38.718ms (20.8%). These
  harness round timings are not an estimate of accepted output token throughput.

A 32768-token A/B harness attempt ran out of memory while saving prefill state
images. The long operator checks and a 32K public HTTP request passed; a full
32K teacher-forced logit comparison has not been completed.
See [A/B summaries](verify-attention-ab.json).

The restored service passed ordinary chat, SSE, tool calls, fixed-layout seed
replay, changed seeds, 16 simultaneous greedy requests and 16 mixed-sampling
requests. See [HTTP gate](verify-attention-http.json). A public request with
32042 input tokens completed 256 output tokens successfully; its 15.65s total
includes cold prefill, public transport and concurrent regression requests,
and must not be interpreted as decode-only throughput.

A subsequent HES trace contained 30 single-sequence rounds at KV lengths
32142–32311. Mean full round time was 41.42ms; all 16 FlashInfer target attention
calls plus merges totaled 3.16ms per round. Earlier real-traffic traces at
32768–34904 KV tokens measured 83.59ms total and 45.64ms target attention. These
are different prompts and KV ranges, not a controlled before/after speedup.

To reproduce the operator check, preserve the previous bundle as
`out/qwen38-dspine-triton` before regenerating, then run with a local kern wheel:

```sh
uv run --with /path/to/kern_llm.whl scripts/verify_attention_check.py
```

This change does not establish long-run quality, a 262K context gate or a broad
coding benchmark. GPU KV cache is lost when the service restarts.
