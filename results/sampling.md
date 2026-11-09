# Target sampling and DSpine token matching

Validated on 2026-11-09 with one RTX PRO 5000 72GB Blackwell (SM120).
Ordinary decode and DSpine target verification use the FlashInfer sampling chain
adapted from kern-qwen3: temperature, top-k, top-p, min-p and seed. Temperature
zero selects the existing two-pass argmax. The draft lookup stays greedy.

DSpine compares sampled target tokens with draft tokens, accepts the matching
prefix and emits the first differing target token, or the bonus token after
15 matches. It does not use rejection sampling. The per-request parameters are
expanded to 16 verification rows. The runtime supplies the length at the end of
the round; row j uses RNG offset `seq_len - 15 + j`.

## Validation

- Three GPU integration tests passed: filter boundaries and mixed parameters at
  vocabularies 32, 1024 and 248320, repeatable position-dependent draws, CUDA Graph
  replay, and verification parameter/position expansion across distinct requests.
  The 4096-draw distribution check had chi-square 7.3559 (threshold 40).
- Both ordinary and DSpine HTTP paths passed completions, chat, SSE, fixed-layout
  seed replay, changed-seed variation, 16 greedy requests and 16 mixed requests.
  See [ordinary HTTP](sampling-http-plain.json) and [DSpine HTTP](sampling-http-dspine.json).
- Complete-model batched logits were compared against restored singleton states
  at batch sizes 1, 2, 3, 4, 8, 9 and 16. Maximum KL was below 0.002 against the
  0.02 limit. Two batch-9 rows changed top-1 with a 0.0625 logit margin; all other
  rows matched. See [batch results](sampling-batch.json).
- The generated DSpine manifest passed `kern verify`.

## Small acceptance sample

Four concurrent requests used the prompt "Write a Python function that merges two
sorted lists, then explain its complexity." followed by a newline, a 128-token
limit, top-k 20, top-p 0.9, and seeds 123 through 126. Each run emitted 512 tokens.
The greedy request ran first; this is a short acceptance smoke test, not a broad
performance or quality evaluation. Throughput includes HTTP and prefill time.

| Temperature | Mean tokens per round, including target token | Output tokens/s |
|---|---:|---:|
| 0 | 5.13 | 264.1 |
| 0.8 | 4.77 | 226.8 |

See [raw counters](sampling-acceptance.json). Acceptance depends on the workload
and sampling settings. FlashInfer's RNG includes the flattened row index, so seed
replay assumes a fixed batch layout; ordinary and DSpine output need not match
for the same seed. BF16 recurrent-state rounding still occurs at accepted round
boundaries. These checks do not establish long-context or long-run quality.

To repeat the GPU tests with a locally built kern Python wheel:

```sh
uv run --with /path/to/kern_llm.whl tests/test_sampling.py
uv run scripts/http_check.py
```
