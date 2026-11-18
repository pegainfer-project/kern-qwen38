# kern-qwen38

Qwen3.8-27B text generation on one RTX PRO 5000 72GB Blackwell (SM120),
with FP8 weights, calibrated FP8 KV cache and BF16 Gated DeltaNet state.
The native kern bundle runs chunked prefill and greedy CUDA Graph decode.
It supports up to 16 concurrent sequences and the checkpoint's 262,144-token
context limit. The optional [DSpine bundle](docs/dspine.md) adds native
16-row speculative rounds; the target checkpoint's own MTP head is not connected.

This repository contains its own sources, generator, build scripts and GPU gates.
Outside `sm103/`, it does not read sibling repository directories. FlashInfer is a pinned submodule
under `third_party/flashinfer`. Operator origins are
[kern-qwen3](https://github.com/pegainfer-project/kern-qwen3),
[kern](https://github.com/pegainfer-project/kern) and
[pegainfer](https://github.com/pegainfer-project/pegainfer); copied/adapted source
revisions are recorded in `upstreams.json`.

## Build and run

Requirements: an SM120 GPU, CUDA toolkit supporting SM120, Python 3.12 and `uv`.
Build or obtain the separate [kern runtime](https://github.com/pegainfer-project/kern);
the tested revision is `b118a55c55f236a37046f54f066ecb8e6c12caff` (manifest schema 7).
The compiler/runtime do not require a local checkout of another model repository.

```sh
git submodule update --init
uv sync
uv run scripts/build.py
uv run scripts/build_dsl.py --gemm-tuning results/gemm-tuning.json
uv run scripts/gen.py --checkpoint /path/to/Qwen3.8-27B-FP8 \
  --kv-dtype fp8 --kv-scales results/kv-scales.json
KERN=/path/to/kern/target/release/kern
"$KERN" verify out/qwen38/model.json
"$KERN" run --config out/qwen38/kern.toml \
  --prompt 'The capital of France is' --steps 32
```

Use the official [FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
The generated bundle includes its cubins and a local `kern.toml` pointing to the
checkpoint supplied on the command line. Raw prompts have no chat template.
`--context` and `--chunk` on the generator set context capacity and prefill bound
(defaults: checkpoint maximum 262144 and chunk 1024). `kern run --chunk 64` selects a smaller prefill chunk.
`--kv-dtype bf16` generates a KV reference bundle without calibration scales.

The included KV factors come from four short English/Chinese/code calibration
prompts with headroom. They are an initial usable profile, not evidence of
long-context quality on every workload. Recalibrate and evaluate for deployment.

## HTTP serving

Use explicit executable paths; no global installation is needed. `kern server`
launches `kern-serve` beside the kern executable, or uses `KERN_SERVE_BIN` when
set to another explicit path.

```sh
KERN=/path/to/kern/target/release/kern
"$KERN" server --config out/qwen38/kern.toml qwen38 \
  --port 8000 --max-seqs 16 --served-model-name qwen3.8-27b
```

```sh
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
uv run scripts/http_check.py
```

`/v1/models`, completions, chat completions and SSE are validated. Chat supports
`chat_template_kwargs: {"enable_thinking": false}`. Sampling supports `temperature`, `top_k`, `top_p`, `min_p`, and `seed`;
`temperature: 0` selects greedy decoding. DSpine independently samples each target
verification row and accepts the matching draft prefix, then emits the first
mismatching target token (or the bonus token). It uses token matching without
rejection sampling; stochastic requests can have shorter accepted prefixes.
Seed replay is guaranteed for a fixed batch layout; the FlashInfer RNG also uses
the row index, so changing batch composition can change sampled continuations. Up to 16 HTTP requests run together. Omit `--capacity` to use the available
device memory for the shared KV/state pool. The 262144 limit is per request;
16 full-length contexts do not fit simultaneously in 72GB. Requests exceeding
the available shared pool wait for admission.
`results/http.json` records CLI agreement, queue isolation and streaming checks.

## Precision

- Weights: checkpoint E4M3 with BF16 block128 dequantization factors; exclusions
  such as embeddings, LM head, normalization and gate projections remain BF16.
- Prefill projections: native FP8 Tensor Core GEMM, dynamic per-row/per-128-value
  activation quantization. Decode keeps BF16 activations and uses autotuned Tensor Core GEMM at
  every batch size, with BF16-decoded weight tiles.
- Full-attention KV: E4M3 with fixed, separate K/V scales per layer. Attention
  folds K scale into softmax scaling and applies V scale to the output.
- GDN convolution history and persistent matrix state: BF16. Prefill chunk
  arithmetic and decode state updates accumulate in FP32, with BF16 state writeback.
- Residual activations: BF16. Norm/softmax reductions: FP32.

The checkpoint defaults to FP32 recurrent storage. Our BF16 policy is an explicit
runtime choice. Prefill rounds it at fixed internal 64-token boundaries; decode writes it
every step. Use aligned prefill chunks (the default is 1024). The current gates cover short reference agreement and chunk continuity,
not long-run recurrent drift.

FP8 KV payload is 32 KiB/token. BF16 matrix state is 72 MiB/sequence, plus
2.8125 MiB of convolution history, excluding alignment and pool reservations.
Weights resident on device occupy about 27.4 GiB. See
[operator contracts](docs/operators.md) and [results](results/README.md).

## Verification

```sh
uv run scripts/inspect_checkpoint.py --checkpoint /path/to/Qwen3.8-27B-FP8
uv run scripts/check.py
uv run scripts/check.py --sanitize
uv run scripts/reference.py --checkpoint /path/to/Qwen3.8-27B-FP8 --out out/reference
uv run scripts/e2e.py --kern /path/to/kern --bundle out/qwen38 \
  --reference out/reference --out out/e2e
```

The independent Transformers reference dequantizes the checkpoint to BF16 on GPU
and uses FP32 recurrent state. Run it separately from native kern: both models
cannot coexist in 72GB. It records logits and generates calibration factors.
Use `out/reference/kv-scales.json` when generating a newly calibrated bundle.

The AOT build pins Triton 3.8 because the migrated GDN chunk kernels use block
pointers removed in Triton 3.9. Triton is a build dependency, not a serving-time
JIT. FlashInfer is pinned to main commit `023a8474e1f7469572c0d4284535fc6ffa699574`.

Batch and context gates use kern's public Python runtime and externally supplied
cache memory. Supply a wheel built from the tested runtime revision with `uv run
--with /path/to/kern_llm.whl scripts/batch_check.py ...`; the wheel is a test-only
dependency and is not installed globally. `--boundary-only` checks the last valid
context position with explicitly supplied zero prior KV/state; it tests indexing,
not full-length model quality. `scripts/long_check.py` exercises real 16K prefill.

## GB300 (SM103)

[`sm103/`](sm103/) is the earlier BF16 line for one GB300, with the DFlash2
draft and the manifests hosted by vLLM and SGLang: kernels captured from vLLM,
published as [Pegainfer/kern-qwen38-sm103](https://huggingface.co/Pegainfer/kern-qwen38-sm103).
It moved here from the kern repository and generates against a kern checkout.
