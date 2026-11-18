# Qwen3.8-27B on GB300 (SM103), from captured kernels

The first Qwen3.8-27B line on kern: BF16, one GB300, with the DFlash2 draft
and the manifests that run as the model inside vLLM and SGLang. It is what
[Pegainfer/kern-qwen38-sm103](https://huggingface.co/Pegainfer/kern-qwen38-sm103)
publishes. It shares no source with the SM120 FP8 bundle in the rest of this
repository: its kernels are captured from vLLM (Triton), NVIDIA's TRTLLM-GEN
attention cubins, and hand-written kernels pinned through kern's kernel index.

This directory moved out of [kern](https://github.com/pegainfer-project/kern)
after its commit [`a0101a6`](https://github.com/pegainfer-project/kern/tree/a0101a6),
where its history is. The layout is kern's (`tools/`, `examples/`, `docs/`,
`kern.toml`), so every command in the scripts and in the bring-up record runs
from here unchanged.

| Path | What |
|---|---|
| `examples/qwen3.8-27b.json`, `examples/qwen3.8-27b-dflash2.json` | The manifests: target alone, and with its DFlash2 draft (`round`). |
| `tools/capture_qwen38.sh`, `tools/capture_qwen38_spec.sh` | Capture vLLM's target and draft launches. |
| `tools/gen_qwen35.py` | Captured launches to a base manifest (Triton attention). |
| `tools/trtllm_attention.py`, `tools/trtllm-gen/` | Full attention on NVIDIA's TRTLLM-GEN kernels, page 64. |
| `tools/qwen38_fuse.py` | The fused GDN decode step, PDL, pruning. |
| `tools/qwen_weights.py`, `tools/qwen_constants.py` | Checkpoint bindings, derived tables, named dimensions. |
| `tools/qwen38_vllm.py`, `tools/qwen38_sglang.py` | The hosted manifests for `kern_vllm` and `kern_sglang`. |
| `tools/test_*.py` | Manifest regression tests (host), kernel checks against vLLM's ops (GPU). |
| `tools/qwen38_ref.py`, `qwen38_compare.py`, `qwen38_margins.py`, `qwen38_probe_*.py`, … | The bring-up's reference, comparison and probe tools. |
| `docs/` | The [bring-up record](docs/qwen38-bringup.md), its [task](docs/qwen38-bringup-prompt.md) and evidence (`docs/qwen38/`), and the [hosted transform](docs/qwen38-vllm.md). |

## Requirements

A kern checkout, named by `KERN`. The generators import its shared tools
(`tools/kern_manifest.py`, `once.py`, `sampling.py`, `kernels/` over
`kernel_index/`), the capture scripts use its `tools/kernel-capture/`, and
`tools/extract_kernels.sh` collects the cubins a manifest pins.

```sh
cd sm103
export KERN=/path/to/kern
```

## Generate

```sh
python3 tools/gen_qwen35.py dumped-kernels/<target capture> base.json
python3 tools/trtllm_attention.py --input base.json \
  --cache-metadata tools/trtllm-gen/cache.json --output trtllm.json
python3 tools/qwen38_fuse.py --input trtllm.json --output fused.json
```

`fused.json` equals `examples/qwen3.8-27b.json` in everything but two
things the checked-in file keeps: its `chat` block, and ten hand-written
modules named as local files (four at an older build) where the generator
emits registry refs. `--spec dumped-kernels/<draft capture>` on the first
step adds the DFlash2 draft. The hosted manifests are pure functions of the example:

```sh
python3 tools/qwen38_vllm.py --output qwen3.8-27b-vllm.json [--sampled]
python3 tools/qwen38_sglang.py --output qwen3.8-27b-sglang.json [--sampled]
```

## Run and check

`kern.toml` names the two targets; their kernels (`kernels-qwen38*`, from
`$KERN/tools/extract_kernels.sh` or the published `cubins/`) and checkpoints
(`weights/`) are local and ignored.

```sh
"$KERN"/target/release/kern run qwen3.8-27b --steps 320
"$KERN"/target/release/kern test qwen3.8-27b --reference <manifest A>   # a change against its baseline
python3 -m unittest discover -s tools -p 'test_qwen_constants.py'
python3 -m unittest discover -s tools -p 'test_trtllm_manifest.py'
```

A manifest names one program per row shape (kern `a0101a6`): the target's
`decode` takes 1 to 128 sequences.
