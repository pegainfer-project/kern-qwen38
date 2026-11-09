#!/usr/bin/env python3
"""Prepare DSpine's greedy lookup tables once, before native runtime loading."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@torch.inference_mode()
def prepare(target, draft, out):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config = json.loads((draft / "config.json").read_text())
    bridge = config["dflash_config"]
    expected = (
        config["block_size"],
        config["num_hidden_layers"],
        bridge["target_layer_ids"],
        bridge["selected_prefix_code_bridge"]["rank"],
    )
    if expected != (16, 5, [1, 16, 31, 46, 61], 512):
        raise ValueError(f"Unexpected DSpine geometry: {expected}")
    if bridge.get("selected_prefix_terminal", True) or bridge.get(
        "selected_prefix_terminal_residual_rank", 0
    ):
        raise ValueError("Only the terminal-disabled Qwen3.8 DSpine checkpoint is supported")
    with safe_open(draft / "model.safetensors", framework="pt", device="cuda") as f:
        get = lambda name: f.get_tensor("selected_prefix_bridge." + name)
        for name in ("independent_write_version", "terminal_disabled_version"):
            if get(name).item() != 1:
                raise ValueError(f"Unsupported DSpine version: {name}")
        if get("out.weight").count_nonzero().item():
            raise ValueError("Terminal-disabled out.weight must be zero")
        packed = torch.cat([get(n) for n in ("gate.weight", "predecessor_gate.weight", "prev.weight")])
        codebook = get("codebook")
        write = get("feature_write_weights")[4].contiguous()
        norm = f.get_tensor("norm.weight")
    index = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(target / index["lm_head.weight"], framework="pt", device="cuda") as f:
        lm = f.get_tensor("lm_head.weight")
    vocab, hidden = lm.shape
    tg = torch.empty((vocab, 512), device="cuda", dtype=torch.bfloat16)
    tsv = torch.empty((vocab, 1024), device="cuda", dtype=torch.bfloat16)
    for start in range(0, vocab, 4096):
        end = min(start + 4096, vocab)
        tg[start:end] = (lm[start:end].float() * norm.float()).bfloat16() @ write
        tsv[start:end] = (32.0 * codebook[start:end].float()).bfloat16() @ packed[512:].T
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.safetensors"
    save_file({"dspine.packed": packed.cpu(), "dspine.tg": tg.cpu(), "dspine.tsv": tsv.cpu()}, path)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "draft_config_sha256": hashlib.sha256((draft / "config.json").read_bytes()).hexdigest(),
                "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dtype": "BF16",
                "tf32": False,
                "vocab": vocab,
                "hidden": hidden,
            },
            indent=2,
        )
        + "\n"
    )
    print(path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/dspine-tables"))
    a = p.parse_args()
    prepare(a.checkpoint, a.draft, a.out)
