#!/usr/bin/env python3
"""Inspect checkpoint geometry and indexed safetensors without loading tensors."""

import argparse
import json
import struct
import tomllib
from collections import Counter
from pathlib import Path


def inspect(root):
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    quant = config.get("quantization_config", text.get("quantization_config", {}))
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("expected official block-FP8 checkpoint with 128x128 scales")
    layers = text["layer_types"]
    if len(layers) != text["num_hidden_layers"]:
        raise ValueError("layer_types does not cover all layers")
    if set(layers) - {"full_attention", "linear_attention"}:
        raise ValueError("unsupported attention type")
    policy = tomllib.loads((Path(__file__).resolve().parents[1] / "precision.toml").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {}
    for name, filename in index.items():
        if Path(filename).name != filename:
            raise ValueError("expected flat checkpoint filenames")
        expected.setdefault(filename, set()).add(name)
    problems, dtypes = [], Counter()
    for filename, names in sorted(expected.items()):
        path = root / filename
        if not path.exists():
            problems.append(f"missing: {filename}")
            continue
        try:
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < header_size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
                    raise ValueError("invalid header length")
                tensors = json.loads(stream.read(header_size))
            tensors.pop("__metadata__", None)
            if names - tensors.keys():
                raise ValueError("indexed tensors absent from header")
            data_size = path.stat().st_size - 8 - header_size
            end = 0
            for tensor in tensors.values():
                lo, hi = tensor["data_offsets"]
                if not 0 <= lo <= hi <= data_size:
                    raise ValueError("invalid tensor offsets")
                end = max(end, hi)
                dtypes[tensor["dtype"]] += 1
            if end != data_size:
                raise ValueError("file size does not match tensor payload")
        except (ValueError, KeyError, struct.error) as exc:
            problems.append(f"invalid: {filename}: {exc}")
    full, recurrent = layers.count("full_attention"), layers.count("linear_attention")
    kv = full * 2 * text["num_key_value_heads"] * text["head_dim"]
    ssm = recurrent * text["linear_num_value_heads"] * text["linear_key_head_dim"] * text["linear_value_head_dim"]
    conv_dim = (2 * text["linear_num_key_heads"] * text["linear_key_head_dim"]
                + text["linear_num_value_heads"] * text["linear_value_head_dim"])
    report = {
        "model_type": config["model_type"],
        "full_attention_layers": full,
        "recurrent_layers": recurrent,
        "precision": policy,
        "checkpoint_ssm_dtype": text.get("mamba_ssm_dtype"),
        "payload_bytes": {
            "fp8_kv_per_token": kv,
            "bf16_kv_per_token": kv * 2,
            "bf16_ssm_per_sequence": ssm * 2,
            "fp32_ssm_per_sequence": ssm * 4,
            "bf16_conv_per_sequence": recurrent * conv_dim * (text["linear_conv_kernel_dim"] - 1) * 2,
        },
        "indexed_files": len(expected),
        "tensor_dtype_counts": dict(dtypes),
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return bool(problems)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(inspect(args.checkpoint))
