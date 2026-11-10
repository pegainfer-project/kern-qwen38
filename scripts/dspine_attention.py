#!/usr/bin/env python3
"""Build an isolated GQA-packed Split-KV draft attention candidate."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import triton
from build_dsl import load
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def compile_aot(kernel, signature, constexprs, warps, stages, path):
    compiled = triton.compile(
        ASTSource(kernel, signature=signature, constexprs=constexprs),
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": warps, "num_stages": stages, "enable_fp_fusion": False},
    )
    if (
        compiled.metadata.global_scratch_size
        or compiled.metadata.profile_scratch_size
        or compiled.metadata.num_ctas != 1
    ):
        raise ValueError(f"{path.stem}: unsupported AOT scratch or cluster requirements")
    cubin = compiled.asm["cubin"]
    path.write_bytes(cubin)
    return {
        "source": path.name,
        "sha256": hashlib.sha256(cubin).hexdigest(),
        "entry": compiled.name,
        "shared_mem": compiled.metadata.shared,
        "block": [32 * warps, 1, 1],
    }


def build(baseline, out, splits=8, bk=32, split_warps=4, reduce_warps=8, stages=2):
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    if splits not in (4, 8, 16) or bk not in (32, 64):
        raise ValueError("Unsupported split count or KV tile size")
    original = json.loads((baseline / "model.json").read_text())
    if (
        "dspine_attention" not in original["ops"]
        or original["states"]["draft_kv"]["bytes_per_seq"] != 5 * 8208 * 2048 * 2
    ):
        raise ValueError("Expected Qwen3.8 DSpine bundle with five-layer 8208-slot draft KV ring")

    out.mkdir(parents=True)
    kernels = out / "kernels"
    shutil.copytree(baseline / "kernels", kernels)
    shutil.copyfile(baseline / "kern.toml", out / "kern.toml")

    io = load("dspine_io")
    split_meta = compile_aot(
        io.draft_attention_split,
        {
            "Q": "*bf16",
            "Cache": "*bf16",
            "Lines": "*i32",
            "Lengths": "*i32",
            "Partial": "*fp32",
            "MaxBuf": "*fp32",
            "SumBuf": "*fp32",
            "KS": "fp32",
        },
        {"SPLITS": splits, "BK": bk, "RING": 8208},
        split_warps,
        stages,
        kernels / "dspine_attention_split.cubin",
    )
    reduce_meta = compile_aot(
        io.draft_attention_reduce,
        {
            "Q": "*bf16",
            "K": "*bf16",
            "V": "*bf16",
            "Lengths": "*i32",
            "Partial": "*fp32",
            "MaxBuf": "*fp32",
            "SumBuf": "*fp32",
            "Output": "*bf16",
            "KS": "fp32",
            "VS": "fp32",
        },
        {"SPLITS": splits, "BK": bk},
        reduce_warps,
        2,
        kernels / "dspine_attention_reduce.cubin",
    )

    m = copy.deepcopy(original)
    m["modules"].pop("dspine_attention", None)
    (kernels / "dspine_attention.cubin").unlink(missing_ok=True)
    for mod_name, meta in (
        ("dspine_attention_split", split_meta),
        ("dspine_attention_reduce", reduce_meta),
    ):
        m["modules"][mod_name] = {"source": meta["source"], "sha256": meta["sha256"]}

    m["ops"]["dspine_attention"]["impl"] = {
        "scratch": {
            "partial": {"dtype": "f32", "shape": ["tokens", splits * 4096]},
            "max_buf": {"dtype": "f32", "shape": ["tokens", splits * 32]},
            "sum_buf": {"dtype": "f32", "shape": ["tokens", splits * 32]},
        },
        "launches": [
            {
                "module": "dspine_attention_split",
                "entry": split_meta["entry"],
                "params": [
                    "in buffer<bf16>",
                    "in state",
                    "in buffer<i32>",
                    "in buffer<i32>",
                    "out buffer<f32>",
                    "out buffer<f32>",
                    "out buffer<f32>",
                    "f32",
                    "i64",
                    "i64",
                ],
                "grid": ["seqs", 8 * splits, 1],
                "block": split_meta["block"],
                "shared_mem": split_meta["shared_mem"],
                "args": [
                    {"param": 0},
                    {"param": 3},
                    {"param": 4},
                    {"param": 6},
                    {"scratch": "partial"},
                    {"scratch": "max_buf"},
                    {"scratch": "sum_buf"},
                    {"param": 9},
                    {"i64": 0},
                    {"i64": 0},
                ],
            },
            {
                "module": "dspine_attention_reduce",
                "entry": reduce_meta["entry"],
                "params": [
                    "in buffer<bf16>",
                    "in buffer<bf16>",
                    "in buffer<bf16>",
                    "in buffer<i32>",
                    "in buffer<f32>",
                    "in buffer<f32>",
                    "in buffer<f32>",
                    "out buffer<bf16>",
                    "f32",
                    "f32",
                    "i64",
                    "i64",
                ],
                "grid": ["seqs", 8, 1],
                "block": reduce_meta["block"],
                "shared_mem": reduce_meta["shared_mem"],
                "args": [
                    {"param": 0},
                    {"param": 1},
                    {"param": 2},
                    {"param": 6},
                    {"scratch": "partial"},
                    {"scratch": "max_buf"},
                    {"scratch": "sum_buf"},
                    {"param": 7},
                    {"param": 9},
                    {"param": 10},
                    {"i64": 0},
                    {"i64": 0},
                ],
            },
        ],
    }

    (out / "model.json").write_text(json.dumps(m, indent=2) + "\n")
    (out / "build-meta.json").write_text(
        json.dumps(
            {
                "splits": splits,
                "bk": bk,
                "split_warps": split_warps,
                "reduce_warps": reduce_warps,
                "stages": stages,
                "split": split_meta,
                "reduce": reduce_meta,
            },
            indent=2,
        )
        + "\n"
    )
    print(out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", type=int, default=8)
    p.add_argument("--bk", type=int, default=32)
    p.add_argument("--split-warps", type=int, default=4)
    p.add_argument("--reduce-warps", type=int, default=8)
    p.add_argument("--stages", type=int, default=2)
    args = p.parse_args()
    build(args.baseline, args.out, args.splits, args.bk, args.split_warps, args.reduce_warps, args.stages)
