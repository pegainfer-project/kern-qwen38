#!/usr/bin/env python3
"""Build a combined Qwen3.8 DSpine optimization bundle (or ablation subset) from source."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import triton
from build_dsl import load
from dspine_attention import compile_aot as compile_draft_attention_aot
from dspine_build import specifications as dspine_specifications
from gemm_manifest import prefill_linear
from prefill_fp8 import compile_native
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]

ALL_FEATURES = (
    "verify-w8a8",
    "prefill-fp8",
    "replay-spill",
    "verify-fp8",
    "dspine-attention",
)

VERIFY_ROWS = (16, 32, 64, 128, 256)
VERIFY_GEOMETRIES = ((34816, 5120), (5120, 17408))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_verify_w8a8(
    m: dict,
    base_manifest: dict,
    dst_kernels: Path,
    tuning_json: Path,
    w8a8_min_seqs: int,
    w8a8_ops: tuple[str, ...],
):
    gemm = load("fp8_gemm")
    tuning_entries = json.loads(tuning_json.read_text())
    quant_launch = m["ops"]["gemm_34816_5120"]["impl"]["launches"][0]
    dsl = {"fp8_act_quant": {**quant_launch, "abi_tail": ["i64", "i64"]}}
    selected = {}

    cfg_re = re.compile(
        r"BM:\s*(\d+),\s*BN:\s*(\d+),\s*BK:\s*(\d+),\s*num_warps:\s*(\d+),\s*num_ctas:\s*1,\s*num_stages:\s*(\d+)"
    )
    signature = dict(
        zip(
            gemm.fp8_w8a8_gemm.arg_names[:8],
            ("*fp8e4nv", "*fp32", "*fp8e4nv", "*bf16", "*bf16", "i32", "i32", "i32"),
            strict=True,
        )
    )
    for entry in tuning_entries:
        match = cfg_re.search(entry["config"])
        if not match:
            raise ValueError(f"Unrecognized tuning config format: {entry['config']}")
        bm, bn, bk, warps, stages = map(int, match.groups())
        name = f"verify_w8a8_{bm}_{bn}_{warps}_{stages}"
        if name not in dsl:
            compiled = triton.compile(
                ASTSource(
                    gemm.fp8_w8a8_gemm,
                    signature=signature,
                    constexprs={"BM": bm, "BN": bn, "BK": bk},
                    attrs={(j,): [["tt.divisibility", 16]] for j in (0, 1, 2, 3, 4, 6, 7)},
                ),
                target=GPUTarget("cuda", 120, 32),
                options={"num_warps": warps, "num_stages": stages},
            )
            if (
                compiled.metadata.global_scratch_size
                or compiled.metadata.profile_scratch_size
                or compiled.metadata.num_ctas != 1
            ):
                raise ValueError(f"Unsupported AOT launch: {name}")
            path = dst_kernels / (name + ".cubin")
            path.write_bytes(compiled.asm["cubin"])
            m["modules"][name] = {"source": path.name, "sha256": sha256_file(path)}
            dsl[name] = {
                "entry": compiled.name,
                "abi_tail": ["i64", "i64"],
                "block": [32 * warps, 1, 1],
                "shared_mem": compiled.metadata.shared,
            }
        selected[f"{entry['rows']},{entry['n']},{entry['k']}"] = {"module": name, "bm": bm, "bn": bn}

    for n, k in VERIFY_GEOMETRIES:
        op_name = f"verify_linear_{n}_{k}"
        if op_name not in w8a8_ops:
            continue
        w_op = prefill_linear(n, k, selected, dsl, lambda mod: mod, buckets=VERIFY_ROWS)
        if w8a8_min_seqs <= 1:
            m["ops"][op_name] = w_op
        else:
            cutoff_tokens = (w8a8_min_seqs - 1) * 16
            b_op = copy.deepcopy(base_manifest["ops"][op_name])
            kept_bf16 = []
            for launch in b_op["impl"]["launches"]:
                w_clause = launch["when"]
                if w_clause["var"] != "seqs":
                    raise ValueError(f"Expected seqs dispatch in baseline {op_name}")
                if w_clause["min"] < w8a8_min_seqs:
                    c = copy.deepcopy(launch)
                    c["when"] = {
                        "var": "tokens",
                        "min": (w_clause["min"] - 1) * 16 + 1,
                        "max": min(w_clause["max"], w8a8_min_seqs - 1) * 16,
                    }
                    kept_bf16.append(c)
            w8a8_launches = []
            for idx, launch in enumerate(w_op["impl"]["launches"]):
                c = copy.deepcopy(launch)
                if idx == 0:
                    c["when"] = {"var": "tokens", "min": cutoff_tokens + 1}
                    w8a8_launches.append(c)
                else:
                    w_clause = c["when"]
                    max_tok = w_clause.get("max")
                    if max_tok is not None and max_tok <= cutoff_tokens:
                        continue
                    new_when = {"var": "tokens", "min": max(w_clause["min"], cutoff_tokens + 1)}
                    if max_tok is not None:
                        new_when["max"] = max_tok
                    c["when"] = new_when
                    w8a8_launches.append(c)
            m["ops"][op_name] = {
                "params": w_op["params"],
                "impl": {
                    "scratch": w_op["impl"]["scratch"],
                    "launches": kept_bf16 + w8a8_launches,
                },
            }


def apply_native_fp8_attention(
    m: dict,
    dst_kernels: Path,
    enable_prefill: bool,
    enable_verify: bool,
    verify_splits: int = 8,
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        module, (original_size, size, scale_at, shared) = compile_native(tmp_dir, tmp_dir)
        shutil.copyfile(tmp_dir / module["source"], dst_kernels / module["source"])
    m["modules"]["prefill_fp8"] = module

    if enable_prefill:
        tiles = {"ceil_div": [{"mul": ["tokens", 6]}, 16]}
        maximum = m["vars"]["tokens"]["max"]
        for name in ("plan.request", "plan.tile", "plan.zero"):
            m["buffers"][name]["shape"][0] = max(m["buffers"][name]["shape"][0], (maximum * 6 + 15) // 16)
        plan = m["ops"]["prefill_plan"]["impl"]["launches"][0]
        plan.update(module="prefill_fp8", entry="prefill_plan_fp8")
        for name, op in m["ops"].items():
            if not name.endswith(".attention_prefill"):
                continue
            rows = {"param": len(op["params"])}
            op["params"].append("i32")
            launch = op["impl"]["launches"][0]
            pack = launch["args"][0]["pack"]
            if pack["size"] != original_size:
                raise ValueError("Baseline prefill attention ABI differs from compiled headers")
            pack["size"] = size
            fields = pack["fields"]
            for field in fields:
                if field.get("param") == 0:
                    del field["param"]
                    field["scratch"] = "q8"
            fields.append({"at": scale_at, "scratch": "scales"})
            for field in fields:
                if field.get("expr") == {"ceil_div": [{"mul": ["tokens", 6]}, 64]}:
                    field["expr"] = tiles
            launch.update(
                module="prefill_fp8",
                entry="prefill_fp8",
                grid=[tiles, 1, 4],
                block=[32, 1, 4],
                shared_mem=shared,
            )
            launch["params"] = [f"bytes<{size}>", "i32"]
            launch["args"] += [rows]
            quant = {
                "module": "prefill_fp8",
                "entry": "quantize_q",
                "params": ["in buffer<bf16>", "out buffer<fp8e4m3>", "out buffer<f32>", "i32"],
                "args": [{"param": 0}, {"scratch": "q8"}, {"scratch": "scales"}, rows],
                "grid": [{"mul": ["tokens", 6]}, 1, 1],
                "block": [128, 1, 1],
            }
            op["impl"]["scratch"] = {
                "q8": {"dtype": "fp8e4m3", "shape": ["tokens", 6144]},
                "scales": {"dtype": "f32", "shape": ["tokens", 24]},
            }
            op["impl"]["launches"] = [quant, launch]
        for program in m["programs"].values():
            for call in program["calls"]:
                if call["op"].endswith(".attention_prefill"):
                    call["args"].append({"var": "tokens"})

    if enable_verify:
        groups = m["vars"]["seqs"]["max"]
        v_tiles = 6
        for name in ("plan.request", "plan.tile", "plan.zero", "plan.valid"):
            m["buffers"][name]["shape"][0] = max(m["buffers"][name]["shape"][0], groups * verify_splits)
        for name in ("request", "kv_tile", "qo_tile", "valid"):
            m["buffers"]["verify.attn." + name]["shape"][0] = groups * verify_splits * v_tiles
        for name in ("partial", "lse"):
            m["buffers"]["verify.attn." + name]["shape"][0] = groups * verify_splits * 16
        v_plan = copy.deepcopy(m["ops"]["batch_plan"])
        v_plan["impl"]["launches"][0]["args"][13] = {"i32": verify_splits}
        v_plan["impl"]["launches"][0]["args"][15] = {"i32": groups * verify_splits}
        m["ops"]["verify_kv_plan"] = v_plan
        m["programs"]["round"]["calls"][0]["op"] = "verify_kv_plan"
        expand = m["ops"]["verify_attention_plan"]["impl"]["launches"][0]
        expand["args"][-1] = {"i32": verify_splits}
        expand.update(module="prefill_fp8", entry="verify_plan_fp8")
        op = m["ops"]["verify_attention_flashinfer"]
        launch = op["impl"]["launches"][0]
        field_list = launch["args"][0]["pack"]["fields"]
        for field in field_list:
            if field.get("expr") == {"mul": ["seqs", 64]}:
                field["expr"] = {"mul": ["seqs", verify_splits * v_tiles]}
        launch["grid"][0] = {"mul": ["seqs", verify_splits * v_tiles]}
        pack = launch["args"][0]["pack"]
        if pack["size"] != original_size:
            raise ValueError("Baseline verify attention ABI differs from compiled headers")
        pack["size"] = size
        for field in field_list:
            if field.get("param") == 0:
                del field["param"]
                field["scratch"] = "q8"
        field_list.append({"at": scale_at, "scratch": "scales"})
        launch.update(
            module="prefill_fp8",
            entry="prefill_fp8",
            params=[f"bytes<{size}>", "i32"],
            block=[32, 1, 4],
            shared_mem=shared,
        )
        launch["args"].append({"param": 18})
        quant = {
            "module": "prefill_fp8",
            "entry": "quantize_q",
            "params": ["in buffer<bf16>", "out buffer<fp8e4m3>", "out buffer<f32>", "i32"],
            "args": [{"param": 0}, {"scratch": "q8"}, {"scratch": "scales"}, {"param": 18}],
            "grid": [{"mul": ["tokens", 6]}, 1, 1],
            "block": [128, 1, 1],
        }
        op["impl"]["scratch"] = {
            "q8": {"dtype": "fp8e4m3", "shape": ["tokens", 6144]},
            "scales": {"dtype": "f32", "shape": ["tokens", 24]},
        }
        op["impl"]["launches"].insert(0, quant)


def apply_replay_spill(m: dict, dst_kernels: Path):
    specs = {item[0]: item for item in dspine_specifications(load)}
    name, kernel, signature, warps, extra_const = specs["replay_verify"]
    sig, constants = {}, {}
    for arg, typ in zip(kernel.arg_names, signature.split(","), strict=True):
        if typ.isdecimal():
            constants[arg] = int(typ)
        else:
            sig[arg] = typ
    constants.update(extra_const)
    compiled = triton.compile(
        ASTSource(kernel, signature=sig, constexprs=constants),
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": warps, "num_stages": 2, "enable_fp_fusion": True},
    )
    if (
        compiled.metadata.global_scratch_size
        or compiled.metadata.profile_scratch_size
        or compiled.metadata.num_ctas != 1
    ):
        raise ValueError("replay_verify: unsupported scratch or cluster requirements")
    path = dst_kernels / "replay_verify.cubin"
    path.write_bytes(compiled.asm["cubin"])
    m["modules"][name] = {"source": path.name, "sha256": sha256_file(path)}
    launch = m["ops"]["replay_verify"]["impl"]["launches"][0]
    launch["entry"] = compiled.name
    launch["block"] = [32 * warps, 1, 1]
    launch["shared_mem"] = compiled.metadata.shared


def apply_dspine_attention(
    m: dict,
    dst_kernels: Path,
    splits: int = 8,
    bk: int = 32,
    split_warps: int = 4,
    reduce_warps: int = 8,
    stages: int = 2,
):
    io = load("dspine_io")
    split_meta = compile_draft_attention_aot(
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
        dst_kernels / "dspine_attention_split.cubin",
    )
    reduce_meta = compile_draft_attention_aot(
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
        dst_kernels / "dspine_attention_reduce.cubin",
    )
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


def build(
    baseline: Path,
    out: Path,
    enabled: list[str],
    w8a8_min_seqs: int = 2,
    w8a8_ops: tuple[str, ...] = ("verify_linear_34816_5120", "verify_linear_5120_17408"),
    tuning_json: Path = ROOT / "results/verify-w8a8-tuning.json",
    kern_bin: Path | None = None,
):
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")
    unknown = set(enabled) - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")

    base_manifest = json.loads((baseline / "model.json").read_text())
    if (
        base_manifest["states"]["kv"].get("bytes_per_token") != 32768
        or base_manifest["buffers"]["q"]["dtype"] != "bf16"
        or base_manifest["buffers"]["q"]["shape"] != ["tokens", 6144]
    ):
        raise ValueError("Expected Qwen3.8 BF16 queries and calibrated FP8 KV baseline")

    out.mkdir(parents=True)
    dst_kernels = out / "kernels"
    shutil.copytree(baseline / "kernels", dst_kernels)
    m = copy.deepcopy(base_manifest)

    if "verify-w8a8" in enabled:
        apply_verify_w8a8(m, base_manifest, dst_kernels, tuning_json, w8a8_min_seqs, w8a8_ops)
    if "prefill-fp8" in enabled or "verify-fp8" in enabled:
        apply_native_fp8_attention(
            m,
            dst_kernels,
            enable_prefill="prefill-fp8" in enabled,
            enable_verify="verify-fp8" in enabled,
        )
    if "replay-spill" in enabled:
        apply_replay_spill(m, dst_kernels)
    if "dspine-attention" in enabled:
        apply_dspine_attention(m, dst_kernels)

    used_modules = {
        launch["module"]
        for op in m["ops"].values()
        for launch in op["impl"].get("launches", [])
        if "module" in launch
    }
    m["modules"] = {name: mod for name, mod in m["modules"].items() if name in used_modules}
    used_sources = {mod["source"] for mod in m["modules"].values()}
    for cubin in list(dst_kernels.iterdir()):
        if cubin.name not in used_sources:
            cubin.unlink()

    for name, mod in m["modules"].items():
        cubin_path = dst_kernels / mod["source"]
        if not cubin_path.exists() or sha256_file(cubin_path) != mod["sha256"]:
            raise ValueError(f"Cubin verification failed for module {name}")

    manifest_path = out / "model.json"
    manifest_path.write_text(json.dumps(m, indent=2) + "\n")

    config = tomllib.loads((baseline / "kern.toml").read_text())["targets"]["qwen38"]
    weights = [str((baseline / p).resolve()) if not Path(p).is_absolute() else p for p in config["weights"]]
    (out / "kern.toml").write_text(
        '[targets.qwen38]\nmanifest = "model.json"\nkernels = "kernels"\nweights = '
        + json.dumps(weights)
        + "\n"
    )

    if kern_bin is not None:
        subprocess.run([str(kern_bin), "verify", str(manifest_path)], check=True)
    print(manifest_path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--enable", nargs="*", default=list(ALL_FEATURES))
    p.add_argument("--disable", nargs="*", default=[])
    p.add_argument(
        "--w8a8-min-seqs",
        type=int,
        default=2,
        help="Minimum sequence batch size using W8A8 verification MLPs (default: 2; 1 enables batch 1)",
    )
    p.add_argument(
        "--w8a8-ops",
        nargs="+",
        default=["verify_linear_34816_5120", "verify_linear_5120_17408"],
    )
    p.add_argument("--tuning", type=Path, default=ROOT / "results/verify-w8a8-tuning.json")
    p.add_argument("--kern", type=Path)
    args = p.parse_args()
    selected = [f for f in args.enable if f not in set(args.disable)]
    build(
        args.baseline,
        args.out,
        selected,
        args.w8a8_min_seqs,
        tuple(args.w8a8_ops),
        args.tuning,
        args.kern,
    )
