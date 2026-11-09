#!/usr/bin/env python3
"""Compile local Triton operators to standalone cubins for kern; no JIT at serving time."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "source" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(dspine=False, gemm_tuning=None):
    out = ROOT / "build/sm_120"
    out.mkdir(parents=True, exist_ok=True)
    gdn = load("gdn_prefill")
    gemm = load("fp8_gemm")
    specs = [
        ("fp8_block_gemm", gemm.fp8_block_gemm, "*bf16,*fp8e4nv,*bf16,*bf16,i32,i32,i32,32,64,128,1", 4),
        ("fp8_act_quant", gemm.fp8_act_quant, "*bf16,*fp8e4nv,*fp32,i32,i32,16", 4),
        (
            "gdn_prepare",
            gdn.gdr_prepare_qkv_gbeta_qwen35_kernel,
            "*bf16,*bf16,*bf16,*bf16,*fp32,*bf16,*bf16,*bf16,*fp32,*fp32,i32,i32,i32,i32,128,128",
            4,
        ),
        ("gdn_cumsum", gdn.gdr_chunk_local_cumsum_qwen35_kernel, "*fp32,*fp32,i32,i32,64", 1),
        (
            "gdn_kkt",
            gdn.gdr_chunk_scaled_dot_kkt_qwen35_kernel,
            "*bf16,*fp32,*fp32,*fp32,i32,i32,64,64,128",
            4,
        ),
        ("gdn_solve", gdn.gdr_solve_tril_64_qwen35_kernel, "*fp32,*bf16,i32,i32", 4),
        (
            "gdn_recompute",
            gdn.gdr_recompute_w_u_qwen35_kernel,
            "*bf16,*bf16,*fp32,*bf16,*bf16,*bf16,*fp32,i32,i32,128,128,64,64,64",
            4,
        ),
        (
            "gdn_state",
            gdn.gdr_chunk_state_qwen35_kernel,
            "*bf16,*bf16,*bf16,*fp32,*fp32,*fp32,*bf16,*fp32,i32,i32,32,64,128,128,64",
            4,
        ),
        (
            "gdn_output",
            gdn.gdr_chunk_o_qwen35_kernel,
            "*bf16,*bf16,*bf16,*fp32,*fp32,*bf16,i32,i32,fp32,64,32,64,128,128",
            4,
        ),
    ]
    if dspine:
        from dspine_build import specifications

        specs += specifications(load)
    selected = {}
    prefill_selected = {}
    if gemm_tuning is not None:
        profile = json.loads(gemm_tuning.read_text())
        if profile["triton"] != triton.__version__:
            raise ValueError("GEMM tuning profile must match the build Triton version")
        variants = {}
        for case in profile["cases"]:
            config = tuple(case[k] for k in ("bm", "bn", "bk", "warps", "stages"))
            bm, bn, bk, warps, stages = config
            name = f"fp8_linear_{bm}_{bn}_{bk}_{warps}_{stages}"
            variants[name] = (
                name,
                gemm.fp8_block_gemm,
                f"*bf16,*fp8e4nv,*bf16,*bf16,i32,i32,i32,{bm},{bn},{bk},0",
                warps,
                {},
                {"num_stages": stages},
            )
            selected[f"{case['m']},{case['n']},{case['k']}"] = {
                "module": name,
                "bm": bm,
                "bn": bn,
            }
        for case in profile.get("prefill_cases", []):
            config = tuple(case[k] for k in ("bm", "bn", "bk", "warps", "stages"))
            bm, bn, bk, warps, stages = config
            name = f"fp8_w8a8_{bm}_{bn}_{bk}_{warps}_{stages}"
            variants[name] = (
                name,
                gemm.fp8_w8a8_gemm,
                f"*fp8e4nv,*fp32,*fp8e4nv,*bf16,*bf16,i32,i32,i32,{bm},{bn},{bk}",
                warps,
                {},
                {"num_stages": stages},
            )
            prefill_selected[f"{case['m']},{case['n']},{case['k']}"] = {
                "module": name,
                "bm": bm,
                "bn": bn,
            }
        specs += list(variants.values())
    records = {}
    for item in specs:
        name, kernel, signature, warps = item[:4]
        sig, constants = {}, {}
        for arg, typ in zip(kernel.arg_names, signature.split(","), strict=True):
            if typ.isdecimal():
                constants[arg] = int(typ)
            else:
                sig[arg] = typ
        if len(item) >= 5:
            constants.update(item[4])
        options = {"num_warps": warps, "num_stages": 2, "enable_fp_fusion": not name.startswith("dspine_")}
        if len(item) >= 6:
            options.update(item[5])
        attrs = {}
        if name.startswith("fp8_linear") or name == "fp8_block_gemm":
            # Match JIT: pointers and checkpoint N/K align; live decode M may be 1..16.
            attrs = {(j,): [["tt.divisibility", 16]] for j in (0, 1, 2, 3, 5, 6)}
        elif name == "fp8_act_quant":
            attrs = {(j,): [["tt.divisibility", 16]] for j in (0, 1, 2, 4)}
        elif name.startswith("fp8_w8a8_"):
            attrs = {(j,): [["tt.divisibility", 16]] for j in (0, 1, 2, 3, 4, 6, 7)}
        compiled = triton.compile(
            ASTSource(kernel, signature=sig, constexprs=constants, attrs=attrs),
            target=GPUTarget("cuda", 120, 32),
            options=options,
        )
        if (
            compiled.metadata.global_scratch_size
            or compiled.metadata.profile_scratch_size
            or compiled.metadata.num_ctas != 1
        ):
            raise ValueError(f"{name}: unsupported scratch or cluster launch requirements")
        path = out / (name + ".cubin")
        path.write_bytes(compiled.asm["cubin"])
        (out / (name + ".ptx")).write_text(compiled.asm["ptx"])
        records[name] = {
            "source": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "entry": compiled.name,
            "shared_mem": compiled.metadata.shared,
            "block": [32 * warps, 1, 1],
            "signature": list(sig.values()),
            "arguments": list(sig),
            "abi_tail": ["i64", "i64"],
        }
        print("Built", name, "shared", compiled.metadata.shared, flush=True)
    (out / "dsl.json").write_text(
        json.dumps(
            {
                "triton": triton.__version__,
                "modules": records,
                "gemm_tuning": selected,
                "prefill_gemm_tuning": prefill_selected,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dspine", action="store_true")
    parser.add_argument("--gemm-tuning", type=Path)
    args = parser.parse_args()
    build(args.dspine, args.gemm_tuning)
