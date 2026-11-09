#!/usr/bin/env python3
"""Compile local Triton operators to standalone cubins for kern; no JIT at serving time."""

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


def build():
    out = ROOT / "build/sm_120"
    out.mkdir(parents=True, exist_ok=True)
    gdn = load("gdn_prefill")
    gemm = load("fp8_gemm")
    specs = [
        ("fp8_weight_gemm", gemm.fp8_block_gemm, "*bf16,*fp8e4nv,*bf16,*bf16,i32,i32,i32,128,64,64,0", 8),
        ("fp8_block_gemm", gemm.fp8_block_gemm, "*bf16,*fp8e4nv,*bf16,*bf16,i32,i32,i32,32,64,128,1", 4),
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
    records = {}
    for name, kernel, signature, warps in specs:
        sig, constants = {}, {}
        for arg, typ in zip(kernel.arg_names, signature.split(","), strict=True):
            if typ.isdecimal():
                constants[arg] = int(typ)
            else:
                sig[arg] = typ
        compiled = triton.compile(
            ASTSource(kernel, signature=sig, constexprs=constants),
            target=GPUTarget("cuda", 120, 32),
            options={"num_warps": warps, "num_stages": 2},
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
            "abi_tail": ["i64", "i64"],
        }
        print("Built", name, "shared", compiled.metadata.shared, flush=True)
    (out / "dsl.json").write_text(
        json.dumps({"triton": triton.__version__, "modules": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    build()
