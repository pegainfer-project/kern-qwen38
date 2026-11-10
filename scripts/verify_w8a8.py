#!/usr/bin/env python3
"""Autotune and build an isolated W8A8 verification MLP candidate."""

import argparse
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import torch
import triton
import triton.testing
from build_dsl import load
from gemm_manifest import prefill_linear
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROWS = (16, 32, 64, 128, 256)
GEOMETRIES = ((34816, 5120), (5120, 17408))


def experiment(baseline, out):
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    kernels = out / "kernels"
    shutil.copytree(baseline / "kernels", kernels)
    m = json.loads((baseline / "model.json").read_text())
    gemm = load("fp8_gemm")
    configs = [
        triton.Config({"BM": bm, "BN": bn, "BK": 128}, num_warps=warps, num_stages=stages)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for warps in (4, 8)
        for stages in (2, 3)
    ]
    tuned = triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        do_bench=lambda fn, quantiles: triton.testing.do_bench(fn, warmup=5, rep=20, quantiles=quantiles),
    )(gemm.fp8_w8a8_gemm)
    quant_launch = m["ops"]["gemm_34816_5120"]["impl"]["launches"][0]
    dsl = {"fp8_act_quant": {**quant_launch, "abi_tail": ["i64", "i64"]}}
    selected, results = {}, []
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    for n, k in GEOMETRIES:
        w = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
        scales = (torch.rand((n // 128, k // 128), device="cuda") + 0.5).bfloat16()
        decoded = (w.float() * scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
        for rows in ROWS:
            a = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
            aq = torch.empty_like(a, dtype=torch.float8_e4m3fn)
            sa = torch.empty((rows, k // 128), device="cuda", dtype=torch.float32)
            y = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)
            qgrid = (triton.cdiv(rows, 16), k // 128)
            quant = partial(gemm.fp8_act_quant[qgrid], a, aq, sa, rows, k, BM=16, num_warps=4, num_stages=2)
            quant()
            grid = lambda meta, rows=rows, n=n: (triton.cdiv(rows, meta["BM"]), triton.cdiv(n, meta["BN"]))
            tuned[grid](aq, sa, w, scales, y, rows, n, k)
            cfg = tuned.best_config
            ref = a.float() @ decoded.float().T
            error = float((y.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt())
            if not torch.isfinite(y).all() or error > 0.03:
                raise ValueError(f"W8A8 numerical gate: {rows}/{n}/{k}, relative RMS {error}")
            name = f"verify_w8a8_{cfg.kwargs['BM']}_{cfg.kwargs['BN']}_{cfg.num_warps}_{cfg.num_stages}"
            if name not in dsl:
                signature = dict(
                    zip(
                        gemm.fp8_w8a8_gemm.arg_names[:8],
                        ("*fp8e4nv", "*fp32", "*fp8e4nv", "*bf16", "*bf16", "i32", "i32", "i32"),
                        strict=True,
                    )
                )
                compiled = triton.compile(
                    ASTSource(
                        gemm.fp8_w8a8_gemm,
                        signature=signature,
                        constexprs=cfg.kwargs,
                        attrs={(j,): [["tt.divisibility", 16]] for j in (0, 1, 2, 3, 4, 6, 7)},
                    ),
                    target=GPUTarget("cuda", 120, 32),
                    options={"num_warps": cfg.num_warps, "num_stages": cfg.num_stages},
                )
                if (
                    compiled.metadata.global_scratch_size
                    or compiled.metadata.profile_scratch_size
                    or compiled.metadata.num_ctas != 1
                ):
                    raise ValueError(f"Unsupported AOT launch: {name}")
                path = kernels / (name + ".cubin")
                path.write_bytes(compiled.asm["cubin"])
                m["modules"][name] = {
                    "source": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                dsl[name] = {
                    "entry": compiled.name,
                    "abi_tail": ["i64", "i64"],
                    "block": [32 * cfg.num_warps, 1, 1],
                    "shared_mem": compiled.metadata.shared,
                }
            selected[f"{rows},{n},{k}"] = {"module": name, "bm": cfg.kwargs["BM"], "bn": cfg.kwargs["BN"]}
            launch = partial(tuned[grid], aq, sa, w, scales, y, rows, n, k)

            def candidate(quant=quant, launch=launch):
                quant()
                launch()

            elapsed = triton.testing.do_bench(candidate, warmup=5, rep=50)
            result = {
                "rows": rows,
                "n": n,
                "k": k,
                "config": str(cfg),
                "quant_gemm_ms": elapsed,
                "relative_rms": error,
            }
            results.append(result)
            (out / "tuning.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        del w, scales, decoded
    for n, k in GEOMETRIES:
        op = f"verify_linear_{n}_{k}"
        m["ops"][op] = prefill_linear(n, k, selected, dsl, lambda name: name, buckets=ROWS)
    used = {
        launch["module"]
        for op in m["ops"].values()
        for launch in op["impl"].get("launches", [])
        if "module" in launch
    }
    m["modules"] = {name: module for name, module in m["modules"].items() if name in used}
    (out / "model.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"Candidate: {out / 'model.json'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    experiment(args.baseline, args.out)
