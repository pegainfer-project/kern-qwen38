#!/usr/bin/env python3
"""Autotune ordinary decode and block verification GEMMs before AOT compilation."""

import argparse
import importlib.util
import json
import time
from functools import partial
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = ((16384, 5120, 48), (14336, 5120, 16), (34816, 5120, 64), (5120, 6144, 64), (5120, 17408, 64))


def tune(out):
    spec = importlib.util.spec_from_file_location("fp8_gemm", ROOT / "source/fp8_gemm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    configurations = [
        (16, 32, 64, 4),
        (16, 32, 128, 4),
        (16, 64, 64, 4),
        (16, 64, 128, 8),
        (32, 32, 64, 4),
        (32, 32, 128, 4),
        (32, 64, 64, 4),
        (32, 64, 128, 8),
        (64, 32, 64, 4),
        (64, 32, 128, 8),
        (64, 64, 64, 8),
        (64, 64, 128, 8),
        (128, 32, 64, 8),
        (128, 32, 128, 8),
        (128, 64, 64, 8),
        (128, 64, 128, 8),
    ]
    configs = [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "A8": 0}, num_warps=warps, num_stages=stages)
        for bm, bn, bk, warps in configurations
        for stages in (2, 3)
    ]
    tuned = triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        do_bench=lambda fn, quantiles: triton.testing.do_bench(fn, warmup=5, rep=20, quantiles=quantiles),
    )(module.fp8_block_gemm)
    started = time.monotonic()
    cases = []
    for n, k, multiplicity in GEOMETRIES:
        w = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
        scales = (torch.rand((n // 128, k // 128), device="cuda") + 0.5).bfloat16()
        decoded = (w.float() * scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            grid = lambda meta, m=m, n=n: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))
            tuned[grid](a, w, scales, output, m, n, k)
            cfg = tuned.best_config
            reference = (a.float() @ decoded.float().T).bfloat16()
            relative_rms = float(
                (output.float() - reference.float()).square().mean().sqrt()
                / reference.float().square().mean().sqrt()
            )
            if not torch.isfinite(output).all() or relative_rms > 0.0005:
                raise ValueError(
                    f"GEMM numerical gate failed: M/N/K={m}/{n}/{k}, relative RMS={relative_rms}"
                )
            elapsed = triton.testing.do_bench(
                partial(tuned[grid], a, w, scales, output, m, n, k), warmup=5, rep=50
            )
            result = {
                "m": m,
                "n": n,
                "k": k,
                "calls_per_round": multiplicity,
                "bm": cfg.kwargs["BM"],
                "bn": cfg.kwargs["BN"],
                "bk": cfg.kwargs["BK"],
                "warps": cfg.num_warps,
                "stages": cfg.num_stages,
                "milliseconds": elapsed,
                "relative_rms_error": relative_rms,
            }
            cases.append(result)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "triton": triton.__version__,
                        "configurations": len(configs),
                        "elapsed_seconds": time.monotonic() - started,
                        "cases": cases,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(result), flush=True)
        del w, scales, decoded
    print(
        json.dumps(
            {
                "estimated_projection_ms": {
                    m: sum(r["milliseconds"] * r["calls_per_round"] for r in cases if r["m"] == m)
                    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                }
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "out/gemm-tuning.json")
    tune(p.parse_args().out)
