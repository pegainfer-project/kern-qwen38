#!/usr/bin/env python3
"""Compile and execute standalone GPU operator gates; no Python CUDA framework required."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(nvcc, sanitize):
    out = ROOT / "build/check"
    out.mkdir(parents=True, exist_ok=True)
    cases = [("gemv", "gemv_check", [])]
    cases += [(f"gdn_{dtype}", "gdn_check", [f"-DGDN_STATE_BF16={flag}"])
              for dtype, flag in (("fp32", 0), ("bf16", 1))]
    for dtype, flag in (("bf16", 0), ("fp8", 1)):
        for name in ("attn_prep", "attention"):
            cases.append((f"{name}_{dtype}", f"{name}_check", [f"-DKERN_KV_FP8={flag}"]))
    results = []
    for name, source, flags in cases:
        binary = out / name
        subprocess.run([nvcc, "-std=c++17", "--expt-relaxed-constexpr", "-O3", "-lineinfo", "-arch=sm_120",
                        "-I" + str(ROOT / "third_party/flashinfer/include"), *flags,
                        str(ROOT / "tests" / f"{source}.cu"), "-o", str(binary)], check=True)
        command = [str(binary)]
        if sanitize:
            command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1", *command]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, check=True)
        print(completed.stdout, end="", flush=True)
        results += [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    report = {"gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
        "compiler": subprocess.check_output([nvcc, "--version"], text=True),
        "flashinfer_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/flashinfer", text=True).strip(),
        "sanitized": sanitize, "cases": results}
    (ROOT / ("results/operators-memcheck.json" if sanitize else "results/operators.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if not args.nvcc:
        parser.error("nvcc is required")
    check(args.nvcc, args.sanitize)
