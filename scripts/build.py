#!/usr/bin/env python3
"""Build SM120 cubins against the pinned FlashInfer headers and record provenance."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASHINFER_REVISION = "023a8474e1f7469572c0d4284535fc6ffa699574"


def build(nvcc, jobs):
    upstream = ROOT / "third_party/flashinfer"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != FLASHINFER_REVISION:
        raise ValueError(f"FlashInfer: expected {FLASHINFER_REVISION}, got {revision}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=upstream):
        raise ValueError("FlashInfer has local modifications")
    out = ROOT / "build/sm_120"
    out.mkdir(parents=True, exist_ok=True)
    include = "-I" + str(upstream / "include")
    modules = [
        (name, name, [])
        for name in (
            "model_glue",
            "dspine_state",
            "embedding",
            "gemma_rms_norm",
            "weight_prep",
            "sigmoid_mul",
            "silu_mul",
            "argmax",
            "copy_rows",
            "line_copy",
            "sample_glue",
            "sampling",
            "attention_plan",
        )
    ]
    for kv, value in (("bf16", 0), ("fp8", 1)):
        modules.append((f"attn_prep_{kv}", "attn_prep", [f"-DKERN_KV_FP8={value}"]))
        modules.append((f"attention_{kv}", "attention", [f"-DKERN_KV_FP8={value}", "-DKERN_HEAD_DIM=256"]))
    for state, value in (("fp32", 0), ("bf16", 1)):
        modules.append((f"gdn_decode_{state}", "gdn_decode", [f"-DGDN_STATE_BF16={value}"]))

    def compile_module(item):
        name, source, defines = item
        src = ROOT / "source" / f"{source}.cu"
        target = out / f"{name}.cubin"
        command = [
            nvcc,
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-O3",
            "-lineinfo",
            "-cubin",
            "-arch=sm_120",
            include,
            *defines,
            str(src),
            "-o",
            str(target),
        ]
        subprocess.run(command, check=True)
        print(f"Built {name}", flush=True)
        return name, {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "command": command,
        }

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        built = dict(pool.map(compile_module, modules))
    metadata = {}
    for kv, value in (("bf16", 0), ("fp8", 1)):
        probe = out / f"attention_metadata_{kv}"
        subprocess.run(
            [
                nvcc,
                "-std=c++17",
                "--expt-relaxed-constexpr",
                include,
                "-DKERN_HEAD_DIM=256",
                f"-DKERN_KV_FP8={value}",
                str(ROOT / "source/attention_metadata.cu"),
                "-o",
                str(probe),
            ],
            check=True,
        )
        metadata[kv] = json.loads(subprocess.check_output([str(probe)], text=True))
    probe = out / "sampling_metadata"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "--expt-relaxed-constexpr",
            include,
            str(ROOT / "source/sampling_metadata.cu"),
            "-o",
            str(probe),
        ],
        check=True,
    )
    usage = subprocess.check_output(
        [str(Path(nvcc).resolve().with_name("cuobjdump")), "-res-usage", str(out / "sampling.cubin")],
        text=True,
    )
    entries = {}
    for entry, resources in re.findall(r"Function ([^:]+):\s*([^\n]+)", usage):
        tag = next(
            (
                name
                for name in (
                    "OnlineSoftmaxFusedKernel",
                    "RadixTopKRenormProbKernel_MultiCTA",
                    "TopPRenormProbKernel",
                    "MinPSamplingFromProbKernel",
                )
                if name in entry
            ),
            None,
        )
        if tag:
            if tag in entries:
                raise ValueError(f"ambiguous sampling entry {tag}")
            entries[tag] = {"entry": entry, "static_shared": int(re.search(r"SHARED:(\d+)", resources)[1])}
    if len(entries) != 4:
        raise ValueError(f"expected four sampling entries, got {entries}")
    report = {
        "architecture": "sm_120",
        "flashinfer_revision": revision,
        "compiler": subprocess.check_output([nvcc, "--version"], text=True),
        "modules": built,
        "attention": metadata,
        "sampling": {
            "sizes": json.loads(subprocess.check_output([str(probe)], text=True)),
            "entries": entries,
            "revision": revision,
        },
    }
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not args.nvcc or args.jobs < 1:
        parser.error("nvcc and a positive --jobs value are required")
    build(args.nvcc, args.jobs)
