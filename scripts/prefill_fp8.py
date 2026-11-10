#!/usr/bin/env python3
"""Build an isolated native FP8 prefill candidate from a served manifest."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(baseline, out):
    out.mkdir(parents=True, exist_ok=False)
    kernels = out / "kernels"
    shutil.copytree(baseline / "kernels", kernels)
    m = json.loads((baseline / "model.json").read_text())
    if (
        m["buffers"]["q"]["dtype"] != "bf16"
        or m["buffers"]["q"]["shape"] != ["tokens", 6144]
        or m["states"]["kv"].get("bytes_per_token") != 32768
    ):
        raise ValueError("Expected Qwen3.8 BF16 queries and the calibrated FP8 KV layout")
    path = kernels / "prefill_fp8.cubin"
    subprocess.run(
        [
            "nvcc",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-O3",
            "-lineinfo",
            "-cubin",
            "-arch=sm_120",
            "-I" + str(ROOT / "third_party/flashinfer/include"),
            str(ROOT / "source/prefill_fp8.cu"),
            "-o",
            str(path),
        ],
        check=True,
    )
    probe = out / "metadata.cu"
    probe.write_text(
        '#include "' + str(ROOT / "source/prefill_fp8.cu") + '"\n#include <cstdio>\n'
        'int main(){printf("%zu %zu %zu %zu\\n",sizeof(Prefill),sizeof(Prefill8),'
        "offsetof(Prefill8,q_scales),sizeof(NativeTraits::SharedStoragePaged));}\n"
    )
    subprocess.run(
        [
            "nvcc",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-arch=sm_120",
            "-I" + str(ROOT / "third_party/flashinfer/include"),
            str(probe),
            "-o",
            str(out / "metadata"),
        ],
        check=True,
    )
    original_size, size, scale_at, shared = map(
        int, subprocess.check_output([str(out / "metadata")], text=True).split()
    )
    m["modules"]["prefill_fp8"] = {
        "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
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
            raise ValueError("Baseline attention ABI differs from the compiled headers")
        pack["size"] = size
        fields = pack["fields"]
        for field in fields:
            if field.get("param") == 0:
                del field["param"]
                field["scratch"] = "q8"
        fields.append({"at": scale_at, "scratch": "scales"})
        # The ABI image is inherited from the baseline; only its tile count changes.
        for field in fields:
            if field.get("expr") == {"ceil_div": [{"mul": ["tokens", 6]}, 64]}:
                field["expr"] = tiles
        launch.update(
            module="prefill_fp8", entry="prefill_fp8", grid=[tiles, 1, 4], block=[32, 1, 4], shared_mem=shared
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
    (out / "model.json").write_text(json.dumps(m, indent=2) + "\n")
    # Keep checkpoint resolution anchored at the source configuration.
    config = tomllib.loads((baseline / "kern.toml").read_text())["targets"]["qwen38"]
    weights = [str((baseline / p).resolve()) if not Path(p).is_absolute() else p for p in config["weights"]]
    (out / "kern.toml").write_text(
        '[targets.qwen38]\nmanifest = "model.json"\nkernels = "kernels"\nweights = '
        + json.dumps(weights)
        + "\n"
    )
    print(out / "model.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.baseline, args.out)
