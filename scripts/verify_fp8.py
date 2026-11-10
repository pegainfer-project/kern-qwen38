#!/usr/bin/env python3
"""Build independent verification-attention candidates across split counts."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from prefill_fp8 import compile_native


def build(baseline, out, splits, native=True):
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((baseline / "model.json").read_text())
    if (
        original["states"]["kv"].get("bytes_per_token") != 32768
        or original["buffers"]["q"]["dtype"] != "bf16"
        or original["buffers"]["q"]["shape"] != ["tokens", 6144]
    ):
        raise ValueError("Expected Qwen3.8 BF16 queries and calibrated FP8 KV")
    if any(s not in (4, 8, 16, 32, 64) for s in splits):
        raise ValueError("Supported split counts: 4, 8, 16, 32, 64")
    if native:
        compiled = out / "compiled"
        compiled.mkdir()
        module, (original_size, size, scale_at, shared) = compile_native(compiled, compiled)
    groups = original["vars"]["seqs"]["max"]
    for count in splits:
        bundle = out / f"splits-{count}"
        shutil.copytree(baseline / "kernels", bundle / "kernels")
        shutil.copyfile(baseline / "kern.toml", bundle / "kern.toml")
        m = copy.deepcopy(original)
        tiles = 6 if native else 2
        for name in ("plan.request", "plan.tile", "plan.zero", "plan.valid"):
            m["buffers"][name]["shape"][0] = max(m["buffers"][name]["shape"][0], groups * count)
        for name in ("request", "kv_tile", "qo_tile", "valid"):
            m["buffers"]["verify.attn." + name]["shape"][0] = groups * count * tiles
        for name in ("partial", "lse"):
            m["buffers"]["verify.attn." + name]["shape"][0] = groups * count * 16
        plan = copy.deepcopy(m["ops"]["batch_plan"])
        plan["impl"]["launches"][0]["args"][13] = {"i32": count}
        plan["impl"]["launches"][0]["args"][15] = {"i32": groups * count}
        m["ops"]["verify_kv_plan"] = plan
        m["programs"]["round"]["calls"][0]["op"] = "verify_kv_plan"
        expand = m["ops"]["verify_attention_plan"]["impl"]["launches"][0]
        expand["args"][-1] = {"i32": count}
        op = m["ops"]["verify_attention_flashinfer"]
        launch = op["impl"]["launches"][0]
        field_list = launch["args"][0]["pack"]["fields"]
        for field in field_list:
            if field.get("expr") == {"mul": ["seqs", 64]}:
                field["expr"] = {"mul": ["seqs", count * tiles]}
        launch["grid"][0] = {"mul": ["seqs", count * tiles]}
        if native:
            shutil.copyfile(compiled / module["source"], bundle / "kernels" / module["source"])
            m["modules"]["prefill_fp8"] = module
            expand.update(module="prefill_fp8", entry="verify_plan_fp8")
            pack = launch["args"][0]["pack"]
            if pack["size"] != original_size:
                raise ValueError("Baseline attention ABI differs from compiled headers")
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
        (bundle / "model.json").write_text(json.dumps(m, indent=2) + "\n")
        print(bundle, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()
    build(args.baseline, args.out, args.splits, not args.bf16)
