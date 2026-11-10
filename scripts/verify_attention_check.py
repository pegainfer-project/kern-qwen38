#!/usr/bin/env python3
"""Exercise AOT FlashInfer verification against the previous Triton kernel."""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import host_region, region

ROOT = Path(__file__).resolve().parents[1]


def check(bundle, baseline, out):
    torch.manual_seed(204)
    candidate = json.loads((bundle / "model.json").read_text())
    original = json.loads((baseline / "model.json").read_text())
    old = copy.deepcopy(
        next(c for c in original["programs"]["round"]["calls"] if c["op"] == "verify_attention")
    )
    new = copy.deepcopy(
        next(c for c in candidate["programs"]["round"]["calls"] if c["op"] == "verify_attention_flashinfer")
    )
    plan = copy.deepcopy(candidate["programs"]["round"]["calls"][:2])
    used = {a["buf"] for c in [old, new, *plan] for a in c["args"] if "buf" in a}
    ops = {c["op"]: candidate["ops"].get(c["op"], original["ops"].get(c["op"])) for c in [old, new, *plan]}
    names = {l["module"] for op in ops.values() for l in op["impl"]["launches"]}
    modules = {n: candidate["modules"].get(n, original["modules"].get(n)) for n in names}
    buffers = {
        n: {
            "dtype": candidate["buffers"][n]["dtype"],
            "shape": candidate["buffers"][n]["shape"],
            "kind": "carry"
            if n.startswith(("plan.", "verify.attn."))
            else "workspace"
            if n == "attn.out"
            else "input",
        }
        for n in used
    }
    m = {
        "schema_version": 7,
        "model": "verify-attention-test",
        "vars": candidate["vars"],
        "buffers": buffers,
        "states": {
            "kv": {
                "host": {
                    "dtype": "u8",
                    "shape": [0, 16, 2, 4, 64, 256],
                    "strides": [2097152, 131072, 65536, 16384, 256, 1],
                }
            }
        },
        "modules": modules,
        "ops": ops,
        "programs": {"plan": {"calls": plan}, "baseline": {"calls": [old]}, "candidate": {"calls": [new]}},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.json"
        path.write_text(json.dumps(m))
        rt = kern.Runtime(str(path), str(ROOT / "build/sm_120"), [], 0)
    views = {n: region(rt, m, n) for n in buffers}
    cases = []
    for batch, length in [
        (1, 16),
        (1, 65),
        (1, 1024),
        (1, 8192),
        (1, 32768),
        (1, 34904),
        (1, 131072),
        (2, 32768),
        (3, 1025),
        (8, 8193),
        (16, 4097),
    ]:
        pages = (length + 63) // 64
        cache = torch.zeros((pages + 1, 16, 2, 4, 64, 256), device="cuda", dtype=torch.uint8)
        values = (torch.randn((pages + 1, 2, 4, 64, 256), device="cuda") * 16).to(torch.float8_e4m3fn)
        cache.view(torch.float8_e4m3fn)[:, 0].copy_(values)
        rt.bind_host({"kv": host_region(cache)})
        lengths = [max(16, length - j * 17) for j in range(batch)]
        views["lengths"][:batch].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        views["q_indptr"][: batch + 1].copy_(torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 16)
        for j in range(batch):
            views["pages"][j, :pages].copy_(torch.randperm(pages, device="cuda") + 1)
        views["q"][: batch * 16].normal_()
        variables = {"seqs": batch, "tokens": batch * 16}
        torch.cuda.synchronize()
        rt.run("plan", variables)
        rt.run("baseline", variables)
        ref = views["attn.out"][: batch * 16].clone()
        rt.run("candidate", variables)
        actual = views["attn.out"][: batch * 16].clone()
        assert actual.isfinite().all(), (batch, length, "non-finite output")
        error = (actual.float() - ref.float()).abs()
        torch.testing.assert_close(actual, ref, atol=0.002, rtol=0.02)
        times = {}
        for name in ("baseline", "candidate"):
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                rt.enqueue_after(name, variables, stream.cuda_stream)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                rt.enqueue_after(name, variables, stream.cuda_stream)
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            times[name] = start.elapsed_time(end) / 30
        case = {
            "batch": batch,
            "length": length,
            "max_abs_error": error.max().item(),
            "baseline_ms": times["baseline"],
            "flashinfer_ms": times["candidate"],
            "speedup": times["baseline"] / times["candidate"],
        }
        cases.append(case)
        print(json.dumps(case), flush=True)
        del cache, values, ref, actual, graph
        torch.cuda.empty_cache()
    out.write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "out/qwen38-dspine-flashinfer")
    p.add_argument("--baseline", type=Path, default=ROOT / "out/qwen38-dspine-triton")
    p.add_argument("--out", type=Path, default=ROOT / "out/verify-attention-check.json")
    args = p.parse_args()
    check(args.bundle, args.baseline, args.out)
