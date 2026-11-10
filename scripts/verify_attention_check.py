#!/usr/bin/env python3
"""Compare complete verification attention operators with independent split plans."""

import argparse
import copy
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import host_region, region

ROOT = Path(__file__).resolve().parents[1]


def check(bundle, baseline, out, rms_limit=None):
    if rms_limit is not None and not 0 < rms_limit < 1:
        raise ValueError("rms-limit must be finite and between zero and one")
    torch.manual_seed(204)
    candidate = json.loads((bundle / "model.json").read_text())
    original = json.loads((baseline / "model.json").read_text())
    used, ops, modules, programs = set(), {}, {}, {}
    stores = {}
    for side, source, directory in [("baseline", original, baseline), ("candidate", candidate, bundle)]:
        calls = copy.deepcopy(source["programs"]["round"]["calls"][:2])
        attention = copy.deepcopy(
            next(c for c in source["programs"]["round"]["calls"] if c["op"] == "verify_attention_flashinfer")
        )
        for call in [*calls, attention]:
            used.update(a["buf"] for a in call["args"] if "buf" in a)
            name = call["op"]
            renamed = side + "." + name
            if renamed not in ops:
                op = copy.deepcopy(source["ops"][name])
                for launch in op["impl"]["launches"]:
                    old_name = launch["module"]
                    new_name = side + "." + old_name
                    record = copy.deepcopy(source["modules"][old_name])
                    stores[new_name] = directory / "kernels" / record["source"]
                    record["source"] = new_name + ".cubin"
                    modules[new_name] = record
                    launch["module"] = new_name
                ops[renamed] = op
            call["op"] = renamed
        programs[side + ".plan"] = {"calls": calls}
        programs[side] = {"calls": [attention]}
    buffers = {
        n: {
            "dtype": candidate["buffers"][n]["dtype"],
            "shape": [
                max(a, b) if isinstance(a, int) else a
                for a, b in zip(
                    original["buffers"][n]["shape"], candidate["buffers"][n]["shape"], strict=True
                )
            ],
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
        "programs": programs,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.json"
        path.write_text(json.dumps(m))
        kernels = Path(td) / "kernels"
        kernels.mkdir()
        for name, source in stores.items():
            shutil.copyfile(source, kernels / modules[name]["source"])
        rt = kern.Runtime(str(path), str(kernels), [], 0)
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
        (4, 32768),
        (8, 32768),
        (16, 32768),
        (16, 131072),
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
        rt.run("baseline.plan", variables)
        rt.run("baseline", variables)
        ref = views["attn.out"][: batch * 16].clone()
        rt.run("candidate.plan", variables)
        rt.run("candidate", variables)
        actual = views["attn.out"][: batch * 16].clone()
        assert actual.isfinite().all(), (batch, length, "non-finite output")
        error = (actual.float() - ref.float()).abs()
        relative_rms = float(error.square().mean().sqrt() / ref.float().square().mean().sqrt())
        if rms_limit is None:
            torch.testing.assert_close(actual, ref, atol=0.002, rtol=0.02)
        elif relative_rms > rms_limit:
            raise AssertionError(
                f"Relative RMS {relative_rms} exceeds {rms_limit}: batch={batch}, KV={length}"
            )
        if batch == 1 and length == 16:
            # The first causal query sees one key, so its output is exactly the scaled V.
            torch.testing.assert_close(actual[0], ref[0], atol=0, rtol=0)
        graphs = {}
        for name in ("baseline", "candidate"):
            rt.run(name + ".plan", variables)
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                rt.enqueue_after(name, variables, stream.cuda_stream)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                rt.enqueue_after(name, variables, stream.cuda_stream)
            graphs[name] = graph
        samples = {"baseline": [], "candidate": []}
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for trial in range(12):
            for name in ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline"):
                rt.run(name + ".plan", variables)
                flush.fill_(trial)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        times = {name: statistics.median(values) for name, values in samples.items()}
        case = {
            "batch": batch,
            "length": length,
            "max_abs_error": error.max().item(),
            "baseline_ms": times["baseline"],
            "candidate_ms": times["candidate"],
            "relative_rms": relative_rms,
            "samples_ms": samples,
            "speedup": times["baseline"] / times["candidate"],
        }
        cases.append(case)
        print(json.dumps(case), flush=True)
        del cache, values, ref, actual, graph, graphs, flush
        torch.cuda.empty_cache()
    out.write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "out/verify-fp8/splits-16")
    p.add_argument("--baseline", type=Path, default=ROOT / "out/qwen38-dspine")
    p.add_argument("--out", type=Path, default=ROOT / "out/verify-attention-check.json")
    p.add_argument("--rms-limit", type=float, help="Explicit relative RMS gate for quantized candidates")
    args = p.parse_args()
    check(args.bundle, args.baseline, args.out, args.rms_limit)
