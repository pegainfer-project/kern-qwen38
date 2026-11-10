#!/usr/bin/env python3
"""Audit SASS spills, FP32/baseline numerics, and cold 5-layer graph timings for dspine_attention."""

import argparse
import copy
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import host_region, region

ROOT = Path(__file__).resolve().parents[1]


def audit_cubin(path):
    res = subprocess.check_output(["cuobjdump", "-res-usage", str(path)], text=True)
    sass = subprocess.check_output(["cuobjdump", "-sass", str(path)], text=True)
    reg_lines = [ln.strip() for ln in res.splitlines() if "REG:" in ln]
    stl = sum(1 for ln in sass.splitlines() if "\tSTL" in ln or " STL" in ln)
    ldl = sum(1 for ln in sass.splitlines() if "\tLDL" in ln or " LDL" in ln)
    if not reg_lines:
        raise ValueError(f"Missing resource usage in {path}")
    info = reg_lines[0]
    regs = int(info.split("REG:")[1].split()[0])
    stack = int(info.split("STACK:")[1].split()[0])
    return {"file": path.name, "registers": regs, "stack_bytes": stack, "static_stl": stl, "static_ldl": ldl}


@torch.inference_mode()
def fp32_oracle(q, k, v, cache, lines, lengths, ring=8208):
    # Independent FP32 5-layer reference; returns layer-4 output (or single-layer for layer=0)
    out = torch.zeros_like(q)
    scale = 128**-0.5
    for s in range(lengths.shape[0]):
        end = int(lengths[s].item())
        if end <= 0:
            continue
        prefix = end - 16
        first = max(0, prefix - 8192)
        line = int(lines[s].item())
        pos = torch.arange(first, end, device=q.device, dtype=torch.int64)
        is_hist = pos < prefix
        c_slice = cache[line, 0, pos % ring]
        old_k = c_slice[:, :1024].view(-1, 8, 128)
        old_v = c_slice[:, 1024:].view(-1, 8, 128)
        live = (pos - prefix).clamp(0, 15)
        new_k = k[s * 16 + live].view(-1, 8, 128)
        new_v = v[s * 16 + live].view(-1, 8, 128)
        kk = torch.where(is_hist[:, None, None], old_k, new_k).float()
        vv = torch.where(is_hist[:, None, None], old_v, new_v).float()
        qq = q[s * 16 : (s + 1) * 16].view(16, 32, 128).float()
        qpos = prefix + torch.arange(16, device=q.device, dtype=torch.int64)
        valid = (
            (pos[None, :] <= qpos[:, None]) & (pos[None, :] < end) & (pos[None, :] >= qpos[:, None] - 8191)
        )
        for h in range(32):
            kh = h // 4
            scores = (qq[:, h] @ kk[:, kh].T) * scale
            scores = scores.masked_fill(~valid, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(valid.any(-1, keepdim=True), probs, torch.zeros_like(probs))
            out[s * 16 : (s + 1) * 16, h * 128 : (h + 1) * 128] = (probs @ vv[:, kh]).to(torch.bfloat16)
    return out


@torch.inference_mode()
def check(bundle, baseline, out):
    torch.manual_seed(204)
    candidate = json.loads((bundle / "model.json").read_text())
    original = json.loads((baseline / "model.json").read_text())

    sass_audit = {
        "baseline": [
            audit_cubin(baseline / "kernels" / original["modules"][l["module"]]["source"])
            for l in original["ops"]["dspine_attention"]["impl"]["launches"]
        ],
        "candidate": [
            audit_cubin(bundle / "kernels" / candidate["modules"][l["module"]]["source"])
            for l in candidate["ops"]["dspine_attention"]["impl"]["launches"]
        ],
    }
    for entry in sass_audit["candidate"]:
        if entry["stack_bytes"] != 0 or entry["static_stl"] != 0 or entry["static_ldl"] != 0:
            raise AssertionError(f"Candidate cubin has stack spill: {entry}")

    used, ops, modules, programs, stores = set(), {}, {}, {}, {}
    for side, source, directory in [("baseline", original, baseline), ("candidate", candidate, bundle)]:
        attn_calls = [
            copy.deepcopy(c) for c in source["programs"]["round"]["calls"] if c["op"] == "dspine_attention"
        ]
        if len(attn_calls) != 5:
            raise ValueError(f"Expected 5 draft attention calls, got {len(attn_calls)}")
        renamed = side + ".dspine_attention"
        op = copy.deepcopy(source["ops"]["dspine_attention"])
        for launch in op["impl"]["launches"]:
            old_mod = launch["module"]
            new_mod = side + "." + old_mod
            rec = copy.deepcopy(source["modules"][old_mod])
            stores[new_mod] = directory / "kernels" / rec["source"]
            rec["source"] = new_mod + ".cubin"
            modules[new_mod] = rec
            launch["module"] = new_mod
        ops[renamed] = op
        for c in attn_calls:
            used.update(a["buf"] for a in c["args"] if "buf" in a)
            c["op"] = renamed
        programs[side + ".layer0"] = {"calls": [attn_calls[0]]}
        programs[side + ".all5"] = {"calls": attn_calls}

    buffers = {
        n: {
            "dtype": candidate["buffers"][n]["dtype"],
            "shape": candidate["buffers"][n]["shape"],
            "kind": "workspace" if n == "d.attn" else "input",
        }
        for n in used
    }
    ring = 8208
    m = {
        "schema_version": 7,
        "model": "dspine-attention-test",
        "vars": candidate["vars"],
        "buffers": buffers,
        "states": {
            "draft_kv": {
                "host": {
                    "dtype": "bf16",
                    "shape": [0, 5, ring, 2048],
                    "strides": [5 * ring * 2048, ring * 2048, 2048, 1],
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
        kdir = Path(td) / "kernels"
        kdir.mkdir()
        for mod_name, src in stores.items():
            shutil.copyfile(src, kdir / modules[mod_name]["source"])
        rt = kern.Runtime(str(path), str(kdir), [], 0)

    views = {n: region(rt, m, n) for n in buffers}
    cache = torch.randn((17, 5, ring, 2048), device="cuda", dtype=torch.bfloat16) * 0.5
    rt.bind_host({"draft_kv": host_region(cache)})

    cases = []
    shapes = [
        (1, [16]),
        (1, [65]),
        (1, [1024]),
        (1, [8192]),
        (1, [8208]),
        (1, [8225]),
        (1, [47708]),
        (2, [95663, 1024]),
        (4, [50183, 16, 8215, 0]),
        (4, [50183, 49811, 32768, 12000]),
        (8, [44423 - j * 503 for j in range(8)]),
        (16, [27057 - j * 311 for j in range(16)]),
    ]
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for batch, lens in shapes:
        tokens = batch * 16
        lines = [0 if ln == 0 else (j + 1) for j, ln in enumerate(lens)]
        views["lengths"][:batch].copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        views["d.lines"][0, :batch].copy_(torch.tensor(lines, device="cuda", dtype=torch.int32))
        views["pages"][:batch].zero_()
        views["d.q"][:tokens].normal_(0, 0.5)
        views["d.k"][:tokens].normal_(0, 0.5)
        views["d.v"][:tokens].normal_(0, 0.5)
        variables = {"seqs": batch, "tokens": tokens}
        torch.cuda.synchronize()

        rt.run("baseline.layer0", variables)
        base_out = views["d.attn"][:tokens].clone()
        rt.run("candidate.layer0", variables)
        cand_out = views["d.attn"][:tokens].clone()
        oracle_out = fp32_oracle(
            views["d.q"][:tokens],
            views["d.k"][:tokens],
            views["d.v"][:tokens],
            cache,
            views["d.lines"][0, :batch],
            views["lengths"][:batch],
        )

        assert cand_out.isfinite().all(), (batch, lens, "non-finite candidate output")
        live_rows = [r for j, ln in enumerate(lens) if ln > 0 for r in range(j * 16, (j + 1) * 16)]
        c_live = cand_out[live_rows]
        b_live = base_out[live_rows]
        o_live = oracle_out[live_rows]
        torch.testing.assert_close(c_live, b_live, atol=0.002, rtol=0.02)
        torch.testing.assert_close(c_live, o_live, atol=0.004, rtol=0.02)
        if batch == 1 and lens == [16]:
            torch.testing.assert_close(c_live[0], b_live[0], atol=0, rtol=0)

        err_base = (c_live.float() - b_live.float()).abs()
        err_oracle = (c_live.float() - o_live.float()).abs()
        rms_vs_base = float(err_base.square().mean().sqrt() / b_live.float().square().mean().sqrt())
        rms_vs_oracle = float(err_oracle.square().mean().sqrt() / o_live.float().square().mean().sqrt())

        # Cold 5-layer CUDA graph timing (matches round's 5 draft attention calls)
        graphs = {}
        for side in ("baseline", "candidate"):
            prog = side + ".all5"
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                rt.enqueue_after(prog, variables, stream.cuda_stream)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                rt.enqueue_after(prog, variables, stream.cuda_stream)
            graphs[side] = graph

        samples = {"baseline": [], "candidate": []}
        for trial in range(12):
            order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
            for side in order:
                flush.fill_(trial)
                start_ev.record()
                graphs[side].replay()
                end_ev.record()
                end_ev.synchronize()
                samples[side].append(start_ev.elapsed_time(end_ev))

        med = {s: statistics.median(v) for s, v in samples.items()}
        case = {
            "batch": batch,
            "lengths": lens,
            "max_abs_vs_baseline": err_base.max().item(),
            "relative_rms_vs_baseline": rms_vs_base,
            "max_abs_vs_fp32_oracle": err_oracle.max().item(),
            "relative_rms_vs_fp32_oracle": rms_vs_oracle,
            "five_layer_baseline_ms": med["baseline"],
            "five_layer_candidate_ms": med["candidate"],
            "reduction_percent": (1.0 - med["candidate"] / med["baseline"]) * 100.0,
            "speedup": med["baseline"] / med["candidate"],
            "samples_ms": samples,
        }
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "samples_ms"}), flush=True)

    report = {"sass": sass_audit, "cases": cases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=ROOT / "out/qwen38-dspine")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    check(args.bundle, args.baseline, args.out)
