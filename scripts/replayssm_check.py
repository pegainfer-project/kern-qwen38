#!/usr/bin/env python3
"""Compare AOT ReplaySSM verify/commit against a direct FP32 recurrence with BF16 storage."""

import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import region


@torch.inference_mode()
def check(bundle, out, timing=False):
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = json.loads((bundle / "model.json").read_text())
    metadata = json.loads((Path(__file__).resolve().parents[1] / "build/sm_120/dsl.json").read_text())[
        "modules"
    ]
    line = 3 * 10240 + 48 * 128 * 128
    shapes = {
        "qkv": ("bf16", [256, 10240]),
        "ba": ("bf16", [256, 96]),
        "alog": ("f32", [48]),
        "dt": ("bf16", [48]),
        "output": ("bf16", [256, 6144]),
        "state": ("bf16", [17, line]),
        "D": ("bf16", [16, 48 * 16 * 128]),
        "K": ("bf16", [16, 16 * 16 * 128]),
        "G": ("f32", [16, 48 * 16]),
        "Dr": ("bf16", [16, 48 * 16 * 128]),
        "Kr": ("bf16", [16, 16 * 16 * 128]),
        "qsl": ("i32", [17]),
        "lines": ("i32", [16]),
        "indices": ("i64", [16]),
        "zero": ("i32", [16]),
        "one": ("i32", [16]),
        "accepted": ("i32", [16]),
    }
    buffers = {
        name: {
            "dtype": dtype,
            "shape": shape,
            "kind": "carry"
            if name in ("D", "K", "G", "Dr", "Kr", "output")
            else "inout"
            if name == "state"
            else "input",
        }
        for name, (dtype, shape) in shapes.items()
    }
    buffers["qkv"]["shape"] = ["tokens", 10240]
    arg = lambda name, offset=0: {"buf": name, **({"offset": offset} if offset else {})}
    common = {
        "h0": arg("state", 3 * 10240 * 2),
        "d_cache": arg("D"),
        "k_cache": arg("K"),
        "g_cache": arg("G"),
        "ssm_state_indices": arg("lines"),
        "replay_indices": arg("indices"),
        "cache_base": arg("zero"),
        "is_flush_flags": arg("zero"),
    }
    verify = common | {
        "q": arg("qkv"),
        "k": arg("qkv", 2048 * 2),
        "v": arg("qkv", 4096 * 2),
        "a": arg("ba", 48 * 2),
        "b": arg("ba"),
        "A_log": arg("alog"),
        "dt_bias": arg("dt"),
        "o": arg("output"),
        "rawv_cache": arg("Dr"),
        "rawk_cache": arg("Kr"),
        "beta_cache": arg("G"),
        "query_start_loc": arg("qsl"),
        "write_pos": arg("zero"),
    }
    commit = common | {
        "d_residual_cache": arg("Dr"),
        "k_residual_cache": arg("Kr"),
        "write_pos": arg("accepted"),
        "is_flush_flags": arg("one"),
        "accept_lens": arg("accepted"),
        "mamba_track_indices": arg("lines"),
        "mamba_steps_to_track": arg("accepted"),
    }
    ops = {}
    programs = {}
    for name, values in [("replay_verify", verify), ("replay_commit", commit)]:
        record = metadata[name]
        op = json.loads(json.dumps(original["ops"][name]))
        op["params"] = [
            p.replace("inout state", "inout buffer<bf16>").replace("in state", "in buffer<bf16>")
            for p in op["params"]
        ]
        op["impl"]["launches"][0]["params"] = op["params"] + record["abi_tail"]
        ops[name] = op
        programs[name] = {"calls": [{"op": name, "args": [values[n] for n in record["arguments"]]}]}
    m = {
        "schema_version": 7,
        "model": "replayssm-gate",
        "vars": {"seqs": {"max": 16}, "tokens": {"max": 256}},
        "buffers": buffers,
        "modules": {n: original["modules"][n] for n in ops},
        "ops": ops,
        "programs": programs,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(m))
        rt = kern.Runtime(str(path), str(bundle / "kernels"), [], 0)
    views = {name: region(rt, m, name) for name in buffers}
    cases = []
    for groups in (1, 2, 4, 8, 16):
        rows = groups * 16
        qkv = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16) * 0.2
        ba = torch.randn((rows, 96), device="cuda", dtype=torch.bfloat16)
        alog = torch.full((48,), -2.0, device="cuda")
        dt = torch.zeros(48, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn((groups, 48, 128, 128), device="cuda", dtype=torch.bfloat16) * 0.03
        accepted = torch.linspace(1, 16, groups, device="cuda").int()
        if groups == 4:
            initial[-1].zero_()
            qkv.view(groups, 16, 10240)[-1].zero_()
        state = initial.float()
        expected = []
        committed = []
        for token in range(16):
            x = qkv.view(groups, 16, 10240)[:, token].float()
            q = x[:, :2048].view(groups, 16, 128)
            k = x[:, 2048:4096].view(groups, 16, 128)
            q = (q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(3, dim=1) * (
                128**-0.5
            )
            k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(3, dim=1)
            v = x[:, 4096:].view(groups, 48, 128)
            gates = ba.view(groups, 16, 96)[:, token].float()
            decay = torch.exp(-alog.exp() * torch.nn.functional.softplus(gates[:, 48:] + dt.float()))
            beta = gates[:, :48].sigmoid()
            state = state * decay[:, :, None, None]
            delta = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
            state = state + delta[:, :, :, None] * k[:, :, None, :]
            expected.append((state * q[:, :, None, :]).sum(-1))
            committed.append(state.bfloat16().clone())
        reference = torch.stack(expected, dim=1).reshape(rows, 6144).bfloat16()
        final = torch.stack([committed[int(accepted[s]) - 1][s] for s in range(groups)])
        for view in views.values():
            view.zero_()
        views["qkv"][:rows].copy_(qkv)
        views["ba"][:rows].copy_(ba)
        views["alog"].copy_(alog)
        views["dt"].copy_(dt)
        views["state"][1 : groups + 1, 3 * 10240 :].view(groups, 48, 128, 128).copy_(initial)
        views["qsl"][: groups + 1].copy_(torch.arange(groups + 1, device="cuda", dtype=torch.int32) * 16)
        views["lines"][:groups].copy_(torch.arange(1, groups + 1, device="cuda", dtype=torch.int32))
        if groups == 4:
            views["lines"][groups - 1] = 0
        views["indices"][:groups].copy_(torch.arange(groups, device="cuda"))
        views["one"].fill_(1)
        views["accepted"][:groups].copy_(accepted)
        torch.cuda.synchronize()
        rt.run("replay_verify", {"seqs": groups, "tokens": rows})
        actual = views["output"][:rows].clone()
        unchanged = views["state"][1 : groups + 1, 3 * 10240 :].view_as(initial)
        if not torch.equal(unchanged, initial):
            raise AssertionError("verify mutated committed state")
        times = []
        if timing:
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            variables = {"seqs": groups, "tokens": rows}
            with torch.cuda.stream(stream):
                rt.enqueue_after("replay_verify", variables, stream.cuda_stream)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                rt.enqueue_after("replay_verify", variables, stream.cuda_stream)
            for _ in range(4):
                graph.replay()
            flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for sample in range(12):
                flush.fill_(sample)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            del graph, flush
        rt.run("replay_commit", {"seqs": groups, "tokens": rows})
        actual_state = views["state"][1 : groups + 1, 3 * 10240 :].view_as(initial).clone()
        torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.02)
        torch.testing.assert_close(actual_state, final, atol=0.002, rtol=0.02)
        case = {
            "batch": groups,
            "accepted": accepted.tolist(),
            "verify_max_error": float((actual.float() - reference.float()).abs().max()),
            "commit_max_error": float((actual_state.float() - final.float()).abs().max()),
            "verify_preserves_state": True,
            "null_slots": 1 if groups == 4 else 0,
        }
        if timing:
            case.update(
                verify_ms=statistics.median(times),
                samples_ms=times,
                verify_sha256=hashlib.sha256(actual.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                commit_sha256=hashlib.sha256(
                    actual_state.view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
            )
        cases.append(case)
        print(json.dumps(case), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"cases": cases, "reference": "direct FP32 recurrence, BF16 checkpoint at accepted boundary"},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/replayssm.json"))
    p.add_argument("--timing", action="store_true")
    a = p.parse_args()
    check(a.bundle.resolve(), a.out, a.timing)
