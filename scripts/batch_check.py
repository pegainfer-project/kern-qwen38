#!/usr/bin/env python3
"""Compare the complete batched model against singleton execution with identical prior states."""

import argparse
import json
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import host_region, region


def check(bundle, checkpoint, out, boundary=False):
    torch.manual_seed(117)
    m = json.loads((bundle / "model.json").read_text())
    maximum = m["vars"]["seqs"]["max"]
    line = m["buffers"]["gdn.lines"]["domain"]["stride"]
    pages_per_seq = 2
    count = maximum * pages_per_seq + 1
    kv = torch.zeros((count, 16, 2, 4, 64, 256), device="cuda", dtype=torch.uint8)
    state = torch.zeros((4 if boundary else maximum + 1, 48, line // 2), device="cuda", dtype=torch.bfloat16)
    for name, tensor in [("kv", kv), ("gdn", state)]:
        m["states"][name] = {
            "host": {
                "dtype": "u8" if name == "kv" else "bf16",
                "shape": [0, *tensor.shape[1:]],
                "strides": list(tensor.stride()),
            }
        }
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "model.json"
        manifest.write_text(json.dumps(m))
        rt = kern.Runtime(str(manifest), str(bundle / "kernels"), [str(checkpoint)], 0)
    rt.bind_host({"kv": host_region(kv), "gdn": host_region(state)})
    rt.run("load", {})
    buffers = {name: region(rt, m, name) for name, value in m["buffers"].items() if value["kind"] != "weight"}
    physical = (torch.randperm(count - 1, device="cuda") + 1).to(torch.int32).view(maximum, pages_per_seq)
    lengths = [2 + j % 69 for j in range(maximum)]

    def stage(sequences, ids, positions, prefill=False):
        groups = len(sequences)
        rows = len(ids)
        buffers["token_ids"][:rows].copy_(ids)
        buffers["positions"][:rows].copy_(positions)
        buffers["q_indptr"][: groups + 1].copy_(
            torch.tensor([0, rows] if prefill else list(range(groups + 1)), device="cuda", dtype=torch.int32)
        )
        for column, seq in enumerate(sequences):
            if boundary:
                buffers["pages"][column].zero_()
                buffers["pages"][column, int(positions[column]) // 64] = physical[seq, 0]
            else:
                buffers["pages"][column, :pages_per_seq].copy_(physical[seq])
            buffers["gdn.lines"][:, column].copy_(
                torch.arange(48, device="cuda", dtype=torch.int32) + (seq + 1) * 48
            )
            lo, hi = (0, rows) if prefill else (column, column + 1)
            pos = positions[lo:hi]
            source = physical[seq, 0].expand_as(pos) if boundary else physical[seq, pos // 64]
            buffers["slots"][lo:hi].copy_(source.long() * 64 + pos % 64)
            buffers["lengths"][column] = int(pos[-1]) + 1
        torch.cuda.synchronize()
        return {"tokens": rows, "seqs": groups}

    # Distinct, variable-length prefixes establish both KV and nonzero recurrent states.
    for seq, n in [] if boundary else enumerate(lengths):
        ids = torch.randint(10, 20000, (n,), device="cuda")
        rt.run("prefill", stage([seq], ids, torch.arange(n, device="cuda"), True))
    current = torch.randint(10, 20000, (maximum,), device="cuda")
    cases = []
    if boundary:
        lengths[:3] = [m["buffers"]["positions"]["domain"]["max"], 65535, 4095]
    for groups in [3] if boundary else sorted({1, 2, 3, 4, 8, 9, maximum}):
        if groups > maximum:
            continue
        for step in range(1 if boundary else 2):
            ids = current[:groups].clone()
            positions = torch.tensor(lengths[:groups], device="cuda")
            before = state[1 : groups + 1].clone()
            rt.run("decode_batch", stage(list(range(groups)), ids, positions))
            batch = buffers["logits"][:groups].float().clone()
            next_ids = buffers["next_token"][:groups].clone()
            after = state[1 : groups + 1].clone()
            kv_after = kv.clone()
            singleton = []
            for seq in range(groups):
                state[seq + 1].copy_(before[seq])
                rt.run("decode", stage([seq], ids[seq : seq + 1], positions[seq : seq + 1]))
                singleton.append(buffers["logits"][0].float().clone())
            state[1 : groups + 1].copy_(after)
            kv.copy_(kv_after)
            single = torch.stack(singleton)
            a, b = single.log_softmax(-1), batch.log_softmax(-1)
            kl = (a.exp() * (a - b)).sum(-1)
            top = single.topk(2, -1).values
            mismatch = single.argmax(-1) != batch.argmax(-1)
            margins = (top[:, 0] - top[:, 1])[mismatch]
            result = {
                "batch": groups,
                "step": step,
                "mean_kl": float(kl.mean()),
                "max_kl": float(kl.max()),
                "top1_matches": int((~mismatch).sum()),
                "mismatch_margins": margins.tolist(),
            }
            cases.append(result)
            print(json.dumps(result), flush=True)
            if not kl.isfinite().all() or float(kl.max()) > 0.02 or (margins > 0.5).any():
                raise ValueError("batched/singleton outputs disagree")
            current[:groups] = next_ids
            for seq in range(groups):
                lengths[seq] += 1
            del before, after, kv_after, singleton, single, batch
    report = {
        "device": torch.cuda.get_device_name(),
        "max_batch": maximum,
        "max_context": m["buffers"]["positions"]["domain"]["max"] + 1,
        "kv_pages_shuffled": True,
        "max_context_boundary": boundary,
        "boundary_lengths": [m["buffers"]["positions"]["domain"]["max"] + 1, 65536, 4096]
        if boundary
        else None,
        "boundary_prior_state": "zero external cache, shared read-only zero pages" if boundary else None,
        "states_restored_before_singletons": True,
        "kl_limit": 0.02,
        "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/batch.json"))
    p.add_argument("--boundary-only", action="store_true")
    a = p.parse_args()
    check(a.bundle.resolve(), a.checkpoint.resolve(), a.out, a.boundary_only)
