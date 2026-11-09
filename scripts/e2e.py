#!/usr/bin/env python3
"""Run native kern against recorded Transformers prompts and compare logits/tokens."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch


def check(kern, bundle, reference, out):
    out.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((reference / "reference.json").read_text())
    cases = []
    for index, item in enumerate(recorded["prompts"]):
        probe = out / f"prompt-{index}"
        cmd = [
            str(kern),
            "run",
            "--config",
            str(bundle / "kern.toml"),
            "--prompt-ids",
            ",".join(map(str, item["prompt_ids"])),
            "--probe-dir",
            str(probe),
            "--probe-labels",
            "lm_head",
            "--probe-steps",
            "7",
        ]
        completed = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        (out / f"prompt-{index}.log").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-4000:])
        logits = torch.frombuffer(
            bytearray((probe / "chunk.lm_head.bin").read_bytes()), dtype=torch.bfloat16
        ).float()
        expected = torch.load(reference / f"logits-{index}.pt", weights_only=True)
        if logits.shape != expected.shape or not logits.isfinite().all():
            raise ValueError("invalid native logits")
        kl = float((expected.softmax(-1) * (expected.log_softmax(-1) - logits.log_softmax(-1))).sum())
        tags = ["chunk"] + [f"decode{step}" for step in range(7)]
        tokens = [
            int.from_bytes((probe / f"{tag}.tokens.bin").read_bytes()[:8], "little", signed=True)
            for tag in tags
        ]
        matches = sum(a == b for a, b in zip(tokens, item["generated_ids"], strict=True))
        result = {
            "prompt": item["prompt"],
            "kl_reference_native": kl,
            "cosine": float(torch.nn.functional.cosine_similarity(logits, expected, dim=0)),
            "max_logit_error": float((logits - expected).abs().max()),
            "generated_ids": tokens,
            "reference_ids": item["generated_ids"],
            "token_matches": matches,
            "tokens": 8,
        }
        cases.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if kl > 0.1:
            raise ValueError(f"prompt {index}: KL {kl} exceeds 0.1")
        if tokens != item["generated_ids"]:
            raise ValueError(f"prompt {index}: generated tokens differ from the reference")
    prompt = (
        "A database transaction writes its log before its modified pages. " * 20
    ) + "The capital of Japan is"
    continuity = []
    for chunk, eager in ((128, False), (64, False), (128, True)):
        command = [
            str(kern),
            "run",
            "--config",
            str(bundle / "kern.toml"),
            "--prompt",
            prompt,
            "--steps",
            "16",
            "--chunk",
            str(chunk),
        ]
        if eager:
            command.append("--eager")
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        (out / f"continuity-{chunk}-{'eager' if eager else 'graph'}.log").write_text(run.stdout)
        match = re.search(r"generated ids: (\[[^\n]+\])", run.stdout)
        if match is None:
            raise ValueError("kern did not report generated IDs")
        tokens = json.loads(match.group(1))
        continuity.append({"chunk": chunk, "eager": eager, "generated_ids": tokens})
        print(json.dumps(continuity[-1]), flush=True)
    if any(case["generated_ids"] != continuity[0]["generated_ids"] for case in continuity[1:]):
        raise ValueError("chunk boundary or graph/eager generation differs")
    report = {
        "reference": "Transformers dequantized FP8 checkpoint, BF16 compute and FP32 recurrent state",
        "candidate": "kern native FP8 weights, W8A8 prefill/W8A16 decode, BF16 recurrent state",
        "kl_limit": 0.1,
        "continuity": continuity,
        "cases": cases,
    }
    (out / "e2e.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kern", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    check(a.kern.resolve(), a.bundle.resolve(), a.reference.resolve(), a.out.resolve())
