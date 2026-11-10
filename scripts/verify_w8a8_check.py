#!/usr/bin/env python3
"""Compare verification logits and accepted prefixes on identical speculative rounds."""

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

import torch
from tokenizers import Tokenizer


def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    config = tomllib.loads(args.config.read_text())["targets"][args.target]
    checkpoint = Path(config["weights"][0])
    if not checkpoint.is_absolute():
        checkpoint = args.config.parent / checkpoint
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    text = "A database transaction writes its log before its modified pages. Recovery replays committed records. "
    unit = tokenizer.encode(text).ids
    tail = tokenizer.encode("Explain why write ahead logging makes crash recovery possible.").ids
    report = []
    for length in args.contexts:
        ids = (unit * (length // len(unit) + 1))[: length - len(tail)] + tail
        for side, manifest, kernels in [
            ("baseline", args.baseline, args.baseline_kernels),
            ("candidate", args.candidate, args.candidate_kernels),
        ]:
            directory = args.out / f"{length}-{side}"
            command = [
                str(args.kern),
                "run",
                args.target,
                "--config",
                str(args.config),
                "--manifest",
                str(manifest),
                "--kernels",
                str(kernels),
                "--capacity",
                str(length + 1024),
                "--rows",
                "16",
                "--chunk",
                "1024",
                "--probe-dir",
                str(directory),
                "--probe-labels",
                "dspine_proposals,verify.lm_head,spec_count",
                "--probe-steps",
                "4",
            ]
            for offset in range(0, len(ids), 1024):
                command.extend(["--prompt-ids", ",".join(map(str, ids[offset : offset + 1024]))])
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
            (args.out / f"{length}-{side}.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(completed.stdout[-4000:])
        history_matches = True
        for step in range(4):
            a, b = (args.out / f"{length}-{s}" for s in ("baseline", "candidate"))
            tag = f"decode{step}"
            proposals_match = (a / f"{tag}.dspine_proposals.bin").read_bytes() == (
                b / f"{tag}.dspine_proposals.bin"
            ).read_bytes()
            counts = [
                int.from_bytes((p / f"{tag}.spec_count.bin").read_bytes()[:4], "little", signed=True)
                for p in (a, b)
            ]
            tokens = [
                (p / f"{tag}.tokens.bin").read_bytes()[: count * 8]
                for p, count in zip((a, b), counts, strict=True)
            ]
            case = {
                "context": length,
                "step": step,
                "same_history": history_matches,
                "same_proposals": proposals_match,
                "accepted": counts,
                "same_accepted_tokens": tokens[0] == tokens[1],
            }
            if history_matches and proposals_match:
                logits = [
                    torch.frombuffer(bytearray((p / f"{tag}.verify.bin").read_bytes()), dtype=torch.bfloat16)
                    .float()
                    .reshape(16, -1)
                    for p in (a, b)
                ]
                if not all(torch.isfinite(v).all() for v in logits):
                    raise ValueError("Nonfinite verification logits")
                kl = (logits[0].softmax(-1) * (logits[0].log_softmax(-1) - logits[1].log_softmax(-1))).sum(-1)
                flipped = (logits[0].argmax(-1) != logits[1].argmax(-1)).nonzero().flatten()
                top = logits[0].topk(2, dim=-1).values
                case.update(
                    mean_kl=kl.mean().item(),
                    max_row_kl=kl.max().item(),
                    top1_flips=len(flipped),
                    flipped_rows=[
                        {
                            "row": row,
                            "baseline_margin": float(top[row, 0] - top[row, 1]),
                            "before_accepted_boundary": row < counts[0],
                        }
                        for row in flipped.tolist()
                    ],
                )
            history_matches = history_matches and tokens[0] == tokens[1]
            report.append(case)
            print(json.dumps(case), flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    compared = [r for r in report if "mean_kl" in r]
    if (
        len([r for r in compared if r["step"] == 0]) != len(args.contexts)
        or max(r["max_row_kl"] for r in compared) > args.kl
    ):
        raise ValueError(f"Verification logits exceed KL limit {args.kl}; inspect report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("kern", "config", "baseline", "baseline-kernels", "candidate", "candidate-kernels", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target", default="qwen38")
    p.add_argument("--kl", type=float, default=0.01)
    p.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 8192, 32768])
    check(p.parse_args())
