#!/usr/bin/env python3
"""Exercise real long prefill and compare aligned chunk profiles through the CLI."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def check(kern, bundle, checkpoint, out, tokens):
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    unit = tokenizer.encode("A database transaction writes its log before its modified pages. ")
    tail = tokenizer.encode("The capital of Japan is")
    ids = (unit * (tokens // len(unit) + 1))[: tokens - len(tail)] + tail
    cases = []
    for chunk in (1024, 512):
        command = [
            str(kern),
            "run",
            "--config",
            str(bundle / "kern.toml"),
            "--prompt-ids",
            ",".join(map(str, ids)),
            "--steps",
            "8",
            "--capacity",
            str(tokens * 2),
            "--chunk",
            str(chunk),
        ]
        run = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        (out / f"chunk-{chunk}.log").write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(run.stdout[-4000:])
        match = re.search(r"generated ids: (\[[^\n]+\])", run.stdout)
        if not match:
            raise ValueError("missing generated token IDs")
        generated = json.loads(match.group(1))
        case = {
            "prompt_tokens": len(ids),
            "chunk": chunk,
            "generated_ids": generated,
            "text": tokenizer.decode(generated),
            "prefill": re.search(r"prefill: [^\n]+", run.stdout).group(0),
        }
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = {"aligned_chunks_match": cases[0]["generated_ids"] == cases[1]["generated_ids"], "cases": cases}
    (out / "long-context.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["aligned_chunks_match"]:
        raise ValueError("long-context aligned chunk profiles differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kern", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/long"))
    parser.add_argument("--tokens", type=int, default=16384)
    args = parser.parse_args()
    check(args.kern.resolve(), args.bundle.resolve(), args.checkpoint.resolve(), args.out, args.tokens)
