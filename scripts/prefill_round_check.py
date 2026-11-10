#!/usr/bin/env python3
"""Retain greedy speculative-round outputs and acceptance counters for an isolated candidate."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(args.prompts.read_text())
    results = []
    for item in prompts:
        case = {"source": item["source"], "prompt_tokens": len(item["prompt"])}
        for side, bundle in [("baseline", args.baseline), ("candidate", args.candidate)]:
            command = [
                str(args.kern),
                "run",
                "--config",
                str(bundle / "kern.toml"),
                "--rows",
                "16",
                "--steps",
                "256",
                "--capacity",
                str(len(item["prompt"]) + 1024),
            ]
            for offset in range(0, len(item["prompt"]), 1024):
                command += ["--prompt-ids", ",".join(map(str, item["prompt"][offset : offset + 1024]))]
            run = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
            (args.out / f"{item['source']}-{side}.log").write_text(run.stdout)
            if run.returncode:
                raise RuntimeError(run.stdout[-3000:])
            tokens = re.search(r"generated ids: (\[[^\n]+\])", run.stdout)
            stats = re.search(
                r"(\d+) tokens generated in (\d+) steps of 16 rows, ([\d.]+) ms/step .*?, ([\d.]+) tokens/step, ([\d.]+)% of the 15 drafted rows taken",
                run.stdout,
            )
            if not tokens or not stats:
                raise ValueError(f"Missing output or round statistics: {item['source']}/{side}")
            count, rounds, ms, per_round, accepted = stats.groups()
            case[side] = {
                "generated_ids": json.loads(tokens.group(1)),
                "tokens": int(count),
                "rounds": int(rounds),
                "ms_per_round": float(ms),
                "tokens_per_round": float(per_round),
                "draft_acceptance_percent": float(accepted),
            }
        case["identical_output"] = case["baseline"]["generated_ids"] == case["candidate"]["generated_ids"]
        results.append(case)
        (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in case.items() if k not in ("baseline", "candidate")}
                | {
                    side: {k: v for k, v in case[side].items() if k != "generated_ids"}
                    for side in ("baseline", "candidate")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("kern", "baseline", "candidate", "prompts", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    check(p.parse_args())
