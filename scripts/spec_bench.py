#!/usr/bin/env python3
"""Replay lm-eval GSM8K chat prompts at small concurrency and retain every HTTP result."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from lm_eval.filters.extraction import RegexFilter
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager, get_task_dict


def request(url, body=None):
    req = Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=1800) as response:
        payload = response.read().decode()
    return payload


def counters(text):
    result = {}
    for line in text.splitlines():
        if line.startswith(
            (
                "vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
            )
        ):
            name, value = line.rsplit(" ", 1)
            result[name.split("{")[0]] = float(value)
    return result


def main(args):
    task = get_task_dict(["gsm8k"], TaskManager())["gsm8k"]
    if args.prompts is not None:
        prompts = json.loads(args.prompts.read_text())
    else:
        api = LocalChatCompletion(
            model="qwen3.8-27b",
            base_url=args.base + "/v1/chat/completions",
            tokenizer_backend="huggingface",
            tokenizer=str(args.checkpoint),
            tokenized_requests=False,
        )
        task.set_config("num_fewshot", 8)
        task.set_fewshot_seed(1234)
        task.build_all_requests(
            limit=8,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            chat_template=api.apply_chat_template,
            tokenizer_name=api.tokenizer_name,
        )
        prompts = [
            {
                "id": r.doc_id,
                "question": r.doc["question"],
                "gold": r.doc["answer"].rsplit("#### ", 1)[-1].strip(),
                "messages": json.loads(r.args[0].prompt),
            }
            for r in task.instances
        ]
    strict = RegexFilter(regex_pattern=r"#### (\-?[0-9\.\,]+)")
    flexible = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n")
    # Warm the exact prompts for both engines; generation and scoring use all eight questions.
    for example in prompts:
        request(
            args.base + "/v1/chat/completions",
            {
                "model": "qwen3.8-27b",
                "messages": example["messages"],
                "temperature": 0,
                "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    summaries = []
    for concurrency in args.concurrency:
        before = counters(request(args.base + "/metrics"))
        json.loads(request(args.trace + "/kern/trace/rotate", {}))
        started = time.monotonic()
        results = []

        def complete(example):
            body = {
                "model": "qwen3.8-27b",
                "messages": example["messages"],
                "temperature": 0,
                "max_tokens": 32768,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            start = time.monotonic()
            response = json.loads(request(args.base + "/v1/chat/completions", body))
            elapsed = time.monotonic() - start
            choice = response["choices"][0]
            text = choice["message"]["content"]
            if text is None:
                raise ValueError("Missing final answer")
            doc = {"question": example["question"], "answer": "#### " + example["gold"]}
            scores = {}
            for name, extract in [("strict", strict), ("flexible", flexible)]:
                predicted = extract.apply([[text]], [doc])[0][0]
                scores[name] = task.process_results(doc, [predicted])["exact_match"]
            return {
                "id": example["id"],
                "gold": example["gold"],
                "seconds": elapsed,
                "response": response,
                "scores": scores,
            }

        with (
            (args.out / f"bs{concurrency}.jsonl").open("w") as output,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            pending = [pool.submit(complete, p) for p in prompts]
            for future in as_completed(pending):
                row = future.result()
                results.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "concurrency": concurrency,
                            "completed": len(results),
                            "total": len(prompts),
                            "id": row["id"],
                            "scores": row["scores"],
                            "tokens": row["response"]["usage"]["completion_tokens"],
                        }
                    ),
                    flush=True,
                )
        elapsed = time.monotonic() - started
        after = counters(request(args.base + "/metrics"))
        drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get(
            "vllm:spec_decode_num_drafts_total", 0
        )
        accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0
        )
        generated = sum(r["response"]["usage"]["completion_tokens"] for r in results)
        trace_after = json.loads(request(args.trace + "/kern/trace/rotate", {}))
        # Only portable artifact names belong in reports.
        summary = {
            "concurrency": concurrency,
            "requests": len(results),
            "prompt_cache": "warmed",
            "cached_prompt_tokens": after.get("vllm:prefix_cache_hits_total", 0)
            - before.get("vllm:prefix_cache_hits_total", 0),
            "seconds": elapsed,
            "generated_tokens": generated,
            "end_to_end_tokens_per_second": generated / elapsed,
            "median_request_seconds": statistics.median(r["seconds"] for r in results),
            "strict_correct": sum(r["scores"]["strict"] for r in results),
            "flexible_correct": sum(r["scores"]["flexible"] for r in results),
            "truncated": sum(r["response"]["choices"][0]["finish_reason"] == "length" for r in results),
            "draft_rounds": drafts,
            "tokens_per_round": (accepted + drafts) / drafts if drafts else None,
            "trace_files": [Path(p).name for p in trace_after["closed"]],
        }
        summaries.append(summary)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--trace", default="http://127.0.0.1:8001")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    args = p.parse_args()
    if args.prompts is None and args.checkpoint is None:
        p.error("provide --checkpoint to build lm-eval prompts, or --prompts to replay saved ones")
    main(args)
