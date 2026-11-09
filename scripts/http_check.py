#!/usr/bin/env python3
"""Check the running HTTP server, including SSE and concurrent requests."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def check(base, out, concurrency):
    def request(path, body):
        return Request(
            base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )

    with urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    body = {"model": model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}

    def complete(_=None):
        with urlopen(request("/v1/completions", body), timeout=180) as response:
            return json.load(response)["choices"][0]["text"]

    first = complete()
    if first != " Paris.\nThe capital of Germany is":
        raise AssertionError(f"completion differs from the validated CLI output: {first!r}")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        queued = list(pool.map(complete, range(concurrency)))
    if queued != [first] * concurrency:
        raise AssertionError("queued requests changed the greedy continuation")
    streamed, done = "", False
    with urlopen(request("/v1/completions", body | {"stream": True}), timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", []):
                streamed += choice.get("text", "")
    if not done or streamed != first:
        raise AssertionError("SSE response differs from the non-streaming continuation")
    chat = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: hello"}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with urlopen(request("/v1/chat/completions", chat), timeout=180) as response:
        text = json.load(response)["choices"][0]["message"]["content"]
    if text != "hello":
        raise AssertionError(f"chat response differs: {text!r}")

    def sample(seed, temperature=0.8):
        payload = body | {
            "prompt": (
                "The capital of France is"
                if temperature == 0
                else "Invent a short name for a friendly robot:"
            ),
            "max_tokens": 8 if temperature == 0 else 32,
            "temperature": temperature,
            "top_k": 20,
            "top_p": 0.9,
            "min_p": 0.05,
            "seed": seed,
        }
        with urlopen(request("/v1/completions", payload), timeout=180) as response:
            result = json.load(response)
        if result["usage"]["completion_tokens"] <= 0:
            raise AssertionError("sampling returned no tokens")
        return result["choices"][0]["text"]

    sampled = sample(123)
    if sample(123) != sampled:
        raise AssertionError("same seed did not replay at singleton batch layout")
    if sample(124) == sampled:
        raise AssertionError("different seeds produced the same test continuation")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        mixed = list(pool.map(lambda j: sample(j, 0 if j % 2 == 0 else 0.8), range(concurrency)))
    if mixed[::2] != [first] * len(mixed[::2]):
        raise AssertionError("mixed-batch greedy continuations differ")
    sampled_stream = body | {"temperature": 0.8, "seed": 123, "stream": True}
    with urlopen(request("/v1/completions", sampled_stream), timeout=180) as response:
        if not any(line.strip() == b"data: [DONE]" for line in response):
            raise AssertionError("sampled SSE did not finish")
    report = {
        "model": model,
        "completion_matches_cli": True,
        "queued_requests_match": True,
        "concurrent_requests": concurrency,
        "sse_matches": True,
        "chat_response": text,
        "same_seed_equal": True,
        "different_seed_changed": True,
        "mixed_sampling_requests": len(mixed),
        "sampled_sse": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("out/http.json"))
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    check(args.base_url.rstrip("/"), args.out, args.concurrency)
