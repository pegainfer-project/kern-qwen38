#!/usr/bin/env python3
"""Check the running HTTP server, including SSE and concurrent requests."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
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
    try:
        with urlopen(request("/v1/completions", body | {"temperature": 0.8}), timeout=10):
            raise AssertionError("the greedy manifest accepted stochastic sampling")
    except HTTPError as error:
        if error.code != 400:
            raise
    report = {
        "model": model,
        "completion_matches_cli": True,
        "queued_requests_match": True,
        "concurrent_requests": concurrency,
        "sse_matches": True,
        "chat_response": text,
        "unsupported_sampling_status": 400,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("out/http.json"))
    parser.add_argument("--concurrency", type=int, default=128)
    args = parser.parse_args()
    check(args.base_url.rstrip("/"), args.out, args.concurrency)
