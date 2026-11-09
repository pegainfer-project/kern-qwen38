#!/usr/bin/env python3
"""Record an independent Transformers baseline and calibrate FP8 KV factors."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, FineGrainedFP8Config


def record(checkpoint, out):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Dequantize one shard at a time onto the GPU; no full BF16 host copy.
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        quantization_config=FineGrainedFP8Config(dequantize=True),
    ).eval()
    text = model.model.language_model
    maxima = {}
    handles = []
    for idx, layer in enumerate(text.layers):
        if not hasattr(layer, "self_attn"):
            continue
        maxima[str(idx)] = {"k": 0.0, "v": 0.0}

        def hook(kind, index):
            def collect(module, args, output):
                value = output.detach().float().abs().max().item()
                maxima[str(index)][kind] = max(maxima[str(index)][kind], value)

            return collect

        handles.append(layer.self_attn.k_norm.register_forward_hook(hook("k", idx)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook("v", idx)))
    calibration = [
        "The capital of France is",
        "Write a Python function that computes the greatest common divisor using Euclid's algorithm.",
        "Explain why a database write ahead log must be persisted before the corresponding data page.",
        "请解释数据库事务的原子性和持久性，并举一个转账的例子。",
    ]
    results = []
    with torch.inference_mode():
        for idx, prompt in enumerate(calibration):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            result = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            torch.save(result.logits[0, -1].float().cpu(), out / f"logits-{idx}.pt")
            # Reuse the resulting cache for a short generation check.
            generated = []
            logits = result.logits
            cache = result.past_key_values
            for step in range(8):
                token = logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(token.item()))
                result = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = result.logits
                cache = result.past_key_values
            results.append(
                {
                    "prompt": prompt,
                    "prompt_ids": ids[0].tolist(),
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(generated),
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            del result, cache, logits
    for handle in handles:
        handle.remove()
    # Partial rotation can increase an individual K component; reserve a sqrt(2) bound
    # and 25% calibration headroom. Scales stay fixed while cached pages are alive.
    scales = {
        idx: {"k": max(m["k"] * 2**0.5 * 1.25 / 448, 1e-8), "v": max(m["v"] * 1.25 / 448, 1e-8)}
        for idx, m in maxima.items()
    }
    (out / "kv-scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    (out / "reference.json").write_text(
        json.dumps(
            {
                "backend": "Transformers dequantized checkpoint BF16",
                "torch": torch.__version__,
                "prompts": results,
                "observed_maxima": maxima,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    record(a.checkpoint, a.out)
