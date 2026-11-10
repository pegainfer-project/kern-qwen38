"""Wire DSpine target verification to FlashInfer paged prefill and split-KV merge."""

import copy
import hashlib
import json
import re
import subprocess


def attach(m, kernels):
    maximum = m["vars"]["seqs"]["max"]
    splits = 32
    b = lambda n: {"buf": n}
    i = lambda n: {"i32": n}
    l = lambda n: {"i64": n}
    mul = lambda a, c: {"expr": {"mul": [a, c]}}
    for name, dtype, shape in [
        ("request", "i32", [maximum * splits * 2]),
        ("kv_tile", "i32", [maximum * splits * 2]),
        ("qo_tile", "i32", [maximum * splits * 2]),
        ("valid", "u8", [maximum * splits * 2]),
        ("merge", "i32", [maximum * 16 + 1]),
        ("output_indptr", "i32", [maximum + 1]),
        ("partial", "bf16", [maximum * 16 * splits, 6144]),
        ("lse", "f32", [maximum * 16 * splits, 24]),
    ]:
        m["buffers"]["verify.attn." + name] = {"dtype": dtype, "shape": shape, "kind": "workspace"}
    for name in ("attention_plan", "attention_fp8"):
        path = kernels / (name + ".cubin")
        m["modules"][name] = {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    params = (
        ["in buffer<i32>"] * 3
        + ["in buffer<u8>"]
        + ["out buffer<i32>"] * 3
        + ["out buffer<u8>", "out buffer<i32>", "out buffer<i32>", "i32"]
    )
    m["ops"]["verify_attention_plan"] = {
        "params": params,
        "impl": {
            "launches": [
                {
                    "module": "attention_plan",
                    "entry": "verify_attention_plan",
                    "params": params + ["i32"],
                    "grid": [1, 1, 1],
                    "block": [256, 1, 1],
                    "args": [{"param": j} for j in range(len(params))] + [i(splits)],
                }
            ]
        },
    }
    plan = copy.deepcopy(m["programs"]["decode_batch"]["calls"][0])
    assert plan["op"] == "batch_plan"
    plan["label"] = "verify.kv_plan"
    expand = {
        "label": "verify.attn_plan",
        "op": "verify_attention_plan",
        "args": [
            *[b("plan." + n) for n in ("merge", "request", "tile", "valid")],
            *[
                b("verify.attn." + n)
                for n in ("request", "kv_tile", "qo_tile", "valid", "merge", "output_indptr")
            ],
            {"var": "seqs"},
        ],
    }
    facts = json.loads((kernels / "build.json").read_text())["attention"]["fp8"]
    abi = facts["abi"]["prefill_g6"]
    # Shared operator: per-layer KV offsets and calibrated scales are arguments.
    params = (
        ["in buffer<bf16>", "out buffer<bf16>", "in state", "in state"]
        + ["in buffer<i32>"] * 8
        + ["in buffer<u8>"]
        + ["in buffer<i32>"] * 2
        + ["out buffer<bf16>", "out buffer<f32>", "i32", "i32", "f32", "f32"]
    )
    pointer_fields = {
        "q": 0,
        "o": 15,
        "lse": 16,
        "paged_kv.k_data": 2,
        "paged_kv.v_data": 3,
        "paged_kv.indices": 4,
        "paged_kv.indptr": 5,
        "paged_kv.last_page_len": 6,
        "request_indices": 7,
        "kv_tile_indices": 8,
        "kv_chunk_size_ptr": 9,
        "q_indptr": 10,
        "qo_tile_indices": 11,
        "block_valid_mask": 12,
        "o_indptr": 13,
    }
    fields = {name: {"param": n} for name, n in pointer_fields.items()}
    fields.update(
        {
            "num_qo_heads": i(24),
            "q_stride_n": i(6144),
            "q_stride_h": i(256),
            "paged_kv.num_heads": i(4),
            "paged_kv.batch_size": {"param": 17},
            "paged_kv.stride_page": i(16 * 2 * 4 * 64 * 256),
            "paged_kv.stride_n": i(256),
            "paged_kv.stride_h": i(64 * 256),
            "paged_kv.v_stride_page": i(16 * 2 * 4 * 64 * 256),
            "paged_kv.v_stride_n": i(256),
            "paged_kv.v_stride_h": i(64 * 256),
            "sm_scale": {"param": 19},
            "v_scale": {"param": 20},
            "partition_kv": {"u8": 1},
            "padded_batch_size": mul("seqs", splits * 2),
            "max_total_num_rows": {"param": 18},
        }
    )
    packed, covered = [], set()
    for name, value in fields.items():
        at = abi["offsets"][name]
        width = 8 if name in pointer_fields else 1 if "u8" in value else 4
        covered.update(range(at, at + width))
        packed.append({"at": at, **value})
    packed = [
        {"at": at, "u8": value} for at, value in enumerate(abi["bytes"]) if value and at not in covered
    ] + packed
    symbols = subprocess.check_output(
        ["cuobjdump", "-symbols", str(kernels / "attention_fp8.cubin")], text=True
    )
    merge = re.search(r"\b(_ZN\S*PersistentVariableLengthMergeStatesKernel\S*)", symbols).group(1)
    m["ops"]["verify_attention_flashinfer"] = {
        "params": params,
        "impl": {
            "launches": [
                {
                    "module": "attention_fp8",
                    "entry": "prefill",
                    "params": [f"bytes<{abi['size']}>"],
                    "args": [{"pack": {"size": abi["size"], "fields": packed}}],
                    "grid": [mul("seqs", splits * 2)["expr"], 1, 4],
                    "block": facts["prefill_block"],
                    "shared_mem": facts["prefill_shared"],
                },
                {
                    "module": "attention_fp8",
                    "entry": merge,
                    "params": [
                        "in buffer<bf16>",
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "out buffer<bf16>",
                        "i64",
                        "i32",
                        "i64",
                        "i32",
                    ],
                    "args": [
                        {"param": 15},
                        {"param": 16},
                        {"param": 14},
                        {"param": 1},
                        l(0),
                        {"param": 18},
                        l(0),
                        i(24),
                    ],
                    "grid": [mul("tokens", 24)["expr"], 1, 1],
                    "block": [16, 8, 1],
                    "shared_mem": 4 * 8 * 256 * 2 + 128 * 4,
                },
            ]
        },
    }

    def call(q, output, cache, layer, ks, vs, label):
        base = layer * 2 * 4 * 64 * 256
        return {
            "label": label,
            "op": "verify_attention_flashinfer",
            "args": [
                q,
                output,
                {"state": cache, "offset": base},
                {"state": cache, "offset": base + 4 * 64 * 256},
                *[b("plan." + n) for n in ("indices", "indptr", "last")],
                b("verify.attn.request"),
                b("verify.attn.kv_tile"),
                b("plan.kv_chunk"),
                b("q_indptr"),
                *[
                    b("verify.attn." + n)
                    for n in ("qo_tile", "valid", "output_indptr", "merge", "partial", "lse")
                ],
                {"var": "seqs"},
                {"var": "tokens"},
                {"f32": ks / 16},
                {"f32": vs},
            ],
        }

    return [plan, expand], call
