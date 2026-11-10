"""Add a native DSpine draft/ReplaySSM round to the target manifest."""

import copy
import hashlib
import json

import sampling
from gemm_manifest import linear

H, V, F, BLOCK = 5120, 248320, 17408, 16
T = {"var": "tokens"}
S = {"var": "seqs"}
b = lambda n, o=0: {"buf": n, **({"offset": o} if o else {})}
st = lambda n, o=0: {"state": n, **({"offset": o} if o else {})}
i = lambda n: {"i32": n}
l = lambda n: {"i64": n}
f = lambda n: {"f32": n}
expr = lambda x: {"expr": x}
ceil = lambda x, d: {"ceil_div": [x, d]}
mul = lambda x, y: {"mul": [x, y]}


def augment(m, kernels, draft, tables, scales):
    cfg = json.loads((draft / "config.json").read_text())
    if (cfg["block_size"], cfg["dflash_config"]["target_layer_ids"], cfg["sliding_window"]) != (
        16,
        [1, 16, 31, 46, 61],
        8192,
    ):
        raise ValueError("Expected the five-layer Qwen3.8 DSpine-b16 checkpoint")
    if m["vars"]["tokens"]["max"] < m["vars"]["seqs"]["max"] * 16:
        raise ValueError("DSpine requires chunk >= max-seqs * 16")
    if m["states"]["kv"]["bytes_per_token"] != 32768:
        raise ValueError("The DSpine bundle currently requires calibrated FP8 target KV")
    if not (tables / "model.safetensors").exists():
        raise ValueError("Run scripts/prepare_dspine.py first")
    build = json.loads((kernels / "dsl.json").read_text())
    dsl = build["modules"]
    tuning = build["gemm_tuning"]
    if not tuning:
        raise ValueError("Build with --dspine --gemm-tuning <profile> before generating a DSpine bundle")
    buffers, ops, modules = m["buffers"], m["ops"], m["modules"]
    maximum = m["vars"]["seqs"]["max"]
    rows = maximum * 16
    line = 3 * 10240 * 2 + 48 * 128 * 128 * 2
    taps = cfg["dflash_config"]["target_layer_ids"]
    load = m["programs"]["load"]["calls"]

    def module(name):
        path = kernels / (name + ".cubin")
        modules[name] = {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return name

    def buffer(name, dtype, shape, kind="workspace", **kwargs):
        buffers[name] = {"dtype": dtype, "shape": shape, "kind": kind, **kwargs}
        return b(name)

    def weight(name, shape, tensors=None):
        return buffer("d." + name, "bf16", shape, "weight", bind=[{"tensor": t} for t in (tensors or [name])])

    def native(name, entry, params, grid, block=256):
        ops[name] = {
            "params": params,
            "impl": {
                "launches": [
                    {
                        "module": module("dspine_state"),
                        "entry": entry,
                        "params": params,
                        "grid": grid,
                        "block": [block, 1, 1],
                        "args": [{"param": j} for j in range(len(params))],
                    }
                ]
            },
        }

    def call(name, *args, label=None):
        return {"label": label or name, "op": name, "args": list(args)}

    outputs = {
        "dspine_norm": {"Y"},
        "dspine_read": {"Raw", "Normalized", "RMS"},
        "dspine_codes": {"Codes"},
        "dspine_message": {"Message"},
        "dspine_finish": {"Output", "Normed"},
        "dspine_topk_max": {"CM"},
        "dspine_topk": {"OUT"},
        "dspine_table": {"Out", "Confidence"},
        "dspine_chain": {"Out", "ConfidenceOut"},
        "dspine_ids": {"Anchor", "Ids"},
        "dspine_proposals": {"Ids"},
        "dspine_skip": {"Y"},
        "dspine_rope": {"Q", "K", "V"},
        "dspine_rope_ctx": {"Q", "K", "V"},
        "dspine_store": {"Cache"},
        "dspine_store_pre": {"Cache"},
        "dspine_attention": {"Output"},
        "verify_attention": {"Output"},
        "replay_verify": {"o", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache"},
        "replay_commit": {"h0"},
    }

    def dc(name, grid, values, label=None):
        record = dsl[name]
        args = [values[n] for n in record["arguments"]]
        if name not in ops:
            params = []
            for arg, typ, value in zip(record["arguments"], record["signature"], args, strict=True):
                if typ.startswith("*"):
                    access = "out" if arg in outputs[name] else "in"
                    if "state" in value:
                        params.append(("inout" if access == "out" else access) + " state")
                    else:
                        dtype = {"fp32": "f32", "fp8e4nv": "fp8e4m3"}.get(typ[1:], typ[1:])
                        params.append(f"{access} buffer<{dtype}>")
                else:
                    params.append({"fp32": "f32"}.get(typ, typ))
            ops[name] = {
                "params": params,
                "impl": {
                    "launches": [
                        {
                            "module": module(name),
                            "entry": record["entry"],
                            "params": params + record["abi_tail"],
                            "grid": grid,
                            "block": record["block"],
                            "shared_mem": record["shared_mem"],
                            "args": [{"param": j} for j in range(len(args))] + [l(0), l(0)],
                        }
                    ]
                },
            }
        return call(name, *args, label=label)

    def gemm(label, x, w, y, n, k, count=T):
        return call("gemm_bf16", x, w, y, count, i(n), i(k), label=label)

    def norm(x, w, y, label):
        return dc("dspine_norm", ["tokens", 1, 1], {"X": x, "W": w, "Y": y}, label)

    def copy_rows(label, src, dst, width):
        return call("spec_copy", src, dst, expr(mul("tokens", width)), label=label)

    native(
        "spec_copy",
        "spec_copy",
        ["in buffer<bf16>", "out buffer<bf16>", "i32"],
        [ceil(mul("tokens", 16384), 256), 1, 1],
    )
    native("spec_tap", "spec_tap", ["in buffer<bf16>", "out buffer<bf16>", "i32"], ["tokens", 20, 1])
    native(
        "spec_conv",
        "spec_conv",
        ["in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "out buffer<bf16>", "i64"],
        [16, 40, "seqs"],
    )
    native(
        "spec_conv_commit",
        "spec_conv_commit",
        ["in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i32>", "i64"],
        ["seqs", 40, 1],
    )
    native(
        "spec_count",
        "spec_count",
        ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>"],
        ["seqs", 1, 1],
        32,
    )
    native(
        "spec_constants",
        "spec_constants",
        ["out buffer<i32>", "out buffer<i32>", "out buffer<i64>", "i32"],
        [ceil(m["vars"]["tokens"]["max"], 256), 1, 1],
    )
    for name, dtype, width in [
        ("h", "bf16", H),
        ("x", "bf16", H),
        ("branch", "bf16", H),
        ("read", "bf16", H),
        ("raw", "bf16", H),
        ("codes", "bf16", 1024),
        ("projected", "bf16", 1024),
        ("packed", "bf16", 1536),
        ("message", "bf16", 512),
        ("delta", "bf16", H),
        ("qkv", "bf16", 6144),
        ("q", "bf16", 4096),
        ("k", "bf16", 1024),
        ("v", "bf16", 1024),
        ("attn", "bf16", 4096),
        ("gate_up", "bf16", 2 * F),
        ("act", "bf16", F),
        ("taps", "bf16", 5 * H),
        ("fc", "bf16", H),
        ("ctx", "bf16", H),
        ("ctxkv", "bf16", 2048),
        ("ctxk", "bf16", 1024),
        ("ctxv", "bf16", 1024),
    ]:
        buffer("d." + name, dtype, ["tokens", width])
    buffer("d.rms", "f32", ["tokens"])
    for name, dtype, shape in [
        ("anchor", "i64", ["seqs"]),
        ("ids", "i64", ["tokens"]),
        ("verify_ids", "i64", ["tokens"]),
        ("draft", "i64", ["seqs", 15]),
        ("candidates", "i64", ["seqs", 15, 16]),
        ("table", "i64", ["seqs", 15, 16]),
        ("topk_max", "f32", ["seqs", 15, 3880]),
        ("last", "bf16", ["seqs", 15, H]),
        ("logits", "bf16", ["seqs", 15, V]),
        ("target_logits", "bf16", ["tokens", V]),
        ("zero", "i32", [m["vars"]["tokens"]["max"]]),
        ("one", "i32", [m["vars"]["tokens"]["max"]]),
        ("indices", "i64", [m["vars"]["tokens"]["max"]]),
    ]:
        buffer("d." + name, dtype, shape, "carry" if name in ("zero", "one", "indices") else "workspace")
    buffer("verify_tokens", "i64", ["seqs", 16], "output", fill="tokens", domain={"min": 0, "max": V - 1})
    buffer("accepted", "i32", ["seqs"], "output", fill="count", domain={"min": 1, "max": 16})
    buffer(
        "d.lines",
        "i32",
        [1, "seqs"],
        "input",
        domain={"index_into": "draft_kv", "stride": 5 * 8208 * 2048 * 2},
    )
    m["states"]["draft_kv"] = {"bytes_per_seq": 5 * 8208 * 2048 * 2}
    for name, dtype, width in [
        ("raw_qkv", "bf16", 16384),
        ("D", "bf16", 48 * 16 * 128),
        ("K", "bf16", 16 * 16 * 128),
        ("G", "f32", 48 * 16),
        ("Dr", "bf16", 48 * 16 * 128),
        ("Kr", "bf16", 16 * 16 * 128),
    ]:
        shape = [48, rows, width] if name == "raw_qkv" else [48, maximum, width]
        buffer("replay." + name, dtype, shape, "carry")
    load.append(
        call("spec_constants", b("d.zero"), b("d.one"), b("d.indices"), i(m["vars"]["tokens"]["max"]))
    )
    fc = weight("fc.weight", [H, 5 * H])
    hidden_norm = weight("hidden_norm.weight", [H])
    final_norm = weight("norm.weight", [H])
    packed = weight("bridge.packed", [1536, 1024], ["dspine.packed"])
    codebook = weight("selected_prefix_bridge.codebook", [V, 1024])
    gate_bias = weight("selected_prefix_bridge.gate.bias", [512])
    tg = weight("tg", [V, 512], ["dspine.tg"])
    tsv = weight("tsv", [V, 1024], ["dspine.tsv"])
    # Explicit slice bindings keep the original checkpoint unchanged.
    early = []
    writes = []
    for layer in range(5):
        if layer < 4:
            buffer(
                f"d.read.{layer}",
                "bf16",
                [1024, H],
                "weight",
                bind=[{"tensor": "selected_prefix_bridge.early_read_weights", "rows": [layer, layer + 1]}],
            )
            early.append(b(f"d.read.{layer}"))
        else:
            early.append(weight("selected_prefix_bridge.read_weight", [1024, H]))
        buffer(
            f"d.write.{layer}",
            "bf16",
            [H, 512],
            "weight",
            bind=[{"tensor": "selected_prefix_bridge.feature_write_weights", "rows": [layer, layer + 1]}],
        )
        writes.append(b(f"d.write.{layer}"))
    layer_weights = []
    for layer in range(5):
        prefix = f"layers.{layer}."
        w = {}
        for key, shape in [
            ("input_layernorm.weight", [H]),
            ("post_attention_layernorm.weight", [H]),
            ("self_attn.q_norm.weight", [128]),
            ("self_attn.k_norm.weight", [128]),
            ("self_attn.o_proj.weight", [H, 4096]),
            ("mlp.down_proj.weight", [H, F]),
        ]:
            w[key] = weight(prefix + key, shape)
        w["qkv"] = weight(
            prefix + "qkv.weight",
            [6144, H],
            [prefix + "self_attn." + n + ".weight" for n in ("q_proj", "k_proj", "v_proj")],
        )
        w["kv"] = weight(
            prefix + "kv.weight",
            [2048, H],
            [prefix + "self_attn." + n + ".weight" for n in ("k_proj", "v_proj")],
        )
        w["gate_up"] = weight(
            prefix + "gate_up.weight",
            [2 * F, H],
            [prefix + "mlp." + n + ".weight" for n in ("gate_proj", "up_proj")],
        )
        layer_weights.append(w)

    def context(prefill):
        result = [
            gemm("context.fc", b("d.taps"), fc, b("d.fc"), H, 5 * H),
            norm(b("d.fc"), hidden_norm, b("d.ctx"), "context.norm"),
        ]
        for layer, w in enumerate(layer_weights):
            result += [
                gemm(f"context.{layer}.kv", b("d.ctx"), w["kv"], b("d.ctxkv"), 2048, H),
                dc(
                    "dspine_rope_ctx",
                    ["tokens", 8, 1],
                    {
                        "X": b("d.ctxkv"),
                        "QW": w["self_attn.k_norm.weight"],
                        "KW": w["self_attn.k_norm.weight"],
                        "Positions": b("positions"),
                        "Q": b("d.ctxk"),
                        "K": b("d.ctxk"),
                        "V": b("d.ctxv"),
                    },
                    f"context.{layer}.rope",
                ),
                dc(
                    "dspine_store_pre" if prefill else "dspine_store",
                    ["tokens", 4, 1],
                    {
                        "K": b("d.ctxk"),
                        "V": b("d.ctxv"),
                        "Positions": b("positions"),
                        "Lines": b("d.lines"),
                        "Cache": st("draft_kv", layer * 8208 * 2048 * 2),
                        "TOKENS": T,
                    },
                    f"context.{layer}.store",
                ),
            ]
        return result

    def with_taps(calls):
        result = []
        for c in calls:
            result.append(c)
            for j, layer in enumerate(taps):
                if c["label"] == f"l{layer}.residual":
                    result.append(call("spec_tap", b("h"), b("d.taps"), i(j), label=f"l{layer}.draft_tap"))
        return result

    m["programs"]["prefill"]["calls"] = with_taps(m["programs"]["prefill"]["calls"]) + context(True)
    draft_calls = [
        dc(
            "dspine_ids",
            ["seqs", 1, 1],
            {"Input": b("token_ids"), "Anchor": b("d.anchor"), "Ids": b("d.ids")},
        )
    ]
    embed = next(c["args"][1] for c in m["programs"]["decode"]["calls"] if c["label"] == "embed")
    draft_calls.append(call("embed", b("d.ids"), embed, b("d.h")))
    draft_calls.append(
        norm(b("d.h"), layer_weights[0]["input_layernorm.weight"], b("d.x"), "draft.input_norm")
    )
    for layer, w in enumerate(layer_weights):
        p = f"draft.{layer}"
        draft_calls += [
            gemm(p + ".qkv", b("d.x"), w["qkv"], b("d.qkv"), 6144, H),
            dc(
                "dspine_rope",
                ["tokens", 40, 1],
                {
                    "X": b("d.qkv"),
                    "QW": w["self_attn.q_norm.weight"],
                    "KW": w["self_attn.k_norm.weight"],
                    "Positions": b("positions"),
                    "Q": b("d.q"),
                    "K": b("d.k"),
                    "V": b("d.v"),
                },
                p + ".rope",
            ),
            dc(
                "dspine_attention",
                ["seqs", 32, 1],
                {
                    "Q": b("d.q"),
                    "K": b("d.k"),
                    "V": b("d.v"),
                    "Cache": st("draft_kv", layer * 8208 * 2048 * 2),
                    "Lines": b("d.lines"),
                    "Pages": b("pages"),
                    "Lengths": b("lengths"),
                    "Output": b("d.attn"),
                    "TABLE": i(m["buffers"]["pages"]["shape"][1]),
                    "KS": f(1.0),
                    "VS": f(1.0),
                },
                p + ".attention",
            ),
            gemm(p + ".out", b("d.attn"), w["self_attn.o_proj.weight"], b("d.branch"), H, 4096),
            call("add", b("d.h"), b("d.h"), b("d.branch"), expr(mul("tokens", H))),
            norm(b("d.h"), w["post_attention_layernorm.weight"], b("d.x"), p + ".post_norm"),
            gemm(p + ".gate_up", b("d.x"), w["gate_up"], b("d.gate_up"), 2 * F, H),
            call("silu", b("d.act"), b("d.gate_up")),
            gemm(p + ".down", b("d.act"), w["mlp.down_proj.weight"], b("d.branch"), H, F),
            call("add", b("d.h"), b("d.h"), b("d.branch"), expr(mul("tokens", H))),
            dc(
                "dspine_read",
                ["tokens", 1, 1],
                {
                    "X": b("d.h"),
                    "Residual": b("d.h"),
                    "Raw": b("d.raw"),
                    "Normalized": b("d.read"),
                    "RMS": b("d.rms"),
                },
                p + ".read",
            ),
            gemm(p + ".project_code", b("d.read"), early[layer], b("d.projected"), 1024, H),
            dc(
                "dspine_codes",
                ["tokens", 1, 1],
                {
                    "Projected": b("d.projected"),
                    "Codebook": codebook,
                    "Anchors": b("d.anchor"),
                    "Codes": b("d.codes"),
                },
            ),
            gemm(p + ".packed", b("d.codes"), packed, b("d.packed"), 1536, 1024),
            dc(
                "dspine_message",
                ["tokens", 1, 1],
                {"Packed": b("d.packed"), "Bias": gate_bias, "Message": b("d.message")},
            ),
            gemm(p + ".delta", b("d.message"), writes[layer], b("d.delta"), H, 512),
            dc(
                "dspine_finish",
                ["tokens", 1, 1],
                {
                    "Raw": b("d.raw"),
                    "Delta": b("d.delta"),
                    "RMS": b("d.rms"),
                    "Valid": b("d.one"),
                    "Weight": final_norm
                    if layer == 4
                    else layer_weights[layer + 1]["input_layernorm.weight"],
                    "Output": b("d.h"),
                    "Normed": b("d.x"),
                },
                p + ".finish",
            ),
        ]
    lm = next(c["args"][1] for c in m["programs"]["decode"]["calls"] if c["label"] == "lm_head")
    draft_calls += [
        dc("dspine_skip", [mul("seqs", 15), 20, 1], {"X": b("d.x"), "Y": b("d.last")}),
        gemm("draft.lm_head", b("d.last"), lm, b("d.logits"), V, H, expr(mul("seqs", 15))),
        dc("dspine_topk_max", [mul("seqs", 15), 61, 1], {"X": b("d.logits"), "CM": b("d.topk_max")}),
        dc(
            "dspine_topk",
            [mul("seqs", 15), 1, 1],
            {"X": b("d.logits"), "CM": b("d.topk_max"), "OUT": b("d.candidates")},
        ),
        dc(
            "dspine_table",
            [mul("seqs", 15), 1, 1],
            {
                "Raw": b("d.raw"),
                "NormW": final_norm,
                "Cand": b("d.candidates"),
                "InputIds": b("d.ids"),
                "LmW": lm,
                "TG": tg,
                "TSV": tsv,
                "Query": b("d.packed"),
                "Rms": b("d.rms"),
                "GateB": gate_bias,
                "Out": b("d.table"),
                "Confidence": b("d.table"),
            },
        ),
        dc(
            "dspine_chain",
            ["seqs", 1, 1],
            {
                "Idx": b("d.table"),
                "Cand": b("d.candidates"),
                "Out": b("d.draft"),
                "Confidence": b("d.table"),
                "ConfidenceOut": b("d.draft"),
            },
        ),
        dc(
            "dspine_proposals",
            ["seqs", 1, 1],
            {"Anchor": b("d.anchor"), "Draft": b("d.draft"), "Ids": b("d.verify_ids")},
        ),
    ]
    # Flatten per-sequence parameters onto the 16 target rows. Each row's
    # Philox offset is its absolute next-token position, including the bonus row.
    for name, dtype in (
        ("temperature", "f32"),
        ("top_k", "i32"),
        ("top_p", "f32"),
        ("min_p", "f32"),
        ("seed", "i64"),
        ("lengths", "i32"),
    ):
        buffer("d.sample." + name, dtype, ["tokens"])
    native(
        "spec_sampling",
        "spec_sampling",
        [
            "in buffer<f32>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<i64>",
            "in buffer<i32>",
            "out buffer<f32>",
            "out buffer<i32>",
            "out buffer<f32>",
            "out buffer<f32>",
            "out buffer<i64>",
            "out buffer<i32>",
        ],
        ["seqs", 1, 1],
    )
    metadata = json.loads((kernels / "build.json").read_text())
    ops["verify_sample"] = sampling.op(
        V, rows, metadata["sampling"], sampling.target(metadata["architecture"])
    )
    ops["verify_sample"] = json.loads(json.dumps(ops["verify_sample"]).replace('"seqs"', '"tokens"'))
    sampling_names = ("temperature", "top_k", "top_p", "min_p", "seed", "lengths")

    def verify_linear(n, k):
        name = f"verify_linear_{n}_{k}"
        if name in ops:
            return name
        ops[name] = linear(n, k, 16, tuning, dsl, module)
        return name

    from verify_attention import attach

    attention_plan, flashinfer_attention = attach(m, kernels)
    plain = copy.deepcopy(m["programs"]["prefill"]["calls"])
    # context() was appended above; rebuild only the target forward here.
    plain = plain[: next(j for j, c in enumerate(plain) if c["label"] == "context.fc")]
    verify = []
    commit = []
    gdn_layer = 0
    attn_layer = 0
    for c in plain:
        label = c["label"]
        if label in ("plan", "prefill_plan", "last", "final_norm", "lm_head", "sample"):
            continue
        if label == "embed":
            c["args"][0] = b("d.verify_ids")
        if c["op"].startswith("gemm_") and c["op"] != "gemm_bf16":
            c["op"].split("_")[-1]
            a = c["args"]
            c["op"] = verify_linear(a[5]["i32"], a[6]["i32"])

        if ".conv" in label or ".state_" in label or ".gdn." in label:
            if label.endswith(".conv"):
                layer = int(label.split(".")[0][1:])
                idx = gdn_layer
                gdn_layer += 1
                lineptr = b("gdn.lines", idx * maximum * 4)
                raw = b("replay.raw_qkv", idx * rows * 16384 * 2)
                prep = next(x for x in plain if x["label"] == f"l{layer}.gdn.prepare")["args"]
                dt, alog = prep[3], prep[4]
                caches = {
                    n: b("replay." + n, idx * maximum * w * size)
                    for n, w, size in [
                        ("D", 48 * 16 * 128, 2),
                        ("K", 16 * 16 * 128, 2),
                        ("G", 48 * 16, 4),
                        ("Dr", 48 * 16 * 128, 2),
                        ("Kr", 16 * 16 * 128, 2),
                    ]
                }
                verify += [
                    copy_rows(label + ".save", b("qkvz"), raw, 16384),
                    call("spec_conv", raw, c["args"][1], st("gdn"), lineptr, b("conv"), l(line), label=label),
                    dc(
                        "replay_verify",
                        [4, "seqs", 48],
                        {
                            "q": b("conv"),
                            "k": b("conv", 2048 * 2),
                            "v": b("conv", 4096 * 2),
                            "a": b("ba", 48 * 2),
                            "b": b("ba"),
                            "A_log": alog,
                            "dt_bias": dt,
                            "o": b("gdn.raw"),
                            "h0": st("gdn", 3 * 10240 * 2),
                            "d_cache": caches["D"],
                            "k_cache": caches["K"],
                            "g_cache": caches["G"],
                            "rawv_cache": caches["Dr"],
                            "rawk_cache": caches["Kr"],
                            "beta_cache": caches["G"],
                            "query_start_loc": b("q_indptr"),
                            "ssm_state_indices": lineptr,
                            "replay_indices": b("d.indices"),
                            "write_pos": b("d.zero"),
                            "cache_base": b("d.zero"),
                            "is_flush_flags": b("d.zero"),
                        },
                        f"l{layer}.replay_verify",
                    ),
                ]
                gate = copy.deepcopy(next(x for x in plain if x["label"] == f"l{layer}.gdn.gate"))
                verify.append(gate)
                commit += [
                    dc(
                        "replay_commit",
                        [4, "seqs", 48],
                        {
                            "h0": st("gdn", 3 * 10240 * 2),
                            "d_cache": caches["D"],
                            "k_cache": caches["K"],
                            "g_cache": caches["G"],
                            "d_residual_cache": caches["Dr"],
                            "k_residual_cache": caches["Kr"],
                            "ssm_state_indices": lineptr,
                            "replay_indices": b("d.indices"),
                            "write_pos": b("accepted"),
                            "cache_base": b("d.zero"),
                            "is_flush_flags": b("d.one"),
                            "accept_lens": b("accepted"),
                            "mamba_track_indices": lineptr,
                            "mamba_steps_to_track": b("accepted"),
                        },
                        f"l{layer}.replay_commit",
                    ),
                    call(
                        "spec_conv_commit",
                        raw,
                        st("gdn"),
                        lineptr,
                        b("accepted"),
                        l(line),
                        label=f"l{layer}.conv_commit",
                    ),
                ]
            continue
        if label.endswith(".attention"):
            layer = int(label.split(".")[0][1:])
            idx = attn_layer
            attn_layer += 1
            verify.append(
                flashinfer_attention(
                    b("q"), b("attn.out"), "kv", idx, scales[str(layer)]["k"], scales[str(layer)]["v"], label
                )
            )
            continue
        verify.append(c)
    norm_template = copy.deepcopy(next(c for c in plain if c["label"] == "l0.input_norm"))
    norm_template["label"] = "verify.final_norm"
    norm_template["args"][0] = b("x")
    norm_template["args"][1] = b("h")
    norm_template["args"][2] = next(c["args"][2] for c in plain if c["label"] == "final_norm")
    verify += [
        norm_template,
        gemm("verify.lm_head", b("x"), lm, b("d.target_logits"), V, H),
        call("spec_sampling", *[b(n) for n in sampling_names], *[b("d.sample." + n) for n in sampling_names]),
        call(
            "verify_sample",
            b("d.target_logits"),
            b("d.sample.lengths"),
            *[b("d.sample." + n) for n in sampling_names[:-1]],
            b("verify_tokens"),
            T,
        ),
    ]
    round_calls = (
        attention_plan
        + draft_calls
        + verify
        + context(False)
        + [call("spec_count", b("d.draft"), b("verify_tokens"), b("accepted"))]
        + commit
    )
    m["programs"]["round"] = {"batch": {"groups": maximum, "rows": 16}, "graph": True, "calls": round_calls}
    m["model"] = "qwen3.8-27b-fp8-dspine"
    return [str(draft.resolve()), str(tables.resolve())]
