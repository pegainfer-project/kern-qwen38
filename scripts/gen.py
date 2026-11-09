#!/usr/bin/env python3
"""Generate a batched Qwen3.8 FP8 bundle, with chunked GDN prefill and recurrent decode."""

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H, F, VOCAB, PAGE = 5120, 17408, 248320, 64
CONV = 3 * 10240 * 2
LINE = CONV + 48 * 128 * 128 * 2
b = lambda name, offset=0: {"buf": name, **({"offset": offset} if offset else {})}
s = lambda name, offset=0: {"state": name, **({"offset": offset} if offset else {})}
i = lambda value: {"i32": value}
l = lambda value: {"i64": value}
f = lambda value: {"f32": value}
T = {"var": "tokens"}
ceil = lambda value, divisor: {"ceil_div": [value, divisor]}


def generate(checkpoint, kernels, out, context, chunk, kv_dtype, scale_file, max_seqs=16):
    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    context = context or text["max_position_embeddings"]
    if context > text["max_position_embeddings"] or context % PAGE or not max_seqs <= chunk <= context:
        raise ValueError(
            "require batch <= chunk <= context <= checkpoint limit, with context divisible by 64"
        )
    if not 1 <= max_seqs <= 16:
        raise ValueError("max-seqs must be within 1..16")
    splits = 32
    plan_slots = max_seqs * splits
    S = {"var": "seqs"}

    if (text["hidden_size"], text["num_hidden_layers"], text["intermediate_size"]) != (H, 64, F):
        raise ValueError("expected Qwen3.8-27B geometry")
    quant = config["quantization_config"]
    if quant["weight_block_size"] != [128, 128] or quant["fmt"] != "e4m3":
        raise ValueError("expected block128 E4M3 checkpoint")
    build = json.loads((kernels / "build.json").read_text())
    dsl = json.loads((kernels / "dsl.json").read_text())["modules"]
    if build["architecture"] != "sm_120":
        raise ValueError("SM120 build required")
    scales = json.loads(scale_file.read_text()) if scale_file else None
    if kv_dtype == "fp8" and scales is None:
        raise ValueError("FP8 KV requires --kv-scales with calibrated per-layer k/v factors")
    kv_bytes = 1 if kv_dtype == "fp8" else 2
    layer_elements = 2 * 4 * PAGE * 256
    page_elements = 16 * layer_elements
    buffers, ops, modules, load = {}, {}, {}, []
    dynamic = {}
    calls = {"prefill": [], "decode": []}

    def buffer(name, dtype, shape, kind="workspace", **extra):
        buffers[name] = dict(dtype=dtype, shape=shape, kind=kind, **extra)
        return b(name)

    def weight(name, shape, tensors=None, dtype="bf16"):
        if name not in buffers:
            buffer(name, dtype, shape, "weight", bind=[{"tensor": t} for t in (tensors or [name])])
        return b(name)

    def module(name):
        if name not in modules:
            path = kernels / (name + ".cubin")
            modules[name] = {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return name

    def op(name, mod, entry, params, grid, block=256, shared=0, extra=()):
        if name in ops:
            return
        original = list(params)
        launch_args = [{"param": j} for j in range(len(original))]
        types = list(original)
        interface = list(original)
        dynamic[name] = []
        for value in extra:
            if "var" in value or "expr" in value:
                launch_args.append({"param": len(interface)})
                interface.append("i32")
                types.append("i32")
                dynamic[name].append(value)
            else:
                launch_args.append(value)
                types.append(next(iter(value)))
        ops[name] = {
            "params": interface,
            "impl": {
                "launches": [
                    {
                        "module": module(mod),
                        "entry": entry,
                        "params": types,
                        "grid": grid,
                        "block": [block, 1, 1] if isinstance(block, int) else block,
                        "shared_mem": shared,
                        "args": launch_args,
                    }
                ]
            },
        }

    def call(mode, label, name, *args):
        calls[mode].append({"label": label, "op": name, "args": list(args) + dynamic.get(name, [])})

    def once_norm(tensor, width, plus=True):
        name = tensor + (".plus_one" if plus else ".f32")
        if name not in buffers:
            weight(tensor, [width])
            buffer(name, "f32", [width], "carry")
            opname = "weight_p1" if plus else "weight_cast"
            op(
                opname,
                "weight_prep",
                "kern_weight_p1_bf16_f32" if plus else "kern_cast_bf16_f32",
                ["in buffer<bf16>", "out buffer<f32>", "i32"],
                [ceil(width, 256), 1, 1],
            )
            load.append({"label": name, "op": opname, "args": [b(tensor), b(name), i(width)]})
        return b(name)

    def fill_scale(name, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("KV scales must be finite and positive")
        buffer(name, "f32", [1], "carry")
        op("fill_scale", "weight_prep", "kern_fill_f32", ["out buffer<f32>", "i32", "f32"], [1, 1, 1])
        load.append({"label": name, "op": "fill_scale", "args": [b(name), i(1), f(value)]})
        return b(name)

    buffer("token_ids", "i64", ["tokens"], "input", fill="token", domain={"min": 0, "max": VOCAB - 1})
    buffer("positions", "i64", ["tokens"], "input", fill="position", domain={"min": 0, "max": context - 1})
    buffer("slots", "i64", ["tokens"], "input", fill="slot", domain={"index_into": "kv"})
    buffer("pages", "i32", ["seqs", context // PAGE], "input", domain={"index_into": "kv", "stride": PAGE})
    buffer("lengths", "i32", ["seqs"], "input", fill="seq_len", domain={"min": 1, "max": context})
    buffer(
        "q_indptr",
        "i32",
        [max_seqs + 1],
        "input",
        fill="cu_seqlens",
        domain={"min": 0, "max": "tokens", "monotone": True},
    )
    buffer("gdn.lines", "i32", [48, "seqs"], "input", domain={"index_into": "gdn", "stride": LINE})
    buffer("next_token", "i64", ["seqs"], "output", fill="tokens", domain={"min": 0, "max": VOCAB - 1})
    for name, width in [
        ("h", H),
        ("x", H),
        ("branch", H),
        ("qkvz", 16384),
        ("ba", 96),
        ("conv", 10240),
        ("gdn.out", 6144),
        ("gdn.raw", 6144),
        ("qkv", 14336),
        ("q", 6144),
        ("k", 1024),
        ("attn.out", 6144),
        ("attn.gated", 6144),
        ("gate_up", 2 * F),
        ("act", F),
    ]:
        buffer(name, "bf16", ["tokens", width])
    buffer("last", "bf16", ["seqs", H])
    buffer("logits", "bf16", ["seqs", VOCAB])
    buffer("cos", "bf16", ["tokens", 32])
    buffer("sin", "bf16", ["tokens", 32])
    for name, shape in [
        ("indices", [max_seqs * context // PAGE]),
        ("indptr", [max_seqs + 1]),
        ("last", [max_seqs]),
        ("request", [max(plan_slots, (chunk * 6 + 63) // 64)]),
        ("tile", [max(plan_slots, (chunk * 6 + 63) // 64)]),
        ("zero", [max(plan_slots, (chunk * 6 + 63) // 64)]),
        ("kv_chunk", [1]),
    ]:
        buffer("plan." + name, "i32", shape)
    buffer("plan.merge", "i32", [max_seqs + 1])
    buffer("plan.valid", "u8", [plan_slots])
    buffer("attn.partial", "bf16", [plan_slots, 6144])
    buffer("attn.lse", "f32", [plan_slots, 24])
    op(
        "batch_plan",
        "attention_plan",
        "attention_plan",
        ["in buffer<i32>"] * 2 + ["out buffer<i32>"] * 6 + ["out buffer<u8>", "out buffer<i32>"],
        [1, 1, 1],
        extra=[S, i(context // PAGE), i(0), i(splits), i(0), i(plan_slots), i(PAGE)],
    )
    # Prefill remains one request; decode plans cover every active sequence.
    op("plan", "model_glue", "decode_plan", ["in buffer<i32>"] * 2 + ["out buffer<i32>"] * 6, [1, 1, 1])
    op("prefill_plan", "model_glue", "prefill_plan", ["out buffer<i32>"] * 3 + ["i32"], [1, 1, 1])
    op(
        "rope",
        "model_glue",
        "rope_gather",
        ["out buffer<bf16>"] * 2 + ["in buffer<i64>"],
        ["tokens", 1, 1],
        block=32,
        extra=[f(1e7)],
    )
    op(
        "embed",
        "embedding",
        "embedding",
        ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>"],
        ["tokens", 1, 1],
        extra=[i(H), i(VOCAB)],
    )
    op(
        "norm",
        "gemma_rms_norm",
        "kern_gemma_rms_norm_bf16",
        ["out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>"],
        ["tokens", 1, 1],
        block=512,
        shared=2 * H * 4 + 512 * 4,
        extra=[i(H), T, i(1), i(H), i(H), i(H), i(H), f(text["rms_norm_eps"])],
    )
    op(
        "last_norm",
        "gemma_rms_norm",
        "kern_gemma_rms_norm_bf16",
        ["out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>"],
        ["seqs", 1, 1],
        block=512,
        shared=2 * H * 4 + 512 * 4,
        extra=[i(H), S, i(1), i(H), i(H), i(H), i(H), f(text["rms_norm_eps"])],
    )
    op(
        "add",
        "model_glue",
        "residual_add",
        ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32"],
        [ceil({"mul": ["tokens", H]}, 256), 1, 1],
    )
    op(
        "gemv_fp8",
        "fp8_gemv",
        "kern_gemv_fp8_block128",
        ["in buffer<fp8e4m3>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        [1, 1, 1],
    )
    # Each projection gets its own row grid; GEMM uses a single shared AOT specialization.
    meta = dsl["fp8_block_gemm"]
    op(
        "gemm_fp8",
        "fp8_block_gemm",
        meta["entry"],
        ["in buffer<bf16>", "in buffer<fp8e4m3>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
        [ceil("tokens", 32), 1, 1],
        block=meta["block"],
        shared=meta["shared_mem"],
    )
    ops["gemm_bf16"] = {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
        "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]},
    }
    op(
        "silu",
        "silu_mul",
        "kern_silu_mul_bf16",
        ["out buffer<bf16>", "in buffer<bf16>"],
        ["tokens", ceil(F, 2048), 1],
        extra=[i(F)],
    )
    op(
        "last",
        "copy_rows",
        "kern_last_row_bf16",
        ["out buffer<bf16>", "in buffer<bf16>"],
        [1, 1, 1],
        extra=[i(H), i(H), T],
    )
    ops["sample"] = {
        "params": ["in buffer<bf16>", "out buffer<i64>"],
        "impl": {
            "scratch": {
                "max": {"dtype": "f32", "shape": ["seqs", 64]},
                "idx": {"dtype": "i32", "shape": ["seqs", 64]},
            },
            "launches": [
                {
                    "module": module("argmax"),
                    "entry": "kern_argmax_partial_bf16",
                    "params": ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                    "grid": ["seqs", 64, 1],
                    "block": [1024, 1, 1],
                    "args": [{"param": 0}, {"scratch": "max"}, {"scratch": "idx"}, i(VOCAB)],
                },
                {
                    "module": module("argmax"),
                    "entry": "kern_argmax_final_i64",
                    "params": ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                    "grid": ["seqs", 1, 1],
                    "block": [64, 1, 1],
                    "args": [{"scratch": "max"}, {"scratch": "idx"}, {"param": 1}, i(64)],
                },
            ],
        },
    }

    def projection(mode, label, src, dst, n, k, tensors):
        fp8 = n not in (96, VOCAB)
        wname = label + ".weight"
        w = weight(wname, [n, k], [t + ".weight" for t in tensors], dtype="fp8e4m3" if fp8 else "bf16")
        if fp8:
            scale = weight(label + ".scale", [n // 128, k // 128], [t + ".weight_scale_inv" for t in tensors])
            opname = f"linear_{n}" if mode == "decode" else f"gemm_{n}"
            if opname not in ops:
                if mode == "decode":
                    small = copy.deepcopy(ops["gemv_fp8"]["impl"]["launches"][0])
                    small.update(grid=[ceil(n, 8), "seqs", 1], when={"var": "seqs", "max": 8})
                    large = copy.deepcopy(ops["gemm_fp8"]["impl"]["launches"][0])
                    weight_meta = dsl["fp8_weight_gemm"]
                    large.update(
                        module=module("fp8_weight_gemm"),
                        entry=weight_meta["entry"],
                        shared_mem=weight_meta["shared_mem"],
                        block=weight_meta["block"],
                    )
                    large.update(
                        grid=[ceil("tokens", 128), ceil(n, 64), 1],
                        when={"var": "seqs", "min": 9},
                        args=[
                            {"param": 2},
                            {"param": 0},
                            {"param": 1},
                            {"param": 3},
                            {"param": 6},
                            {"param": 4},
                            {"param": 5},
                        ],
                    )
                    ops[opname] = {
                        "params": ops["gemv_fp8"]["params"] + ["i32"],
                        "impl": {"launches": [small, large]},
                    }
                else:
                    ops[opname] = copy.deepcopy(ops["gemm_fp8"])
                    ops[opname]["impl"]["launches"][0]["grid"] = [ceil("tokens", 32), ceil(n, 64), 1]
            if mode == "decode":
                call(mode, label, opname, w, scale, b(src), b(dst), i(n), i(k), T)
            else:
                call(mode, label, opname, b(src), w, scale, b(dst), T, i(n), i(k))

        else:
            call(mode, label, "gemm_bf16", b(src), w, b(dst), S if n == VOCAB else T, i(n), i(k))

    for name, dtype, shape in [
        ("gq", "bf16", ["tokens", 48, 128]),
        ("gk", "bf16", ["tokens", 48, 128]),
        ("gv", "bf16", ["tokens", 48, 128]),
        ("g", "f32", ["tokens", 48]),
        ("beta", "f32", ["tokens", 48]),
        ("gc", "f32", ["tokens", 48]),
        ("tril", "f32", ["tokens", 48, 64]),
        ("inv", "bf16", ["tokens", 48, 64]),
        ("w", "bf16", ["tokens", 48, 128]),
        ("u", "bf16", ["tokens", 48, 128]),
        ("vnew", "bf16", ["tokens", 48, 128]),
        ("initial", "f32", [48, 128, 128]),
        ("final", "f32", [48, 128, 128]),
        ("chunks", "f32", [(chunk + 63) // 64, 48, 128, 128]),
    ]:
        buffer("fla." + name, dtype, shape)
    op(
        "conv_prefill",
        "model_glue",
        "conv_prefill",
        ["in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "out buffer<bf16>"],
        ["tokens", 40, 1],
        extra=[T, l(LINE)],
    )
    op(
        "conv_commit",
        "model_glue",
        "conv_commit",
        ["in buffer<bf16>", "inout state", "in buffer<i32>"],
        [40, 1, 1],
        extra=[T, l(LINE)],
    )
    for action in ("gather", "scatter"):
        params = (
            ["in state", "in buffer<i32>", "out buffer<f32>"]
            if action == "gather"
            else ["inout state", "in buffer<i32>", "in buffer<f32>"]
        )
        op("state_" + action, "model_glue", "state_" + action, params, [3072, 1, 1], extra=[l(LINE)])
    op(
        "gdn_gate",
        "model_glue",
        "gdn_norm_gate",
        ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>"],
        ["tokens", 48, 1],
        block=32,
        extra=[f(text["rms_norm_eps"])],
    )
    op(
        "conv_decode",
        "gdn_decode_bf16",
        "kern_gdn_conv_bf16",
        ["inout buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>"],
        ["seqs", 40, 1],
        extra=[i(10240), i(16384), l(LINE)],
    )
    op(
        "gdn_decode",
        "gdn_decode_bf16",
        "kern_gdn_step_state_bf16",
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "inout state",
            "in buffer<i32>",
        ],
        [48, "seqs", 1],
        extra=[f(128**-0.5), f(text["rms_norm_eps"]), i(16384), i(96), l(LINE)],
    )

    def dsl_op(name, params, grid, extra):
        meta = dsl[name]
        op(
            name,
            name,
            meta["entry"],
            params,
            grid,
            block=meta["block"],
            shared=meta["shared_mem"],
            extra=extra,
        )

    dsl_op(
        "gdn_prepare",
        ["in buffer<bf16>"] * 4 + ["in buffer<f32>"] + ["out buffer<bf16>"] * 3 + ["out buffer<f32>"] * 2,
        ["tokens", 48, 1],
        [i(16), i(48), i(10240), T],
    )
    dsl_op("gdn_cumsum", ["in buffer<f32>", "out buffer<f32>"], [ceil("tokens", 64), 48, 1], [T, i(48)])
    dsl_op(
        "gdn_kkt",
        ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>"],
        [ceil("tokens", 64), 48, 1],
        [T, i(48)],
    )
    dsl_op("gdn_solve", ["in buffer<f32>", "out buffer<bf16>"], [ceil("tokens", 64), 48, 1], [T, i(48)])
    dsl_op(
        "gdn_recompute",
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "out buffer<bf16>",
            "out buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
        ],
        [ceil("tokens", 64), 48, 1],
        [T, i(48)],
    )
    dsl_op(
        "gdn_state",
        ["in buffer<bf16>"] * 3
        + ["in buffer<f32>"] * 2
        + ["out buffer<f32>", "out buffer<bf16>", "out buffer<f32>"],
        [4, 48, 1],
        [T, i(48)],
    )
    dsl_op(
        "gdn_output",
        ["in buffer<bf16>"] * 3 + ["in buffer<f32>"] * 2 + ["out buffer<bf16>"],
        [4, ceil("tokens", 64), 48],
        [T, i(48), f(128**-0.5)],
    )

    embed = weight("embed", [VOCAB, H], ["model.language_model.embed_tokens.weight"])
    attn_index = 0
    gdn_index = 0
    for mode in calls:
        if mode == "decode":
            call(
                mode,
                "plan",
                "batch_plan",
                b("pages"),
                b("lengths"),
                *[
                    b("plan." + n)
                    for n in ("indices", "indptr", "last", "request", "tile", "merge", "valid", "kv_chunk")
                ],
            )
        else:
            call(
                mode,
                "plan",
                "plan",
                b("pages"),
                b("lengths"),
                *[b("plan." + n) for n in ("indices", "indptr", "last", "request", "zero", "kv_chunk")],
            )
        if mode == "prefill":
            call(mode, "prefill_plan", "prefill_plan", b("plan.request"), b("plan.tile"), b("plan.zero"), T)
        call(mode, "rope", "rope", b("cos"), b("sin"), b("positions"))
        call(mode, "embed", "embed", b("token_ids"), embed, b("h"))
        attn_index = gdn_index = 0
        for layer, kind in enumerate(text["layer_types"]):
            label = f"l{layer}"
            p = f"model.language_model.layers.{layer}"
            call(
                mode,
                label + ".input_norm",
                "norm",
                b("x"),
                b("h"),
                once_norm(p + ".input_layernorm.weight", H),
            )
            if kind == "linear_attention":
                q = p + ".linear_attn"
                projection(
                    mode, label + ".qkvz", "x", "qkvz", 16384, H, [q + ".in_proj_qkv", q + ".in_proj_z"]
                )
                projection(mode, label + ".ba", "x", "ba", 96, H, [q + ".in_proj_b", q + ".in_proj_a"])
                conv = weight(q + ".conv1d.weight", [10240, 1, 4])
                norm = weight(q + ".norm.weight", [128])
                dt = weight(q + ".dt_bias", [48])
                alog = once_norm(q + ".A_log", 48, False)
                line = b("gdn.lines", gdn_index * max_seqs * 4)
                if mode == "decode":
                    call(mode, label + ".conv", "conv_decode", b("qkvz"), conv, s("gdn"), line)
                    call(
                        mode,
                        label + ".gdn",
                        "gdn_decode",
                        b("qkvz"),
                        b("ba"),
                        alog,
                        dt,
                        norm,
                        b("gdn.out"),
                        s("gdn", CONV),
                        line,
                    )
                else:
                    call(mode, label + ".conv", "conv_prefill", b("qkvz"), conv, s("gdn"), line, b("conv"))
                    call(mode, label + ".conv_commit", "conv_commit", b("qkvz"), s("gdn"), line)
                    call(mode, label + ".state_read", "state_gather", s("gdn", CONV), line, b("fla.initial"))
                    call(
                        mode,
                        label + ".gdn.prepare",
                        "gdn_prepare",
                        b("conv"),
                        b("ba"),
                        b("ba", 48 * 2),
                        dt,
                        alog,
                        *[b("fla." + n) for n in ("gq", "gk", "gv", "g", "beta")],
                    )
                    call(mode, label + ".gdn.cumsum", "gdn_cumsum", b("fla.g"), b("fla.gc"))
                    call(
                        mode,
                        label + ".gdn.kkt",
                        "gdn_kkt",
                        b("fla.gk"),
                        b("fla.gc"),
                        b("fla.beta"),
                        b("fla.tril"),
                    )
                    call(mode, label + ".gdn.solve", "gdn_solve", b("fla.tril"), b("fla.inv"))
                    call(
                        mode,
                        label + ".gdn.recompute",
                        "gdn_recompute",
                        *[b("fla." + n) for n in ("gk", "gv", "beta", "w", "u", "inv", "gc")],
                    )
                    call(
                        mode,
                        label + ".gdn.state",
                        "gdn_state",
                        *[
                            b("fla." + n)
                            for n in ("gk", "w", "u", "gc", "initial", "chunks", "vnew", "final")
                        ],
                    )
                    call(
                        mode,
                        label + ".gdn.output",
                        "gdn_output",
                        *[b("fla." + n) for n in ("gq", "gk", "vnew", "chunks", "gc")],
                        b("gdn.raw"),
                    )
                    call(mode, label + ".state_write", "state_scatter", s("gdn", CONV), line, b("fla.final"))
                    call(mode, label + ".gdn.gate", "gdn_gate", b("gdn.out"), b("gdn.raw"), b("qkvz"), norm)
                projection(mode, label + ".out_proj", "gdn.out", "branch", H, 6144, [q + ".out_proj"])
                gdn_index += 1
            else:
                q = p + ".self_attn"
                aidx = attn_index
                attn_index += 1
                projection(
                    mode, label + ".qkv", "x", "qkv", 14336, H, [q + ".q_proj", q + ".k_proj", q + ".v_proj"]
                )
                qw = once_norm(q + ".q_norm.weight", 256)
                kw = once_norm(q + ".k_norm.weight", 256)
                kval, vval = (scales[str(layer)]["k"], scales[str(layer)]["v"]) if scales else (1.0, 1.0)
                if mode == "prefill":
                    ks, vs = fill_scale(label + ".ks", kval), fill_scale(label + ".vs", vval)
                else:
                    ks, vs = b(label + ".ks"), b(label + ".vs")
                mod = "attn_prep_" + kv_dtype
                name = mod
                params = (
                    ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>"]
                    + ["in buffer<bf16>"] * 2
                    + ["out buffer<bf16>"] * 2
                    + ["inout state"] * 2
                    + ["in buffer<i64>"]
                )
                if kv_dtype == "fp8":
                    params += ["in buffer<f32>"] * 2
                extra = [T, f(text["rms_norm_eps"]), i(14336), l(page_elements), l(256), l(PAGE * 256)]
                op(name, mod, "kern_attn_prep_" + kv_dtype, params, ["tokens", 1, 1], extra=extra)
                launch = ops[name]["impl"]["launches"][0]
                # Scales are trailing CUDA arguments, after scalar geometry.
                if kv_dtype == "fp8":
                    launch["params"] = (
                        params[:10]
                        + [("i32" if "var" in x or "expr" in x else next(iter(x))) for x in extra]
                        + params[10:]
                    )
                    launch["args"] = (
                        [{"param": j} for j in range(10)]
                        + [{"param": 12}]
                        + extra[1:]
                        + [{"param": 10}, {"param": 11}]
                    )
                base = aidx * layer_elements * kv_bytes
                vbase = base + 4 * PAGE * 256 * kv_bytes
                args = [
                    b("qkv"),
                    qw,
                    kw,
                    b("cos"),
                    b("sin"),
                    b("q"),
                    b("k"),
                    s("kv", base),
                    s("kv", vbase),
                    b("slots"),
                ]
                if kv_dtype == "fp8":
                    args += [ks, vs]
                call(mode, label + ".attn_prep", name, *args)
                name = f"{label}.attention_{mode}"
                facts = build["attention"][kv_dtype]
                abi = facts["abi"][("prefill" if mode == "prefill" else "decode") + "_g6"]
                fields = {
                    "q": {"param": 0},
                    "o": {"param": 1},
                    "paged_kv.k_data": {"param": 2},
                    "paged_kv.v_data": {"param": 3},
                    **{
                        n: {"param": j + 4}
                        for j, n in enumerate(
                            (
                                "paged_kv.indices",
                                "paged_kv.indptr",
                                "paged_kv.last_page_len",
                                "request_indices",
                                "kv_tile_indices",
                                "kv_chunk_size_ptr",
                            )
                        )
                    },
                    "num_qo_heads": i(24),
                    "q_stride_n": i(6144),
                    "q_stride_h": i(256),
                    "paged_kv.num_heads": i(4),
                    "paged_kv.batch_size": S,
                    "paged_kv.stride_page": i(page_elements),
                    "paged_kv.stride_n": i(256),
                    "paged_kv.stride_h": i(PAGE * 256),
                    "paged_kv.v_stride_page": i(page_elements),
                    "paged_kv.v_stride_n": i(256),
                    "paged_kv.v_stride_h": i(PAGE * 256),
                    "sm_scale": f(kval / 16),
                    "v_scale": f(vval),
                    "partition_kv": {"u8": 0},
                    "padded_batch_size": i(1),
                }
                params = ["in buffer<bf16>", "out buffer<bf16>", "in state", "in state"] + [
                    "in buffer<i32>"
                ] * 6
                if mode == "prefill":
                    params += ["in buffer<i32>"] * 2
                    fields.update(
                        q_indptr={"param": 10},
                        o_indptr={"param": 10},
                        qo_tile_indices={"param": 11},
                        padded_batch_size={"expr": ceil({"mul": ["tokens", 6]}, 64)},
                        max_total_num_rows=T,
                    )
                if mode == "decode":
                    fields.update(
                        o={"param": 12},
                        lse={"param": 13},
                        block_valid_mask={"param": 10},
                        o_indptr={"param": 11},
                        partition_kv={"u8": 1},
                        padded_batch_size={"expr": {"mul": ["seqs", splits]}},
                    )
                    params += [
                        "in buffer<u8>",
                        "in buffer<i32>",
                        "out buffer<bf16>",
                        "out buffer<f32>",
                        "i32",
                    ]
                packed = []
                covered = set()
                for field, value in fields.items():
                    at = abi["offsets"][field]
                    width = 8 if "param" in value else 1 if "u8" in value else 4
                    covered.update(range(at, at + width))
                    packed.append({"at": at, **value})
                packed = [
                    {"at": at, "u8": value}
                    for at, value in enumerate(abi["bytes"])
                    if value and at not in covered
                ] + packed
                grid = (
                    [ceil({"mul": ["tokens", 6]}, 64), 1, 4]
                    if mode == "prefill"
                    else [{"mul": ["seqs", splits]}, 4, 1]
                )
                block = facts["prefill_block"] if mode == "prefill" else [256 // (16 // kv_bytes), 6, 1]
                ops[name] = {
                    "params": params,
                    "impl": {
                        "launches": [
                            {
                                "module": module("attention_" + kv_dtype),
                                "entry": "prefill" if mode == "prefill" else "decode_g6",
                                "params": [f"bytes<{abi['size']}>"],
                                "args": [{"pack": {"size": abi["size"], "fields": packed}}],
                                "grid": grid,
                                "block": block,
                                "shared_mem": facts["prefill_shared"] if mode == "prefill" else 32768,
                            }
                        ]
                    },
                }
                if mode == "decode":
                    symbols = subprocess.check_output(
                        ["cuobjdump", "-symbols", str(kernels / ("attention_" + kv_dtype + ".cubin"))],
                        text=True,
                    )
                    merge = re.search(
                        r"\b(_ZN\S*PersistentVariableLengthMergeStatesKernel\S*)", symbols
                    ).group(1)
                    x = 256 // (16 // kv_bytes)
                    ops[name]["impl"]["launches"].append(
                        {
                            "module": module("attention_" + kv_dtype),
                            "entry": merge,
                            "grid": [{"mul": ["seqs", 24]}, 1, 1],
                            "block": [x, 128 // x, 1],
                            "shared_mem": 4 * (128 // x) * 256 * 2 + 128 * 4,
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
                                {"param": 12},
                                {"param": 13},
                                {"param": 11},
                                {"param": 1},
                                l(0),
                                {"param": 14},
                                l(0),
                                i(24),
                            ],
                        }
                    )
                call(
                    mode,
                    label + ".attention",
                    name,
                    b("q"),
                    b("attn.out"),
                    s("kv", base),
                    s("kv", vbase),
                    *[
                        b("plan." + n)
                        for n in (
                            "indices",
                            "indptr",
                            "last",
                            "request",
                            "zero" if mode == "prefill" else "tile",
                            "kv_chunk",
                        )
                    ],
                    *(
                        [b("q_indptr"), b("plan.tile")]
                        if mode == "prefill"
                        else [b("plan.valid"), b("plan.merge"), b("attn.partial"), b("attn.lse"), S]
                    ),
                )
                op(
                    "attn_gate",
                    "sigmoid_mul",
                    "kern_sigmoid_mul_bf16",
                    ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>"],
                    ["tokens", 3, 1],
                    extra=[i(24), i(256), i(14336), i(512)],
                )
                call(mode, label + ".gate", "attn_gate", b("attn.gated"), b("attn.out"), b("qkv", 256 * 2))
                projection(mode, label + ".out_proj", "attn.gated", "branch", H, 6144, [q + ".o_proj"])
            call(
                mode,
                label + ".attn_residual",
                "add",
                b("h"),
                b("h"),
                b("branch"),
                {"expr": {"mul": ["tokens", H]}},
            )
            call(
                mode,
                label + ".post_norm",
                "norm",
                b("x"),
                b("h"),
                once_norm(p + ".post_attention_layernorm.weight", H),
            )
            projection(
                mode, label + ".gate_up", "x", "gate_up", 2 * F, H, [p + ".mlp.gate_proj", p + ".mlp.up_proj"]
            )
            call(mode, label + ".silu", "silu", b("act"), b("gate_up"))
            projection(mode, label + ".down_proj", "act", "branch", H, F, [p + ".mlp.down_proj"])
            call(
                mode,
                label + ".residual",
                "add",
                b("h"),
                b("h"),
                b("branch"),
                {"expr": {"mul": ["tokens", H]}},
            )
        if mode == "prefill":
            call(mode, "last", "last", b("last"), b("h"))
        call(
            mode,
            "final_norm",
            "last_norm",
            b("last"),
            b("last") if mode == "prefill" else b("h"),
            once_norm("model.language_model.norm.weight", H),
        )
        projection(mode, "lm_head", "last", "logits", VOCAB, H, ["lm_head"])
        call(mode, "sample", "sample", b("logits"), b("next_token"))
    # Remove unselected generic templates so verification checks the actual operators only.
    for name in ("gemv_fp8", "gemm_fp8"):
        ops.pop(name)
    for operator in ops.values():
        for launch in operator["impl"]["launches"]:
            if launch.get("module") in dsl:
                launch["params"] += dsl[launch["module"]]["abi_tail"]
                launch["args"] += [l(0), l(0)]
    m = {
        "schema_version": 7,
        "model": "qwen3.8-27b-fp8",
        "chat": {"tool_parser": "qwen3_coder"},
        "vars": {"tokens": {"max": chunk, "axis": "rows"}, "seqs": {"max": max_seqs, "axis": "groups"}},
        "states": {"kv": {"bytes_per_token": 16 * 2 * 1024 * kv_bytes}, "gdn": {"bytes_per_seq": 48 * LINE}},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {
            "load": {"once": True, "calls": load},
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": calls["prefill"]},
            "decode": {"batch": {"groups": 1, "rows": 1}, "graph": True, "calls": calls["decode"]},
            "decode_batch": {
                "batch": {"groups": max_seqs, "rows": 1},
                "graph": True,
                "calls": calls["decode"],
            },
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "kernels"
    bundle.mkdir(exist_ok=True)
    for item in modules.values():
        shutil.copyfile(kernels / item["source"], bundle / item["source"])
    (out / "model.json").write_text(json.dumps(m, indent=2) + "\n")
    (out / "kern.toml").write_text(
        '[targets.qwen38]\nmanifest = "model.json"\nkernels = "kernels"\nweights = ['
        + json.dumps(str(checkpoint.resolve()))
        + "]\n"
    )
    print(out / "model.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/sm_120")
    parser.add_argument("--out", type=Path, default=ROOT / "out/qwen38")
    parser.add_argument("--context", type=int)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--kv-scales", type=Path)
    args = parser.parse_args()
    generate(
        args.checkpoint,
        args.kernels,
        args.out,
        args.context,
        args.chunk,
        args.kv_dtype,
        args.kv_scales,
        args.max_seqs,
    )
