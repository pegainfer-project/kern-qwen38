"""AOT specializations for the Qwen3.8 DSpine checkpoint."""


def specifications(load):
    bridge, io, replay = (load(n) for n in ("dspine", "dspine_io", "replayssm"))
    result = []

    def add(name, fn, types, constants, warps=4):
        missing = set(fn.arg_names) - types.keys() - constants.keys()
        if missing:
            raise ValueError(f"{name}: untyped arguments {missing}")
        signature = ",".join(types.get(arg, "0") for arg in fn.arg_names)
        result.append((name, fn, signature, warps, constants))

    bf = lambda *names: {n: "*bf16" for n in names}
    add("dspine_norm", io.rms_norm, bf("X", "W", "Y"), {"H": 5120, "EPS": 1e-6, "B": 8192})
    add(
        "dspine_read",
        bridge.prepare_read,
        bf("X", "Residual", "Raw", "Normalized") | {"RMS": "*fp32"},
        {"H": 5120, "EPS": 1e-6, "ADD": False, "WIDTH": 8192},
    )
    add(
        "dspine_codes",
        bridge.normalize_codes,
        bf("Projected", "Codebook", "Codes") | {"Anchors": "*i64"},
        {"C": 1024, "T": 16, "ANCHOR_ROW": True, "EPS": 1e-6, "WIDTH": 1024},
    )
    add(
        "dspine_message",
        bridge.feature_message,
        bf("Packed", "Bias", "Message"),
        {"R": 512, "T": 16, "WIDTH": 512},
    )
    add(
        "dspine_finish",
        bridge.feature_finish,
        bf("Raw", "Delta", "Weight", "Output", "Normed") | {"RMS": "*fp32", "Valid": "*i32"},
        {"H": 5120, "T": 16, "WRITE": True, "NORM": True, "NORM_EPS": 1e-6, "WIDTH": 8192},
    )
    add(
        "dspine_topk_max",
        bridge._chunk_max,
        {"X": "*bf16", "CM": "*fp32"},
        {"stride_x": 248320, "V": 248320, "NCH": 3880, "BLOCK": 4096, "CH": 64},
    )
    add(
        "dspine_topk",
        bridge._select_topk,
        {"X": "*bf16", "CM": "*fp32", "OUT": "*i64"},
        {"stride_x": 248320, "V": 248320, "NCH": 3880, "K": 16, "CH": 64, "NCHPAD": 4096},
        8,
    )
    add(
        "dspine_table",
        bridge._serial_table,
        bf("Raw", "NormW", "LmW", "TG", "TSV", "Query", "GateB")
        | {"Cand": "*i64", "InputIds": "*i64", "Rms": "*fp32", "Out": "*i64", "Confidence": "*i64"},
        {
            "stride_raw": 5120,
            "stride_w": 5120,
            "stride_q": 1536,
            "H": 5120,
            "RK": 512,
            "K": 16,
            "KP": 16,
            "TM1": 15,
            "BH": 512,
            "BR": 128,
            "WITH_CONFIDENCE": False,
        },
        8,
    )
    add(
        "dspine_chain",
        bridge._serial_chain,
        {n: "*i64" for n in ("Idx", "Cand", "Out", "Confidence", "ConfidenceOut")},
        {"K": 16, "TM1": 15, "RPAD": 16, "WITH_CONFIDENCE": False},
        1,
    )
    add(
        "dspine_ids",
        io.block_ids,
        {n: "*i64" for n in ("Input", "Anchor", "Ids")},
        {"T": 16, "MASK": 248070},
        1,
    )
    add("dspine_proposals", io.proposals, {n: "*i64" for n in ("Anchor", "Draft", "Ids")}, {"T": 16}, 1)
    add("dspine_skip", io.skip_anchors, bf("X", "Y"), {"H": 5120, "T": 16, "B": 256})
    for context in (False, True):
        add(
            "dspine_rope_ctx" if context else "dspine_rope",
            io.qkv_rope,
            bf("X", "QW", "KW", "Q", "K", "V") | {"Positions": "*i64"},
            {"CONTEXT": context, "EPS": 1e-6, "THETA": 10000000.0},
        )
    # Layer offsets are supplied by pointer arguments, so these kernels see layer zero.
    for prefill in (False, True):
        add(
            "dspine_store_pre" if prefill else "dspine_store",
            io.context_store,
            bf("K", "V", "Cache") | {"Positions": "*i64", "Lines": "*i32", "TOKENS": "i32"},
            {"PREFILL": prefill, "LAYER": 0, "RING": 8208, "T": 16},
        )
    for draft in (False, True):
        add(
            "dspine_attention" if draft else "verify_attention",
            io.block_attention,
            bf("Q", "K", "V", "Output")
            | {
                "Cache": "*bf16" if draft else "*fp8e4nv",
                "Lines": "*i32",
                "Pages": "*i32",
                "Lengths": "*i32",
                "TABLE": "i32",
                "KS": "fp32",
                "VS": "fp32",
            },
            {
                "D": 128 if draft else 256,
                "QH": 32 if draft else 24,
                "KH": 8 if draft else 4,
                "T": 16,
                "DRAFT": draft,
                "LAYER": 0,
                "RING": 8208,
                "BK": 64,
            },
            4,
        )
    # State and cache pointers are already offset to the current GDN layer.
    stride = 3 * 10240 + 48 * 128 * 128
    dslot, kslot, gslot = 48 * 16 * 128, 16 * 16 * 128, 48 * 16
    const = {
        "stride_q_t": 10240,
        "stride_k_t": 10240,
        "stride_v_t": 10240,
        "stride_a_t": 96,
        "stride_b_t": 96,
        "stride_o_t": 6144,
        "stride_state_slot": stride,
        "stride_d_slot": dslot,
        "stride_k_slot": kslot,
        "stride_g_slot": gslot,
        "stride_rawv_slot": dslot,
        "stride_rawk_slot": kslot,
        "stride_beta_slot": gslot,
        "stride_qsl": 1,
        "stride_indices": 1,
        "stride_replay_indices": 1,
        "H": 16,
        "HV": 48,
        "K": 128,
        "V": 128,
        "BK": 128,
        "BV": 32,
        "BS": 16,
        "BC": 16,
        # Smaller K slices keep the FP32/TF32x3 intermediates in registers on SM120.
        "NK": 8,
        "BKT": 16,
        "MAX_CACHE_LEN": 16,
        "SOFTPLUS_THRESHOLD": 20.0,
        "USE_QK_L2NORM_IN_KERNEL": True,
        "STATE_FP32": True,
        "STORE_EXACT_RAW": False,
        "STORE_D_RESIDUAL": True,
        "STORE_K_RESIDUAL": True,
        "IS_FLUSH": False,
        "NULL_BLOCK_ID": 0,
        "DOT_PRECISION": "tf32x3",
        "scale": 128**-0.5,
    }
    types = (
        bf("q", "k", "v", "a", "b", "dt_bias", "o", "h0", "d_cache", "k_cache", "rawv_cache", "rawk_cache")
        | {n: "*fp32" for n in ("A_log", "g_cache", "beta_cache")}
        | {
            n: "*i32"
            for n in ("query_start_loc", "ssm_state_indices", "write_pos", "cache_base", "is_flush_flags")
        }
        | {"replay_indices": "*i64"}
    )
    add("replay_verify", replay.empty_history_normal_kernel, types, const, 4)
    const = {
        "stride_state_slot": stride,
        "stride_d_slot": dslot,
        "stride_k_slot": kslot,
        "stride_g_slot": gslot,
        "stride_d_residual_slot": dslot,
        "stride_k_residual_slot": kslot,
        "stride_state_layer": 0,
        "stride_d_layer": 0,
        "stride_k_layer": 0,
        "stride_g_layer": 0,
        "stride_d_residual_layer": 0,
        "stride_k_residual_layer": 0,
        "stride_indices": 1,
        "stride_replay_indices": 1,
        "stride_accept": 1,
        "stride_track": 0,
        "stride_steps": 0,
        "H": 16,
        "HV": 48,
        "K": 128,
        "V": 128,
        "BV": 32,
        "MAX_CACHE_LEN": 16,
        "NULL_BLOCK_ID": 0,
        "HAS_TRACK": False,
        "HAS_RESIDUAL": True,
    }
    types = (
        bf("h0", "d_cache", "k_cache", "d_residual_cache", "k_residual_cache")
        | {"g_cache": "*fp32", "replay_indices": "*i64"}
        | {
            n: "*i32"
            for n in (
                "ssm_state_indices",
                "write_pos",
                "cache_base",
                "is_flush_flags",
                "accept_lens",
                "mamba_track_indices",
                "mamba_steps_to_track",
            )
        }
    )
    add("replay_commit", replay.gdn_replayssm_compact_commit_kernel, types, const, 4)
    return result
