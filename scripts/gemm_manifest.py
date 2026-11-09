"""One autotuned GEMM interface for prefill, ordinary decode and speculative verification."""

PARAMS = (
    "in buffer<bf16>",
    "in buffer<fp8e4m3>",
    "in buffer<bf16>",
    "out buffer<bf16>",
    "i32",
    "i32",
    "i32",
)
QUANT_PARAMS = (
    "in buffer<bf16>",
    "out buffer<fp8e4m3>",
    "out buffer<f32>",
    "i32",
    "i32",
)
W8A8_PARAMS = (
    "in buffer<fp8e4m3>",
    "in buffer<f32>",
    "in buffer<fp8e4m3>",
    "in buffer<bf16>",
    "out buffer<bf16>",
    "i32",
    "i32",
    "i32",
)


def linear(n, k, rows_per_seq, tuning, dsl, module):
    launches = []
    lower = 1
    for groups in (1, 2, 4, 8, 16):
        chosen = tuning[f"{groups * rows_per_seq},{n},{k}"]
        meta = dsl[chosen["module"]]
        launches.append(
            {
                "module": module(chosen["module"]),
                "entry": meta["entry"],
                "params": list(PARAMS) + meta["abi_tail"],
                "args": [{"param": j} for j in range(7)] + [{"i64": 0}, {"i64": 0}],
                "grid": [{"ceil_div": ["tokens", chosen["bm"]]}, {"ceil_div": [n, chosen["bn"]]}, 1],
                "block": meta["block"],
                "shared_mem": meta["shared_mem"],
                "when": {"var": "seqs", "min": lower, "max": groups},
            }
        )
        lower = groups + 1
    return {"params": list(PARAMS), "impl": {"launches": launches}}


def prefill_linear(n, k, tuning, dsl, module):
    q_meta = dsl["fp8_act_quant"]
    launches = [
        {
            "module": module("fp8_act_quant"),
            "entry": q_meta["entry"],
            "params": list(QUANT_PARAMS) + q_meta["abi_tail"],
            "args": [
                {"param": 0},
                {"scratch": "aq"},
                {"scratch": "sa"},
                {"param": 4},
                {"param": 6},
                {"i64": 0},
                {"i64": 0},
            ],
            "grid": [{"ceil_div": ["tokens", 16]}, k // 128, 1],
            "block": q_meta["block"],
            "shared_mem": q_meta["shared_mem"],
        }
    ]
    lower = 1
    buckets = (32, 64, 128, 256, 512, 1024)
    for idx, rows in enumerate(buckets):
        chosen = tuning[f"{rows},{n},{k}"]
        meta = dsl[chosen["module"]]
        when = {"var": "tokens", "min": lower}
        if idx + 1 < len(buckets):
            when["max"] = rows
        launches.append(
            {
                "module": module(chosen["module"]),
                "entry": meta["entry"],
                "params": list(W8A8_PARAMS) + meta["abi_tail"],
                "args": [
                    {"scratch": "aq"},
                    {"scratch": "sa"},
                    {"param": 1},
                    {"param": 2},
                    {"param": 3},
                    {"param": 4},
                    {"param": 5},
                    {"param": 6},
                    {"i64": 0},
                    {"i64": 0},
                ],
                "grid": [{"ceil_div": ["tokens", chosen["bm"]]}, {"ceil_div": [n, chosen["bn"]]}, 1],
                "block": meta["block"],
                "shared_mem": meta["shared_mem"],
                "when": when,
            }
        )
        lower = rows + 1
    return {
        "params": list(PARAMS),
        "impl": {
            "scratch": {
                "aq": {"dtype": "fp8e4m3", "shape": ["tokens", k]},
                "sa": {"dtype": "f32", "shape": ["tokens", k // 128]},
            },
            "launches": launches,
        },
    }
