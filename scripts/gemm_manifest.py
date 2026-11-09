"""One autotuned GEMM interface for ordinary decode and speculative verification."""

PARAMS = (
    "in buffer<bf16>",
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
