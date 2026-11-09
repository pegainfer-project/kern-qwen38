import argparse
import ctypes
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(
    description="Benchmark the pinned FlashInfer W8A8 candidate separately from serving"
)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--out", type=Path, default=Path("out/flashinfer-gemm.json"))
args = parser.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
fn = lib.run_fp8
fn.argtypes = (
    [ctypes.c_void_p, ctypes.c_ulonglong] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
)
fn.restype = ctypes.c_int
workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
results = []
for m in (16, 32, 64, 128):
    n, k = 5120, 17408
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    sw = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32)
    sa = a.float().view(m, k // 128, 128).abs().amax(-1).clamp_min(1e-12) / 448
    aq = (a.float().view(m, k // 128, 128) / sa[:, :, None]).view(m, k).to(torch.float8_e4m3fn)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def run(aq=aq, w=w, sa=sa, sw=sw, out=out, m=m, n=n, k=k):
        rc = fn(
            workspace.data_ptr(),
            workspace.numel(),
            *[x.data_ptr() for x in (aq, w, sa, sw, out)],
            m,
            n,
            k,
            torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError(rc)

    run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        run()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / 100
    ref = a.float() @ w.float().T
    error = (out.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()
    result = {
        "m": m,
        "n": n,
        "k": k,
        "gemm_ms": ms,
        "effective_GBs": w.numel() / ms / 1e6,
        "effective_TFLOPs": 2 * m * n * k / ms / 1e9,
        "relative_RMS_error_vs_BF16_input": float(error),
    }
    results.append(result)
    print(json.dumps(result), flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(results, indent=2) + "\n")
