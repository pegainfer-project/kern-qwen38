# Deployed manifest snapshots

## Qwen3.8-27B FP8 + DSpine, SM120, 2026-11-10

[`qwen38-dspine-sm120-2026-11-10.json`](qwen38-dspine-sm120-2026-11-10.json)
is a byte-for-byte snapshot of the manifest used by the running single-GPU
service on 2026-11-10. It was copied from the deployed
`out/qwen38-dspine/model.json`, whose modification time predates this process's
startup. The generated manifest is captured directly; its identity does not
depend on reproducing the current generator working tree.

| Property | Value |
|---|---|
| Manifest SHA-256 | `ae10b8f7e722dcbc73dafe731d7261eadeae9f4f8f6913fbb192d0ea1f1e8253` |
| Manifest schema | `7` |
| Manifest model | `qwen3.8-27b-fp8-dspine` |
| Public model ID | `Qwen/Qwen3.8-27B-FP8` |
| GPU | One RTX PRO 5000 72GB, SM120 |
| Kern runtime commit | `0d50183` |
| Manifest generator commit | `99337ab686156dc5bda29e278330ff419ce9298e` |
| Service started (Beijing, UTC+08) | 2026-11-10 12:25:09 |
| Prefill chunk / maximum batch | 1024 / 16 |
| Speculative round | DSpine, 16 rows |
| KV tier | Local SSD, 1024 GiB reserved, 16 GiB pinned staging |
| Tool parser | `qwen3_coder` |
| Kernel module digests verified | 53 |

The generator at `99337ab686156dc5bda29e278330ff419ce9298e` reproduces this file byte for byte
using the existing deployed kernel build and calibration inputs. This checks
manifest generation; it does not claim a fresh CUDA rebuild is byte-identical.

The JSON retains every original module name and SHA-256. Every referenced
module was checked against the deployed kernel bundle when this snapshot was
captured. Kernels, weights and host-specific `kern.toml` paths are not included
in this snapshot. Use the referenced kernel bytes and original target/draft
weights when loading it; the manifest file alone is not a complete model bundle.

From the repository root, validate the file with Kern:

```sh
kern verify manifests/qwen38-dspine-sm120-2026-11-10.json
sha256sum manifests/qwen38-dspine-sm120-2026-11-10.json
```

Use a GitHub URL pinned to the snapshot commit when sharing this artifact.
Later generator or deployment changes do not change this recorded version.
