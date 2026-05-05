# Pascal notes (GTX 1080 / GTX 1070 / Quadro P-series)

Pascal predates tensor cores. A few defaults in modern llama.cpp work *against* you on this generation; here's the cheat sheet.

## What's set in this repo

| Build flag | Why |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES=61` | Targets Pascal compute 6.1 specifically. Without it, modern CUDA toolchains may strip the kernel and you'll get `no kernel image is available for execution on the device`. |
| `GGML_CUDA_F16=OFF` | Pascal FP16 throughput is **1:64** of FP32. Don't promote to FP16 — stay in FP32. |
| `GGML_CUDA_FA=ON` + `GGML_CUDA_FA_ALL_QUANTS=ON` | Compiles FlashAttention CUDA kernels for every KV-quant combo, including TurboQuant's. |

| Runtime env | Why |
|---|---|
| `GGML_CUDA_FORCE_MMQ=1` | Force matrix-multiply-quantized kernels (INT8/FP32 paths) instead of cuBLAS-FP16 GEMM. Big win on Pascal. |

## What about FlashAttention?

FA on Pascal works, but only the FP32 path. Recent llama.cpp masters have fixed earlier "no device code" issues. If you ever hit a FA panic, set:

```bash
# .env
FLASH_ATTN=off
```

…and the engine reads it on next boot.

## TurboQuant cache types on Pascal

`turbo3` and `turbo4` use the same FA kernels, so they should work — but performance gains are smaller than on Ampere because FA itself is FP32-only on Pascal. Still worth trying once the basics run; the biggest win is at long context (32K+), where `q8_0` KV would otherwise dominate VRAM.

## Other Pascal cards

If you have a different Pascal card, change `CUDA_ARCH` in `.env`:

| Card | `CUDA_ARCH` |
|---|---|
| Tesla P100 / Quadro GP100 | `60` |
| GTX 1080, 1070, Titan X(P), P40, P4 | `61` |
| Tegra X2 (Jetson TX2) | `62` |

## Newer cards

If you stop being a "GPU-poor enthusiast" and put an Ampere card in:

| Card family | `CUDA_ARCH` | Recommended changes |
|---|---|---|
| RTX 30xx / A-series | `86` | Set `GGML_CUDA_F16=ON` (full-rate FP16 + tensor cores) |
| RTX 40xx | `89` | Same as above |
| RTX 50xx | `90`/`120` | Same as above |
| H100 | `90` | Same as above; consider `GGML_CUDA_FA_TENSOR_CORES=ON` if exposed |
