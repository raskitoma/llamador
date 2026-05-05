# 🦙 llamador

**Run a 35B-parameter MoE model on a humble 8 GB GPU**, with a control plane to drive it from your browser.

A containerized stack around the [TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant) tuned for **NVIDIA Pascal (compute 6.1)** and the **Qwen3.6-35B-A3B** mixture-of-experts model. Inspired by [_Running a 35B AI Model on 6GB VRAM, FAST_](https://www.youtube.com/watch?v=8F_5pdcD3HY).

> _llamador_ is Spanish for "knocker" / "caller" — a knock on the llama's door. 🚪

[![hadolint](https://github.com/Raskitoma/llamador/actions/workflows/lint.yml/badge.svg)](./.github/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-emerald.svg)](LICENSE)

---

## Why this exists

Qwen3.6-35B-A3B has 35 B parameters total but only ~3 B active per token (it's a Mixture-of-Experts). The trick: keep the small "always-on" pieces (attention, embeddings, shared FFN) on the 8 GB GPU and stream the experts from system RAM. llama.cpp's `--n-cpu-moe N` flag does exactly that. TurboQuant adds aggressive KV-cache compression on top, so long contexts don't blow the VRAM budget either.

`llamador` packages the whole thing as Docker services and bolts a small FastAPI + Alpine.js control plane onto the side so you can change models, tweak knobs, restart the engine, rebuild binaries, and tail logs — all from the browser.

## Architecture

```
            ┌────────────────────── Caddy :8088 ──────────────────────┐
 Browser ──►│  /          → frontend (static SPA)                      │
            │  /api/*     → backend  (FastAPI control plane)            │
            │  /v1/*      → llama-engine (OpenAI-compatible API)        │
            └──────────────────────────────────────────────────────────┘
                          │              │                │
                  docker.sock    /config/engine.json    /models
                          │              │                │
                  ┌───────▼──────┐ ┌─────▼────────┐ ┌─────▼─────┐
                  │  backend     │ │ llama-engine │ │  GGUF     │
                  │  (FastAPI)   │ │ (TurboQuant) │ │  files    │
                  └──────────────┘ └──────┬───────┘ └───────────┘
                                          │ NVIDIA runtime
                                       GTX 1080 (Pascal, 8 GB)
```

Optional profiles: **`--profile chat`** adds [open-webui](https://github.com/open-webui/open-webui) (a polished chat UI) and **`--profile metrics`** adds Prometheus + Grafana scraping the engine's `/metrics`.

## Quick start

```bash
git clone https://github.com/raskitoma/llamador.git
cd llamador
./deploy.sh --logs                   # one-shot production deploy + tail engine log
```

That single command:
1. Verifies Docker, the NVIDIA Container Toolkit, and GPU passthrough.
2. Generates an `API_TOKEN` if you don't have one yet.
3. Builds all images.
4. Starts the stack (`engine + backend + frontend + caddy`).
5. Waits until each service reports healthy.
6. Offers to download the default Qwen3.6 GGUF (~21 GB).
7. Tails `llama-engine` logs (because of `--logs`).

Useful flags:

| flag | what it does |
|---|---|
| `--logs [svc\|all]` | tail logs after deploying |
| `--rebuild` | no-cache image rebuild |
| `--no-model` | skip the GGUF download |
| `--systemd` | install `llamador.service` so the stack survives host reboots |
| `--profile chat` / `--profile metrics` | enable open-webui or prom+grafana (repeatable) |
| `--regenerate-token` | rotate the API token |
| `--token <hex>` | use a specific API token |
| `-h` / `--help` | print the inline help |

Once it's up:
- **Control panel:** http://your-host:8088/
- **OpenAI API:**    http://your-host:8088/v1
- **Engine /metrics:** http://your-host:8088/metrics

The control panel exposes everything: model picker, live GPU/VRAM tile, **live token-usage tile (decode rate, prompt rate, KV cache, sparkline)**, knobs (`--n-cpu-moe`, ctx, KV type, FlashAttention, threads, batch), restart/rebuild buttons, model downloader, live logs with **clear / pause / reconnect**, an auto-tune sweep, and a **Clients tab** for issuing API keys + copy-paste config snippets.

## Calling the API from a remote OpenWebUI (or anything OpenAI-compatible)

`llama-server` exposes a full OpenAI-compatible surface at `:8088/v1`. You point any client at that URL.

By default `/v1` is open on your LAN. To lock it down:

1. Open the **Clients** tab in the control panel.
2. Click _Generate key_ (give it a name like `openwebui-prod`).
3. Copy the key — **it's shown exactly once**.
4. Paste the snippet from the same tab into your client. The tab generates ready-to-paste configs for **Open WebUI**, **curl**, **Python (`openai`)**, and **OpenClaw / generic OpenAI clients**.

The moment any key exists, `/v1` requires `Authorization: Bearer sk-…`. Until then, it's open. Revoking a key in the UI is immediate — existing clients get a 401 on their next request.

## Two-tier auth model

| surface | gate | source of truth |
|---|---|---|
| **Control plane** (`/api/*`) | optional `API_TOKEN` env var | `.env` |
| **Inference** (`/v1/*`) | optional revocable keys | `data/config/keys.json` |

Both default to "open" (LAN-trust) when their respective stores are empty. Setting `API_TOKEN` or creating your first inference key snaps the corresponding gate shut. The two are independent — rotating one doesn't disturb the other.

## Knobs

All settings live-edit from the UI; the engine reads `data/config/engine.json` on each boot.

| knob | default | what it does |
|---|---|---|
| `model_file` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | filename inside `/models` |
| `n_cpu_moe` | `30` | how many MoE layers' experts to push to CPU RAM (model has 48) |
| `ctx` | `8192` | context window |
| `kv_type` | `q8_0` | KV-cache quant: `q8_0`, `q4_0`, `turbo3`, `turbo4`, … |
| `flash_attn` | `on` | toggles FlashAttention (FP32 path on Pascal) |
| `threads` | `auto` | CPU threads; `auto` uses all cores |
| `batch / ubatch` | `2048 / 2048` | prompt-processing batch sizes |
| `extra_args` | `""` | raw extra flags appended to `llama-server` |

### Tuning `--n-cpu-moe` on a GTX 1080

| `--n-cpu-moe` | VRAM use | decode speed |
|---|---|---|
| 48 (all to CPU) | ~3.5 GB | slowest |
| 36 | ~5 GB | safe |
| **30** _(default)_ | ~6.5 GB | good speed, ~1.5 GB free |
| 24 | ~7.5 GB | risky on 8 GB |
| <20 | OOM | — |

Use the sweep script to find your spot:

```bash
./scripts/bench.sh 24 28 32 36
```

## Pascal-specific tuning (already applied)

Pascal GPUs (GTX 9xx, 10xx, P-series) have FP16 throughput that is **1:64** of FP32 — modern llama.cpp defaults aren't great here. The image and runtime apply:

- `CMAKE_CUDA_ARCHITECTURES=61` at build time
- `GGML_CUDA_F16=OFF` at build time
- `GGML_CUDA_FORCE_MMQ=1` at runtime (forces integer matmul kernels)
- `GGML_CUDA_FA_ALL_QUANTS=ON` at build time (FA kernels for all KV quants)

If you're running on **Ampere+ (RTX 30xx, 40xx, A-series)** flip `CUDA_ARCH` in `.env` to `86`/`89` and consider removing the `F16=OFF` line — those cards have full-rate FP16.

## Stack profiles

```bash
docker compose up -d                       # default: engine + backend + frontend + caddy
docker compose --profile chat up -d        # also start open-webui on :3000
docker compose --profile metrics up -d     # also start prometheus :9090, grafana :3001
```

## Coexistence with Ollama

Ollama at `/home/raskitoma/ollama` is untouched. The default port collision matrix:

| port | service |
|---|---|
| 11434 | Ollama |
| 8088 | llamador (Caddy) |
| 3000 | open-webui (optional) |
| 9090 | prometheus (optional) |
| 3001 | grafana (optional) |

The GPU itself is the constraint — only one process can use the 8 GB at a time. Easiest: `sudo systemctl stop ollama` while you're using `llamador`.

## Updating

- **Update llama.cpp / TurboQuant binaries** — Click _Rebuild image_ in the UI. The backend pulls the latest commit on the configured branch and rebuilds the engine image. Then click _Restart_.
- **Update llamador itself** — `git pull && docker compose build && docker compose up -d`.

## Security

Default mode is **LAN-trust**: anyone who can reach `:8088` can change models, restart the engine, and call inference. For exposure beyond your LAN:

1. **Lock the control plane** — set `API_TOKEN=<long random string>` in `.env` (or use the deploy menu's _API token_ entry). `/api/*` will require `Authorization: Bearer …`.
2. **Lock inference** — open the **Clients** tab and create at least one API key. `/v1/*` will require `Authorization: Bearer sk-…`.
3. **TLS** — edit `caddy/Caddyfile` to use your real domain — Caddy fetches a Let's Encrypt cert automatically.
4. Consider Caddy's [`basic_auth`](https://caddyserver.com/docs/caddyfile/directives/basic_auth) or [`forward_auth`](https://caddyserver.com/docs/caddyfile/directives/forward_auth) for the control plane on top.

⚠️ The backend mounts `/var/run/docker.sock` — that's effectively root on the host. Don't expose `/api` to the internet without auth.

## API summary

```
GET    /api/health                 liveness
GET    /api/status                 engine + config + models + auth state
GET    /api/config                 current RuntimeConfig
PUT    /api/config?restart=true    write config (and optionally bounce engine)
POST   /api/restart                restart engine container
POST   /api/stop|start             power
POST   /api/rebuild                rebuild engine image (long task)
GET    /api/models                 list /models contents
POST   /api/models/pull            ?repo=…&filename=…
DELETE /api/models/{name}          remove a GGUF
GET    /api/tasks                  list background tasks
GET    /api/gpu                    live nvidia-smi snapshot
GET    /api/usage                  live llama-server token counters (poll ~1 Hz)
GET    /api/capabilities           model name, ctx, supported features, base URL
GET    /api/keys                   list inference keys (no secrets)
POST   /api/keys                   create new key — returns secret EXACTLY once
DELETE /api/keys/{kid}             revoke a key
PATCH  /api/keys/{kid}?enabled=…   soft-disable / re-enable
POST   /api/autotune               sweep --n-cpu-moe via llama-bench
GET    /api/autotune/{id}          poll sweep state
GET    /api/metrics                proxy of llama-server /metrics
WS     /api/logs                   live engine logs
WS     /api/build/{task}           live image-build output
WS     /api/autotune/{id}/ws       live sweep progress
ANY    /v1/*                       OpenAI-compat passthrough (key-gated)
```

## Roadmap / ideas

- Quantization tier presets (chat / code / long-context)
- Multi-GPU split for hosts with more than one card
- Snapshot/restore KV cache states
- Optional vector DB sidecar (qdrant / sqlite-vec) for RAG
- Per-model presets (different `--n-cpu-moe` per model)
- Webhook on `engine.health == unhealthy`

## License

MIT — see [LICENSE](LICENSE).
