# Architecture

```
                 ┌─────────────────────────────────────────────┐
       (browser) │                Caddy :8088                  │
   ──────────────┤  reverse proxy + WebSocket pass-through      │
                 └──┬──────────────┬───────────────┬───────────┘
                    │ /api/*       │ /v1/*         │ /
                    ▼              ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   backend    │ │ llama-engine │ │  frontend    │
            │   (FastAPI)  │ │ (TurboQuant) │ │ (nginx+SPA)  │
            └──────┬───────┘ └──────┬───────┘ └──────────────┘
                   │                │
                   │       ┌────────▼────────┐
                   │       │ /config/        │  ← engine reads this on each boot
                   │       │   engine.json   │
                   │       └────────▲────────┘
                   │                │ writes
                   ├────────────────┘
                   │
                   ├──► /var/run/docker.sock   (restart, rebuild, stream logs)
                   ├──► /models/               (HF downloads land here)
                   └──► /workspace/llama-engine (build context for rebuilds)
```

## Why the engine reads JSON instead of just env vars

When the UI changes a parameter, we need the engine to pick it up. Two clean options exist:

1. **Recreate the container** with new env vars (preserve mounts, network, etc.). Doable via the docker SDK but error-prone — you have to manually preserve everything compose set up, and small omissions (a label, a resource reservation) cause subtle drift.
2. **Mount a config file** that the engine re-reads on boot, then `restart()` the container. The container stays "compose-managed" — same name, same labels, same mounts — and the only thing that changed is what the entrypoint reads off disk.

We picked (2). The flow:
1. UI `PUT /api/config` → backend writes `data/config/engine.json`.
2. UI clicks _Save & restart_ → backend calls `container.restart()`.
3. Engine entrypoint reads `engine.json`, exports the values as env, exec's `llama-server` with new flags.

This makes _every change_ idempotent and durable across host reboots.

## Why a single Caddy front-door

Lets us:
- Serve `/`, `/api`, `/v1`, `/metrics` from one origin (no CORS gymnastics).
- Add TLS / auth / rate-limits in one place when you go beyond the LAN.
- Survive a backend or frontend restart with the user just refreshing the page.

If you don't need Caddy (e.g., already running Traefik or nginx in front), drop the service from `docker-compose.yml` and expose `backend` / `frontend` / `llama-engine` directly.

## Why nginx for the frontend

There's no build step — Tailwind and Alpine come from CDN. nginx-alpine at ~7 MB is the smallest, simplest static server. If you eventually add a build step (Vite, etc.), this is the layer to swap out.

## What lives in `data/`

```
data/
├── models/        ← *.gguf files; survives docker compose down
└── config/
    └── engine.json   ← runtime parameters the UI mutates
```

Both bind-mounted into both `llama-engine` and `backend`.
