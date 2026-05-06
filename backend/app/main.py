"""FastAPI app — the llamador control plane.

Two-tier auth model:
  • Control plane  (`/api/*`)  → optional Settings.api_token (single value)
  • Inference     (`/v1/*`)    → KeyStore (multiple revocable keys)

Both default to "open" if the corresponding store is empty (LAN-trust). The
moment you set api_token / create your first inference key, that surface
becomes authenticated.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .autotune import AutotuneRunner
from .config import (
    RuntimeConfig,
    get_settings,
    load_config,
    save_config,
    to_engine_env,
)
from .docker_manager import DockerManager, EngineNotFound
from .gpu_monitor import GPUMonitor
from .key_store import KeyStore
from .model_manager import ModelManager

log = logging.getLogger("llamador")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("llamador backend %s starting", __version__)
    yield
    log.info("llamador backend stopping")


app = FastAPI(title="llamador control plane", version=__version__, lifespan=lifespan)

settings = get_settings()
docker_mgr = DockerManager()
model_mgr = ModelManager()
gpu_monitor = GPUMonitor(docker_mgr)
autotune = AutotuneRunner(docker_mgr)
keys = KeyStore()


# ---------------------------------------------------------------------- auth
def auth(authorization: str | None = Header(default=None)) -> None:
    """Control-plane gate. Open if api_token unset."""
    if not settings.api_token:
        return
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing token")


def _client_ip(request: Request) -> str | None:
    """Best-effort real IP. Behind Caddy we get X-Forwarded-For (first hop is
    the original client); fall back to the connection peer when header is
    missing (e.g. direct LAN call to the backend on the docker network)."""
    fwd = request.headers.get("x-forwarded-for", "").strip()
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else None


def auth_inference(request: Request) -> dict | None:
    """Inference gate. Open if no keys exist; require Bearer match otherwise.

    On match, the matching key record is updated with the requester's IP,
    User-Agent, and a per-request counter so the UI can show who's been
    using each key.
    """
    if not keys.has_keys():
        return None
    authorization = request.headers.get("authorization")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    presented = authorization.split(" ", 1)[1].strip()
    rec = keys.verify(
        presented,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    if not rec:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
    return rec


# -------------------------------------------------------------- /api/health
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


# -------------------------------------------------------------- /api/status
@app.get("/api/status", dependencies=[Depends(auth)])
def get_status() -> dict:
    cfg = load_config()
    engine = docker_mgr.status()
    return {
        "engine": engine,
        "config": cfg.model_dump(),
        "models": model_mgr.list_models(),
        "auth": {
            "control_plane_locked": bool(settings.api_token),
            "inference_locked": keys.has_keys(),
            "key_count": len(keys.list()),
        },
    }


# -------------------------------------------------------------- /api/config
@app.get("/api/config", dependencies=[Depends(auth)])
def get_config() -> RuntimeConfig:
    return load_config()


@app.put("/api/config", dependencies=[Depends(auth)])
def put_config(cfg: RuntimeConfig, restart: bool = False) -> dict:
    save_config(cfg)
    if restart:
        try:
            docker_mgr.restart()
        except EngineNotFound as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"saved": True, "restarted": restart, "engine_env_preview": to_engine_env(cfg)}


# ----------------------------------------------------------- engine power
@app.post("/api/restart", dependencies=[Depends(auth)])
def restart_engine() -> dict:
    try: docker_mgr.restart()
    except EngineNotFound as e: raise HTTPException(404, str(e)) from e
    return {"restarted": True}


@app.post("/api/stop", dependencies=[Depends(auth)])
def stop_engine() -> dict:
    try: docker_mgr.stop()
    except EngineNotFound as e: raise HTTPException(404, str(e)) from e
    return {"stopped": True}


@app.post("/api/start", dependencies=[Depends(auth)])
def start_engine() -> dict:
    try: docker_mgr.start()
    except EngineNotFound as e: raise HTTPException(404, str(e)) from e
    return {"started": True}


# ----------------------------------------------------------------- models
@app.get("/api/models", dependencies=[Depends(auth)])
def list_models() -> list[dict]:
    return model_mgr.list_models()

@app.post("/api/models/pull", dependencies=[Depends(auth)])
def pull_model(repo: str, filename: str) -> dict:
    return {"task_id": model_mgr.start_pull(repo, filename)}

@app.delete("/api/models/{name}", dependencies=[Depends(auth)])
def delete_model(name: str) -> dict:
    try: model_mgr.delete(name)
    except FileNotFoundError as e: raise HTTPException(404, str(e)) from e
    except PermissionError as e:    raise HTTPException(400, str(e)) from e
    return {"deleted": name}


# ------------------------------------------------------------------ tasks
@app.get("/api/tasks", dependencies=[Depends(auth)])
def list_tasks() -> list[dict]: return model_mgr.list_tasks()

@app.get("/api/tasks/{task_id}", dependencies=[Depends(auth)])
def get_task(task_id: str) -> dict:
    t = model_mgr.get_task(task_id)
    if not t: raise HTTPException(404, "no such task")
    return {"id": task_id, **t}


# ----------------------------------------------------------------- rebuild
_build_streams: dict[str, asyncio.Queue] = {}

@app.post("/api/rebuild", dependencies=[Depends(auth)])
async def rebuild_engine() -> dict:
    task_id = f"build-{int(asyncio.get_event_loop().time())}"
    q: asyncio.Queue = asyncio.Queue(maxsize=2048)
    _build_streams[task_id] = q
    async def _build():
        loop = asyncio.get_event_loop()
        try:
            await q.put({"stream": f"[rebuild] starting {task_id}\n"})
            for chunk in await loop.run_in_executor(None, lambda: list(docker_mgr.rebuild_image())):
                await q.put(chunk)
            await q.put({"stream": "[rebuild] image built. Restart the engine to apply.\n"})
        except Exception as e:  # noqa: BLE001
            await q.put({"errorDetail": {"message": str(e)}})
        finally:
            await q.put({"_eof": True})
    asyncio.create_task(_build())
    return {"task_id": task_id}


# --------------------------------------------------------------- /api/gpu
@app.get("/api/gpu", dependencies=[Depends(auth)])
def get_gpu() -> dict: return gpu_monitor.stats()


# ------------------------------------------------------------- autotune
@app.post("/api/autotune", dependencies=[Depends(auth)])
async def start_autotune(sweep: str = "48,40,36,32,28,24,20", pp: int = 512,
                         tg: int = 128, threads: int = 0, apply_best: bool = True) -> dict:
    try:
        sweep_list = [int(x.strip()) for x in sweep.split(",") if x.strip()]
    except ValueError as e:
        raise HTTPException(400, f"bad sweep value: {e}") from e
    if not sweep_list: raise HTTPException(400, "sweep must have at least one value")
    return {"task_id": autotune.start(sweep=sweep_list, pp=pp, tg=tg,
                                       threads=threads, apply_best=apply_best)}

@app.get("/api/autotune/{task_id}", dependencies=[Depends(auth)])
def get_autotune(task_id: str) -> dict:
    st = autotune.get(task_id)
    if not st: raise HTTPException(404, "no such autotune task")
    return {"id": st.task_id, "state": st.state, "model": st.model_file,
            "sweep": st.sweep, "steps": [s.__dict__ for s in st.steps],
            "best": st.best.__dict__ if st.best else None, "error": st.error}

@app.websocket("/api/autotune/{task_id}/ws")
async def ws_autotune(ws: WebSocket, task_id: str) -> None:
    await ws.accept()
    q = autotune.stream(task_id)
    if q is None:
        await ws.send_json({"type": "error", "message": "no such task"})
        await ws.close(); return
    try:
        while True:
            item = await q.get()
            await ws.send_json(item)
            if item.get("type") == "done":
                await ws.close(); return
    except WebSocketDisconnect: return


# ============================== /api/keys ==================================
class KeyCreate(BaseModel):
    name: str = "unnamed"

@app.get("/api/keys", dependencies=[Depends(auth)])
def list_keys() -> list[dict]:
    return keys.list(include_secrets=False)

@app.post("/api/keys", dependencies=[Depends(auth)])
def create_key(body: KeyCreate) -> dict:
    """Create a new key. The full secret is returned EXACTLY ONCE here."""
    return keys.create(body.name)

@app.delete("/api/keys/{kid}", dependencies=[Depends(auth)])
def revoke_key(kid: str) -> dict:
    if not keys.revoke(kid): raise HTTPException(404, "no such key")
    return {"revoked": kid}

class KeyPatch(BaseModel):
    enabled: bool | None = None
    name: str | None = None

@app.patch("/api/keys/{kid}", dependencies=[Depends(auth)])
def patch_key(kid: str, body: KeyPatch) -> dict:
    """Update enabled state and/or name on a key. At least one field required."""
    if body.enabled is None and body.name is None:
        raise HTTPException(400, "provide at least one of: enabled, name")
    out: dict = {"id": kid}
    if body.enabled is not None:
        if not keys.set_enabled(kid, body.enabled):
            raise HTTPException(404, "no such key")
        out["enabled"] = body.enabled
    if body.name is not None:
        if not keys.rename(kid, body.name):
            raise HTTPException(404, "no such key")
        out["name"] = body.name
    return out


# ============================ /api/capabilities ============================
@app.get("/api/capabilities", dependencies=[Depends(auth)])
async def capabilities(request: Request) -> dict:
    cfg = load_config()
    base = f"{request.url.scheme}://{request.url.netloc}"
    props: dict = {}
    try:
        async with httpx.AsyncClient(timeout=2.0) as cx:
            r = await cx.get(f"{settings.engine_url}/props")
            if r.status_code == 200: props = r.json()
    except httpx.HTTPError:
        pass
    return {
        "base_url": f"{base}/v1",
        "model": cfg.alias,
        "model_file": cfg.model_file,
        "context_length": cfg.ctx,
        "kv_cache_type": cfg.kv_type,
        "flash_attention": cfg.flash_attn,
        "n_cpu_moe": cfg.n_cpu_moe,
        "auth": {
            "required": keys.has_keys(),
            "scheme": "Bearer",
            "header": "Authorization",
        },
        "supports": {
            "chat_completions": True,
            "completions": True,
            "embeddings": False,
            "streaming": True,
            "tools": True,
            "vision": False,
        },
        "engine": {"props": props},
    }


# ---------------------------------------------------------------- metrics
@app.get("/api/metrics", dependencies=[Depends(auth)])
async def metrics_passthrough() -> Response:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{settings.engine_url}/metrics")
        except httpx.HTTPError as e:
            return PlainTextResponse(f"# engine unreachable: {e}\n", status_code=503)
    return Response(content=r.content, media_type="text/plain", status_code=r.status_code)


# ----------------------------------------------------------- /api/usage
_USAGE_KEYS = {
    "llamacpp:prompt_tokens_total":            "prompt_tokens_total",
    "llamacpp:tokens_predicted_total":         "tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total": "predict_seconds_total",
    "llamacpp:prompt_seconds_total":           "prompt_seconds_total",
    "llamacpp:n_decode_total":                 "n_decode_total",
    "llamacpp:kv_cache_tokens":                "kv_cache_tokens",
    "llamacpp:kv_cache_usage_ratio":           "kv_cache_usage",
    "llamacpp:requests_processing":            "requests_processing",
    "llamacpp:requests_deferred":              "requests_deferred",
}

def _parse_prom(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "{" in line:
            name, rest = line.split("{", 1)
            try: _, value = rest.rsplit("}", 1)
            except ValueError: continue
            value = value.strip()
        else:
            parts = line.split()
            if len(parts) < 2: continue
            name, value = parts[0], parts[1]
        try: out[name] = float(value)
        except ValueError: continue
    return out

@app.get("/api/usage", dependencies=[Depends(auth)])
async def get_usage() -> dict:
    """Structured token-usage counters. Frontend polls ~1 Hz, diffs for rates."""
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r = await client.get(f"{settings.engine_url}/metrics")
        except httpx.HTTPError as e:
            return {"available": False, "error": str(e), "ts": asyncio.get_event_loop().time()}
    if r.status_code != 200:
        return {"available": False, "error": f"engine returned {r.status_code}",
                "ts": asyncio.get_event_loop().time()}
    parsed = _parse_prom(r.text)
    return {
        "available": True,
        "ts": asyncio.get_event_loop().time(),
        "counters": {label: parsed.get(metric, 0.0) for metric, label in _USAGE_KEYS.items()},
    }


# ============================ /v1 OpenAI passthrough =======================
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_passthrough(path: str, request: Request) -> Response:
    """OpenAI-compatible proxy with key-store auth."""
    auth_inference(request)
    url = f"{settings.engine_url}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "authorization", "content-length")}
    is_stream = b'"stream"' in body and b'true' in body
    client = httpx.AsyncClient(timeout=None)
    try:
        if is_stream:
            req = client.build_request(method=request.method, url=url,
                                       params=request.query_params,
                                       content=body, headers=headers)
            r = await client.send(req, stream=True)
            async def _gen():
                try:
                    async for chunk in r.aiter_raw(): yield chunk
                finally:
                    await r.aclose(); await client.aclose()
            return StreamingResponse(
                _gen(), status_code=r.status_code,
                headers={k: v for k, v in r.headers.items() if k.lower() != "content-length"},
                media_type=r.headers.get("content-type", "text/event-stream"))
        r = await client.request(method=request.method, url=url,
                                  params=request.query_params, content=body, headers=headers)
        await client.aclose()
        return Response(content=r.content, status_code=r.status_code,
                        headers={k: v for k, v in r.headers.items() if k.lower() != "content-length"})
    except Exception:
        await client.aclose()
        raise


# ---------------------------------------------------------------- ws logs
@app.websocket("/api/logs")
async def ws_logs(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        gen = await loop.run_in_executor(None, docker_mgr.stream_logs)
        while True:
            chunk = await loop.run_in_executor(None, next, gen, b"")
            if not chunk:
                await asyncio.sleep(0.5); continue
            await ws.send_text(chunk.decode(errors="replace"))
    except WebSocketDisconnect: return
    except EngineNotFound as e:
        await ws.send_text(f"[backend] engine not found: {e}"); await ws.close()
    except Exception as e:  # noqa: BLE001
        await ws.send_text(f"[backend] log stream error: {e}"); await ws.close()


# ------------------------------------------------------- ws build logs
@app.websocket("/api/build/{task_id}")
async def ws_build(ws: WebSocket, task_id: str) -> None:
    await ws.accept()
    q = _build_streams.get(task_id)
    if not q:
        await ws.send_text(f"[backend] no such build task: {task_id}")
        await ws.close(); return
    try:
        while True:
            item = await q.get()
            if item.get("_eof"): await ws.close(); return
            if "stream" in item: await ws.send_text(item["stream"])
            elif "errorDetail" in item:
                await ws.send_text(f"[error] {item['errorDetail'].get('message')}\n")
                await ws.close(); return
            elif "status" in item:
                await ws.send_text(f"[{item.get('status')}] {item.get('progress', '')}\n")
    except WebSocketDisconnect: return


# -------------------------------------------------------- static frontend
try:
    app.mount("/", StaticFiles(directory="/static", html=True), name="static")
except RuntimeError:
    log.warning("/static not mounted — frontend reachable only via Caddy")
