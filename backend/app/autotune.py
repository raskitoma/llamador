"""Auto-tune --n-cpu-moe by running llama-bench across a sweep of values.

The GPU is single-occupancy on small VRAM, so we must stop the engine before
each bench run. Sequence per value N:

    docker run --rm --gpus all -v <models>:/models <engine-image> \
        llama-bench -m /models/<model> -ngl 999 --n-cpu-moe N \
                    -fa 1 -p <pp> -n <tg> -t <threads>

We capture stdout, parse the tg<n> row from llama-bench's markdown table, and
record (n_cpu_moe, tg_tokens_per_sec). If the run exits non-zero we treat it
as OOM/error and continue.

After the sweep we pick the best N (highest tg) and (optionally) write it to
engine.json + restart. We bias *slightly* toward higher N (more VRAM headroom)
when two values are within 3% of each other — gives you a safer config that
won't OOM on a longer prompt.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import docker

from .config import RuntimeConfig, get_settings, load_config, save_config
from .docker_manager import DockerManager

log = logging.getLogger(__name__)

# llama-bench markdown row: numbers with optional ± stddev.
_TG_ROW = re.compile(
    r"\|\s*tg\d+\s*\|\s*([\d.]+)\s*(?:±\s*[\d.]+)?\s*\|", re.IGNORECASE
)
_PP_ROW = re.compile(
    r"\|\s*pp\d+\s*\|\s*([\d.]+)\s*(?:±\s*[\d.]+)?\s*\|", re.IGNORECASE
)


@dataclass
class SweepStep:
    n_cpu_moe: int
    kv_type: str = "q8_0"
    tg_tps: float | None = None
    pp_tps: float | None = None
    ok: bool = False
    raw: str = ""
    error: str | None = None


DEFAULT_STEP_TIMEOUT_SEC = 600  # 10 minutes per llama-bench run


@dataclass
class AutotuneState:
    task_id: str
    model_file: str
    sweep: list[int]
    kv_types: list[str] = field(default_factory=list)  # [] = single type from cfg
    pp: int = 512
    tg: int = 128
    threads: int = 0
    apply_best: bool = True
    state: str = "queued"  # queued | running | done | error | cancelled
    steps: list[SweepStep] = field(default_factory=list)
    best: SweepStep | None = None
    error: str | None = None
    cancel_requested: bool = False
    current_container_id: str | None = None  # so cancel can SIGKILL the bench


class AutotuneRunner:
    def __init__(self, dm: DockerManager) -> None:
        self.dm = dm
        self.client = docker.from_env()
        self.settings = get_settings()
        self._tasks: dict[str, AutotuneState] = {}
        self._streams: dict[str, asyncio.Queue[dict]] = {}

    # ---------------------------------------------------------- public API
    def start(
        self,
        sweep: Iterable[int] | None = None,
        kv_types: Iterable[str] | None = None,
        pp: int = 512,
        tg: int = 128,
        threads: int = 0,
        apply_best: bool = True,
        model_file: str | None = None,
    ) -> str:
        cfg = load_config()
        target_model = model_file or cfg.model_file
        sweep_list = list(sweep) if sweep else [48, 40, 36, 32, 28, 24, 20]
        # Empty list = bench against whatever KV type is in the running config.
        # That keeps backward compat with old single-axis callers.
        kv_list = [k.strip() for k in (kv_types or []) if k and k.strip()]

        task_id = f"autotune-{int(asyncio.get_event_loop().time())}"
        st = AutotuneState(
            task_id=task_id,
            model_file=target_model,
            sweep=sweep_list,
            kv_types=kv_list,
            pp=pp,
            tg=tg,
            threads=threads,
            apply_best=apply_best,
        )
        self._tasks[task_id] = st
        self._streams[task_id] = asyncio.Queue(maxsize=1024)

        asyncio.create_task(self._run(st))
        return task_id

    def get(self, task_id: str) -> AutotuneState | None:
        return self._tasks.get(task_id)

    def stream(self, task_id: str) -> asyncio.Queue[dict] | None:
        return self._streams.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Mark a sweep cancelled and kill its currently-running bench container.

        The runner loop checks `cancel_requested` between steps; killing the
        active container makes the in-flight step return immediately with a
        non-zero exit so we don't have to wait for it to finish naturally.
        """
        st = self._tasks.get(task_id)
        if st is None or st.state not in ("queued", "running"):
            return False
        st.cancel_requested = True
        cid = st.current_container_id
        if cid:
            try:
                self.client.containers.get(cid).kill()
            except Exception as e:  # noqa: BLE001
                log.info("cancel: container %s already gone: %s", cid, e)
        return True

    # ------------------------------------------------------------- runner
    async def _run(self, st: AutotuneState) -> None:
        q = self._streams[st.task_id]
        loop = asyncio.get_running_loop()

        async def emit(kind: str, **payload):
            await q.put({"type": kind, **payload})

        def emit_threadsafe(kind: str, **payload):
            """Schedule a queue put from a worker thread (bench streamer)."""
            try:
                loop.call_soon_threadsafe(q.put_nowait, {"type": kind, **payload})
            except RuntimeError:
                pass  # loop may be shutting down

        # Resolve the KV type axis. Empty list = single type taken from the
        # running config (backward-compat: old "1-D sweep over n_cpu_moe").
        cfg_now = load_config()
        kv_axis = st.kv_types or [cfg_now.kv_type]

        st.state = "running"
        await emit(
            "start",
            sweep=st.sweep,
            kv_types=kv_axis,
            model=st.model_file,
            pp=st.pp,
            tg=st.tg,
            threads=st.threads,
        )

        # 1) ensure the engine is stopped (GPU exclusivity)
        try:
            self.dm.stop()
            await emit("info", message="engine stopped")
        except Exception as e:  # noqa: BLE001
            await emit("info", message=f"engine already down: {e}")

        engine_image = self._engine_image()

        # 2) 2-D sweep: outer loop is KV type, inner loop is --n-cpu-moe.
        loop = asyncio.get_event_loop()
        cancelled = False
        for kv in kv_axis:
            if cancelled:
                break
            await emit("info", message=f"--cache-type-k/v {kv}")
            for n in st.sweep:
                if st.cancel_requested:
                    cancelled = True
                    await emit("info", message=f"cancelled before kv={kv} n_cpu_moe={n}")
                    break
                step = SweepStep(n_cpu_moe=n, kv_type=kv)
                st.steps.append(step)
                await emit("step_start", n_cpu_moe=n, kv_type=kv)
                try:
                    def on_line(line: str, _n=n, _kv=kv) -> None:
                        emit_threadsafe("bench_log", n_cpu_moe=_n, kv_type=_kv, line=line)

                    output = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._bench_once, engine_image, st, n, kv, on_line,
                        ),
                        timeout=DEFAULT_STEP_TIMEOUT_SEC,
                    )
                    step.raw = output
                    pp_m = _PP_ROW.search(output)
                    tg_m = _TG_ROW.search(output)
                    if tg_m:
                        step.tg_tps = float(tg_m.group(1))
                    if pp_m:
                        step.pp_tps = float(pp_m.group(1))
                    step.ok = step.tg_tps is not None
                    if not step.ok and not step.error:
                        step.error = "no tg row in output (likely OOM)"
                except asyncio.TimeoutError:
                    step.error = f"timed out after {DEFAULT_STEP_TIMEOUT_SEC}s"
                    step.ok = False
                    cid = st.current_container_id
                    if cid:
                        try: self.client.containers.get(cid).kill()
                        except Exception: pass
                except Exception as e:  # noqa: BLE001
                    step.error = str(e)
                    step.ok = False
                finally:
                    st.current_container_id = None
                await emit(
                    "step_end",
                    n_cpu_moe=n,
                    kv_type=kv,
                    ok=step.ok,
                    pp_tps=step.pp_tps,
                    tg_tps=step.tg_tps,
                    error=step.error,
                )

        if st.cancel_requested:
            st.state = "cancelled"
            await emit("done", best=None, error="cancelled by user")
            try: self.dm.start()
            except Exception: pass
            return

        # 3) pick the best (highest tg, tie-break toward higher n_cpu_moe for headroom)
        candidates = [s for s in st.steps if s.ok and s.tg_tps is not None]
        if not candidates:
            st.state = "error"
            st.error = "no successful runs in sweep"
            await emit("done", best=None, error=st.error)
            return

        top_tg = max(s.tg_tps for s in candidates)  # type: ignore[arg-type]
        # within 3% → prefer higher n_cpu_moe (more VRAM headroom)
        cutoff = top_tg * 0.97
        within = [s for s in candidates if (s.tg_tps or 0) >= cutoff]
        st.best = max(within, key=lambda s: s.n_cpu_moe)

        await emit(
            "best",
            n_cpu_moe=st.best.n_cpu_moe,
            kv_type=st.best.kv_type,
            tg_tps=st.best.tg_tps,
            pp_tps=st.best.pp_tps,
        )

        # 4) optionally apply both knobs
        if st.apply_best and st.best:
            cfg = load_config()
            cfg.n_cpu_moe = st.best.n_cpu_moe
            # Only overwrite kv_type if the schema accepts it — pydantic
            # validates against the KVType literal, so an unknown turbo
            # variant from a future image won't get persisted.
            try:
                cfg.kv_type = st.best.kv_type  # type: ignore[assignment]
            except Exception:
                pass
            save_config(cfg)
            await emit("applied", n_cpu_moe=st.best.n_cpu_moe, kv_type=st.best.kv_type)

        # 5) restart engine (we stopped it earlier)
        try:
            self.dm.start()
            await emit("info", message="engine started")
        except Exception as e:  # noqa: BLE001
            await emit("info", message=f"could not start engine: {e}")

        st.state = "done"
        await emit("done", best=st.best.n_cpu_moe if st.best else None)

    # ---------------------------------------------------------- bench step
    def _bench_once(
        self,
        image: str,
        st: AutotuneState,
        n_cpu_moe: int,
        kv_type: str,
        on_line=None,
    ) -> str:
        """Run llama-bench once in a fresh container, streaming output line-by-line.

        Runs with -r 1 (single repetition) so a slow MoE-on-CPU configuration
        produces a result in minutes instead of tens of minutes. We only need
        a representative tg/pp number per (n_cpu_moe, kv_type) cell to rank them.

        kv_type is passed via -ctk/-ctv so we can compare TurboQuant variants
        (turbo3, turbo4) against the standard q8_0 / q4_0 KV quantizations
        and pick the one that gives the highest tg t/s within the VRAM budget.

        Output is streamed via container.logs(stream=True, follow=True). Each
        complete line is passed to `on_line` (typically wired to the autotune
        WebSocket) so the UI can show what the bench is doing as it happens —
        model load, KV cache alloc, warmup, prompt processing, generation.
        """
        cmd = [
            "llama-bench",
            "-m", f"/models/{st.model_file}",
            "-ngl", "999",
            "--n-cpu-moe", str(n_cpu_moe),
            "-ctk", kv_type,
            "-ctv", kv_type,
            "-fa", "1",
            "-p", str(st.pp),
            "-n", str(st.tg),
            "-r", "1",
        ]
        if st.threads > 0:
            cmd += ["-t", str(st.threads)]

        host_models = self._host_models_dir()
        log.info("autotune step: kv=%s n_cpu_moe=%s", kv_type, n_cpu_moe)
        if on_line:
            on_line(f"[bench] starting llama-bench kv={kv_type} --n-cpu-moe {n_cpu_moe} -p {st.pp} -n {st.tg}")

        container = None
        output_chunks: list[str] = []
        line_buf = ""
        try:
            container = self.client.containers.run(
                image=image,
                command=cmd,
                entrypoint=[""],
                detach=True,
                remove=False,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
                volumes={host_models: {"bind": "/models", "mode": "ro"}},
                stdout=True,
                stderr=True,
            )
            st.current_container_id = container.id

            # Stream stdout+stderr while the bench runs.
            for raw in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                chunk = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
                output_chunks.append(chunk)
                if on_line:
                    line_buf += chunk
                    while "\n" in line_buf:
                        line, line_buf = line_buf.split("\n", 1)
                        if line:
                            on_line(line)
            # Flush any tail without trailing newline
            if on_line and line_buf.strip():
                on_line(line_buf)

            # Drain the exit code; logs() unblocks once the container exits
            # but wait() is what guarantees the inspect is up to date.
            result = container.wait()
            text = "".join(output_chunks)
            if result.get("StatusCode", 0) != 0 and not text.strip():
                text = f"bench exited {result.get('StatusCode')} (no output)"
            if on_line:
                on_line(f"[bench] exit {result.get('StatusCode', '?')}")
            return text
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            st.current_container_id = None

    # ----------------------------------------------------- engine helpers
    def _engine_image(self) -> str:
        try:
            c = self.dm._find_engine()  # noqa: SLF001
            return c.attrs["Config"]["Image"]
        except Exception:
            return "llamador/llama-engine:local"

    def _host_models_dir(self) -> str:
        """Find the host path mounted into the engine as /models.

        The autotune container is launched fresh, so we have to bind the same
        host directory ourselves — Docker doesn't transitively forward bind
        mounts. We read it from the engine's HostConfig.Mounts.
        """
        try:
            c = self.dm._find_engine()  # noqa: SLF001
            for m in c.attrs.get("HostConfig", {}).get("Mounts", []) or []:
                if m.get("Target") == "/models" and m.get("Type") == "bind":
                    return m["Source"]
            # Some compose versions populate 'Mounts' under attrs root instead.
            for m in c.attrs.get("Mounts", []) or []:
                if m.get("Destination") == "/models":
                    return m["Source"]
        except Exception as e:  # noqa: BLE001
            log.warning("could not introspect models mount: %s", e)
        # Fallback to a conventional default.
        return str(Path("/models").as_posix())
