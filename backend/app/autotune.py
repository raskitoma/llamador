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
    tg_tps: float | None = None
    pp_tps: float | None = None
    ok: bool = False
    raw: str = ""
    error: str | None = None


@dataclass
class AutotuneState:
    task_id: str
    model_file: str
    sweep: list[int]
    pp: int = 512
    tg: int = 128
    threads: int = 0
    apply_best: bool = True
    state: str = "queued"  # queued | running | done | error | cancelled
    steps: list[SweepStep] = field(default_factory=list)
    best: SweepStep | None = None
    error: str | None = None


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
        pp: int = 512,
        tg: int = 128,
        threads: int = 0,
        apply_best: bool = True,
        model_file: str | None = None,
    ) -> str:
        cfg = load_config()
        target_model = model_file or cfg.model_file
        sweep_list = list(sweep) if sweep else [48, 40, 36, 32, 28, 24, 20]

        task_id = f"autotune-{int(asyncio.get_event_loop().time())}"
        st = AutotuneState(
            task_id=task_id,
            model_file=target_model,
            sweep=sweep_list,
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

    # ------------------------------------------------------------- runner
    async def _run(self, st: AutotuneState) -> None:
        q = self._streams[st.task_id]

        async def emit(kind: str, **payload):
            await q.put({"type": kind, **payload})

        st.state = "running"
        await emit(
            "start",
            sweep=st.sweep,
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

        # 2) sweep
        loop = asyncio.get_event_loop()
        for n in st.sweep:
            step = SweepStep(n_cpu_moe=n)
            st.steps.append(step)
            await emit("step_start", n_cpu_moe=n)
            try:
                output = await loop.run_in_executor(
                    None, self._bench_once, engine_image, st, n
                )
                step.raw = output
                pp_m = _PP_ROW.search(output)
                tg_m = _TG_ROW.search(output)
                if tg_m:
                    step.tg_tps = float(tg_m.group(1))
                if pp_m:
                    step.pp_tps = float(pp_m.group(1))
                step.ok = step.tg_tps is not None
            except Exception as e:  # noqa: BLE001
                step.error = str(e)
                step.ok = False
            await emit(
                "step_end",
                n_cpu_moe=n,
                ok=step.ok,
                pp_tps=step.pp_tps,
                tg_tps=step.tg_tps,
                error=step.error,
            )

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
            tg_tps=st.best.tg_tps,
            pp_tps=st.best.pp_tps,
        )

        # 4) optionally apply
        if st.apply_best and st.best:
            cfg = load_config()
            cfg.n_cpu_moe = st.best.n_cpu_moe
            save_config(cfg)
            await emit("applied", n_cpu_moe=st.best.n_cpu_moe)

        # 5) restart engine (we stopped it earlier)
        try:
            self.dm.start()
            await emit("info", message="engine started")
        except Exception as e:  # noqa: BLE001
            await emit("info", message=f"could not start engine: {e}")

        st.state = "done"
        await emit("done", best=st.best.n_cpu_moe if st.best else None)

    # ---------------------------------------------------------- bench step
    def _bench_once(self, image: str, st: AutotuneState, n_cpu_moe: int) -> str:
        """Run llama-bench once in a fresh container. Returns combined stdout/stderr."""
        s = self.settings
        cmd = [
            "llama-bench",
            "-m", f"/models/{st.model_file}",
            "-ngl", "999",
            "--n-cpu-moe", str(n_cpu_moe),
            "-fa", "1",
            "-p", str(st.pp),
            "-n", str(st.tg),
        ]
        if st.threads > 0:
            cmd += ["-t", str(st.threads)]

        # Mount the host's models dir into the bench container.
        host_models = self._host_models_dir()
        log.info("autotune step: --n-cpu-moe %s", n_cpu_moe)
        try:
            output = self.client.containers.run(
                image=image,
                command=cmd,
                entrypoint=[""],
                remove=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
                volumes={host_models: {"bind": "/models", "mode": "ro"}},
                stdout=True,
                stderr=True,
            )
            return output.decode(errors="replace") if isinstance(output, bytes) else str(output)
        except docker.errors.ContainerError as e:
            # Non-zero exit → bench failed (most likely OOM). Capture stderr.
            return (e.stderr.decode(errors="replace") if e.stderr else str(e))

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
