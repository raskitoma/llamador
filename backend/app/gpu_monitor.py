"""GPU stats — read nvidia-smi from the host without needing the NVIDIA runtime
in the backend container itself.

Strategy:
  1. If the engine container is running, `docker exec` nvidia-smi there.
     Cheap (no new container), and the engine already has GPU passthrough.
  2. Otherwise, spawn an ephemeral container with the engine image
     (which we know has CUDA libs) just to run nvidia-smi once.

The query format uses csv,noheader,nounits so parsing is a single split.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

import docker

from .docker_manager import DockerManager

log = logging.getLogger(__name__)

# Fields we ask nvidia-smi for, in order. Keep aligned with QUERY_FIELDS.
QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
]


def _build_cmd() -> str:
    fields = ",".join(QUERY_FIELDS)
    return f"nvidia-smi --query-gpu={fields} --format=csv,noheader,nounits"


def _parse_row(row: str) -> dict[str, Any]:
    parts = [p.strip() for p in row.split(",")]
    if len(parts) != len(QUERY_FIELDS):
        return {"raw": row}
    out: dict[str, Any] = {}
    for k, v in zip(QUERY_FIELDS, parts):
        # nvidia-smi prints "[Not Supported]" or "N/A" for some fields on older cards.
        if v in ("[Not Supported]", "N/A", ""):
            out[k] = None
            continue
        # numeric coercion
        try:
            out[k] = float(v) if "." in v or k.startswith(("power.", "utilization.")) else int(v)
        except ValueError:
            out[k] = v
    # Friendly extras
    if isinstance(out.get("memory.total"), (int, float)) and isinstance(out.get("memory.used"), (int, float)):
        total = out["memory.total"]
        used = out["memory.used"]
        out["memory.percent"] = round(100.0 * used / total, 1) if total else None
    return out


class GPUMonitor:
    def __init__(self, dm: DockerManager) -> None:
        self.dm = dm
        self.client = docker.from_env()

    def stats(self) -> dict[str, Any]:
        cmd = _build_cmd()

        # ---- path 1: exec into the running engine ----------------------------
        try:
            c = self.dm._find_engine()  # noqa: SLF001 — internal helper
            c.reload()
            if c.attrs.get("State", {}).get("Running"):
                exit_code, output = c.exec_run(cmd)
                if exit_code == 0:
                    text = output.decode(errors="replace").strip()
                    return self._format(text, source="engine-exec")
                log.warning("nvidia-smi via exec_run rc=%s out=%s", exit_code, output[:200])
        except Exception as e:  # noqa: BLE001
            log.info("engine exec_run not available: %s", e)

        # ---- path 2: ephemeral container with the engine image ---------------
        # We use the engine image because it's guaranteed to be present on the
        # host once the user has built the stack, and it has the CUDA libs.
        engine_image = self._engine_image_name()
        try:
            output = self.client.containers.run(
                image=engine_image,
                command=shlex.split(cmd),
                entrypoint=[""],   # bypass the engine entrypoint
                remove=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
                stdout=True,
                stderr=True,
            )
            text = output.decode(errors="replace").strip() if isinstance(output, bytes) else str(output)
            return self._format(text, source="ephemeral")
        except Exception as e:  # noqa: BLE001
            log.warning("ephemeral nvidia-smi failed: %s", e)
            return {"available": False, "error": str(e)}

    # ----------------------------------------------------------------- helpers
    def _engine_image_name(self) -> str:
        """Best guess at the engine's image name. Falls back to compose convention."""
        try:
            c = self.dm._find_engine()  # noqa: SLF001
            return c.attrs["Config"]["Image"]
        except Exception:
            return "llamador/llama-engine:local"

    def _format(self, text: str, source: str) -> dict[str, Any]:
        if not text.strip():
            return {"available": False, "error": "empty output", "source": source}
        gpus = [_parse_row(line) for line in text.splitlines() if line.strip()]
        return {
            "available": True,
            "source": source,
            "count": len(gpus),
            "gpus": gpus,
        }
