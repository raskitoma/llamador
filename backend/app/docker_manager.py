"""Wrapper around docker SDK for managing the llama-engine container.

The control loop is intentionally boring:
  1. Backend writes /config/engine.json (a shared volume).
  2. Backend calls container.restart() on the engine.
  3. Engine entrypoint re-reads the JSON and exec's llama-server with new flags.

We never recreate the container from inspect data — that's brittle. For
image rebuilds (binary updates) we invoke docker.images.build(), tag the
result, then restart the engine container so it pulls the new image.
"""

from __future__ import annotations

import logging
from typing import Iterator

import docker
from docker.errors import NotFound
from docker.models.containers import Container

from .config import get_settings

log = logging.getLogger(__name__)


class EngineNotFound(RuntimeError):
    """Raised when the llama-engine container can't be located."""


class DockerManager:
    def __init__(self) -> None:
        self.client = docker.from_env()
        self.settings = get_settings()

    # ---------------------------------------------------------------- find
    def _find_engine(self) -> Container:
        # Compose labels are the most reliable way to find the right container
        # — survives renames and is unambiguous when multiple stacks coexist.
        # Note: Compose v5.x bakes project/service labels into the IMAGE, so
        # ephemeral `docker run` containers (e.g. gpu_monitor's nvidia-smi
        # one-off) inherit them too. `container-number` is only set on real
        # compose-managed containers, so requiring it filters those out.
        filters = {
            "label": [
                f"com.docker.compose.project={self.settings.compose_project}",
                f"com.docker.compose.service={self.settings.engine_service}",
                "com.docker.compose.container-number",
            ]
        }
        containers = self.client.containers.list(all=True, filters=filters)
        if containers:
            return containers[0]
        try:
            return self.client.containers.get(self.settings.engine_service)
        except NotFound as e:
            raise EngineNotFound(
                f"no container with compose service '{self.settings.engine_service}' "
                f"in project '{self.settings.compose_project}'"
            ) from e

    # ---------------------------------------------------------------- read
    def status(self) -> dict:
        try:
            c = self._find_engine()
        except EngineNotFound as e:
            return {"present": False, "error": str(e)}

        c.reload()
        state = c.attrs.get("State", {})
        cfg = c.attrs.get("Config", {})
        env = {
            k: v
            for k, v in (e.split("=", 1) for e in cfg.get("Env", []) if "=" in e)
            if not k.startswith(("PATH", "LD_", "NV_", "NVIDIA_", "CUDA_"))
        }
        return {
            "present": True,
            "id": c.short_id,
            "name": c.name,
            "image": cfg.get("Image"),
            "status": state.get("Status"),
            "running": bool(state.get("Running")),
            "started_at": state.get("StartedAt"),
            "health": (state.get("Health") or {}).get("Status"),
            "env": env,
        }

    def logs(self, tail: int = 200) -> str:
        c = self._find_engine()
        return c.logs(tail=tail).decode(errors="replace")

    def stream_logs(self) -> Iterator[bytes]:
        """Yield log chunks as bytes — used by the WebSocket endpoint."""
        c = self._find_engine()
        return c.logs(stream=True, follow=True, tail=50)

    # ----------------------------------------------------------- mutations
    def restart(self) -> None:
        c = self._find_engine()
        log.info("restarting engine container %s", c.name)
        c.restart(timeout=30)

    def stop(self) -> None:
        c = self._find_engine()
        log.info("stopping engine container %s", c.name)
        c.stop(timeout=30)

    def start(self) -> None:
        c = self._find_engine()
        log.info("starting engine container %s", c.name)
        c.start()

    # --------------------------------------------------------- image build
    def rebuild_image(self, build_args: dict[str, str] | None = None) -> Iterator[dict]:
        """Stream build output as we rebuild the engine image.

        Caller is responsible for tagging the engine container's image, then
        calling restart() afterwards. We yield raw build chunks so the
        WebSocket can forward them to the UI.

        Note: the build context lives at /workspace/llama-engine inside this
        container (mounted from the host repo).
        """
        c = self._find_engine()
        image_name = c.attrs["Config"]["Image"]
        log.info("rebuilding image %s", image_name)
        for chunk in self.client.api.build(
            path="/workspace/llama-engine",
            tag=image_name,
            buildargs=build_args or {},
            rm=True,
            decode=True,
            nocache=False,
            pull=True,
        ):
            yield chunk
