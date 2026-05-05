"""Manage GGUF files in the shared /models volume.

We keep the operations narrow on purpose — list, pull, delete — because the
backend doesn't need to understand model internals.
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import Future
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from .config import get_settings

log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._tasks: dict[str, dict] = {}
        self._lock = Lock()

    # ------------------------------------------------------------- listing
    def list_models(self) -> list[dict]:
        out = []
        for p in sorted(self.settings.models_dir.glob("*.gguf")):
            stat = p.stat()
            out.append(
                {
                    "name": p.name,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
        return out

    # -------------------------------------------------------------- delete
    def delete(self, name: str) -> None:
        # Resolve and confirm the file is still inside /models — refuse traversal.
        target = (self.settings.models_dir / name).resolve()
        if self.settings.models_dir.resolve() not in target.parents:
            raise PermissionError(f"refusing to delete outside models dir: {target}")
        if not target.is_file():
            raise FileNotFoundError(name)
        log.info("deleting %s", target)
        target.unlink()

    # ---------------------------------------------------------------- pull
    def start_pull(self, repo: str, filename: str) -> str:
        """Kick off an HF download in a background thread, return a task id."""
        task_id = f"pull-{abs(hash((repo, filename)))}"
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["state"] == "running":
                return task_id  # idempotent
            self._tasks[task_id] = {
                "state": "running",
                "repo": repo,
                "file": filename,
                "progress": None,
                "error": None,
            }

        # Make HF transfer fast where possible.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from threading import Thread

        def _run():
            try:
                local_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(self.settings.models_dir),
                    local_dir_use_symlinks=False,
                    token=self.settings.hf_token,
                )
                # If HF cached to a subpath, move/copy into /models root for the
                # engine to find by name (the engine prepends /models/ to filenames).
                final = self.settings.models_dir / filename
                if Path(local_path) != final:
                    shutil.move(local_path, final)
                with self._lock:
                    self._tasks[task_id]["state"] = "done"
                    self._tasks[task_id]["path"] = str(final)
            except HfHubHTTPError as e:
                with self._lock:
                    self._tasks[task_id]["state"] = "error"
                    self._tasks[task_id]["error"] = f"hf: {e}"
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._tasks[task_id]["state"] = "error"
                    self._tasks[task_id]["error"] = str(e)

        Thread(target=_run, daemon=True, name=task_id).start()
        return task_id

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [{"id": k, **v} for k, v in self._tasks.items()]
