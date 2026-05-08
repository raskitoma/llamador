"""Settings + persisted runtime config.

Two layers:
- Settings (env-only): paths and constants the operator sets at deploy time.
- RuntimeConfig (JSON file): the engine knobs the UI mutates at runtime. We
  persist these in /config/engine.json so a restart preserves them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where the llama-engine container can be addressed (compose service DNS).
    engine_url: str = "http://llama-engine:8080"

    # Compose service name + project, used by the docker SDK to find the container.
    engine_service: str = "llama-engine"
    compose_project: str = "llamador"

    # Persistent storage on the host, mounted into this container.
    models_dir: Path = Path("/models")
    config_dir: Path = Path("/config")

    # HF token for gated repos (Qwen3.6 isn't gated today, but be ready).
    hf_token: str | None = None

    # Optional bearer token to gate the API. Empty = open (LAN trust).
    api_token: str | None = None


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.config_dir.mkdir(parents=True, exist_ok=True)
        _settings.models_dir.mkdir(parents=True, exist_ok=True)
    return _settings


# ---------------------------------------------------------------------------
# RuntimeConfig — the JSON file the UI writes to.
# ---------------------------------------------------------------------------

KVType = Literal["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "turbo3", "turbo4"]
FlashAttn = Literal["on", "off"]


class RuntimeConfig(BaseModel):
    """The set of engine knobs we expose to the UI."""

    model_file: str = Field("Qwen3.6-35B-A3B-Q4_K_M.gguf", description="Filename inside /models")
    alias: str = "qwen3.6-35b-a3b"
    n_cpu_moe: int = Field(30, ge=0, le=64, description="MoE layers offloaded to CPU RAM")
    ctx: int = Field(8192, ge=512, le=262144)
    kv_type: KVType = "q8_0"
    flash_attn: FlashAttn = "on"
    threads: int = Field(0, ge=0, le=256, description="0 = auto (use all cores)")
    batch: int = 2048
    ubatch: int = 2048
    no_mmap: bool = Field(
        False,
        description="Disable mmap and load the whole model into RAM up-front. "
                    "Combined with --mlock this pins the weights resident — recommended "
                    "for low-VRAM / heavy-CPU-offload setups where lazy mmap paging "
                    "causes stalls during inference.",
    )
    extra_args: str = ""


_lock = Lock()


def _config_path() -> Path:
    return get_settings().config_dir / "engine.json"


def load_config() -> RuntimeConfig:
    path = _config_path()
    if not path.exists():
        cfg = RuntimeConfig()
        save_config(cfg)
        return cfg
    return RuntimeConfig.model_validate_json(path.read_text())


def save_config(cfg: RuntimeConfig) -> None:
    with _lock:
        tmp = _config_path().with_suffix(".tmp")
        tmp.write_text(cfg.model_dump_json(indent=2))
        os.replace(tmp, _config_path())


def to_engine_env(cfg: RuntimeConfig) -> dict[str, str]:
    """Translate RuntimeConfig into the env vars the engine entrypoint reads."""
    s = get_settings()
    return {
        "MODEL_FILE": str((s.models_dir / cfg.model_file).as_posix())
        if not cfg.model_file.startswith("/")
        else cfg.model_file,
        "ALIAS": cfg.alias,
        "N_CPU_MOE": str(cfg.n_cpu_moe),
        "CTX": str(cfg.ctx),
        "KV_TYPE": cfg.kv_type,
        "FLASH_ATTN": cfg.flash_attn,
        "THREADS": str(cfg.threads) if cfg.threads > 0 else "auto",
        "BATCH": str(cfg.batch),
        "UBATCH": str(cfg.ubatch),
        "NO_MMAP": "on" if cfg.no_mmap else "off",
        "EXTRA_ARGS": cfg.extra_args,
    }
