"""API key store for the inference (`/v1/*`) endpoints.

Two-tier auth:
  - control plane (`/api/*`)  → gated by Settings.api_token (single value, env-only)
  - inference   (`/v1/*`)     → gated by this KeyStore (multiple revocable keys)

Persistence: a single JSON file in /config so keys survive restarts and a
host reboot. Format:

  {
    "keys": [
      {
        "id": "kid_5jx…",      # short id surfaced to the UI
        "name": "openwebui-prod",
        "key": "sk-ABCDEF…",   # the secret — full value
        "created_at": 1714914812.3,
        "last_used_at": null,
        "enabled": true
      },
      …
    ]
  }

Behaviour:
  - If the store has *no enabled keys*, /v1 is open (LAN-trust mode).
    The moment you create one, the gate snaps shut.
  - `verify(token)` returns the matching record or None and updates
    last_used_at as a side-effect.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from threading import Lock

from .config import get_settings

log = logging.getLogger(__name__)

KEY_PREFIX = "sk-"
KEY_LEN = 32      # alphanum after the prefix


def _new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(KEY_LEN)[:KEY_LEN]


def _new_id() -> str:
    return "kid_" + secrets.token_hex(6)


class KeyStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.path = self.settings.config_dir / "keys.json"
        self._lock = Lock()
        self._cache: dict | None = None

    # ----------------------------------------------------------------- io
    def _read(self) -> dict:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            self._cache = {"keys": []}
            return self._cache
        try:
            self._cache = json.loads(self.path.read_text())
            self._cache.setdefault("keys", [])
        except json.JSONDecodeError as e:
            log.error("keys.json malformed (%s) — starting fresh", e)
            self._cache = {"keys": []}
        return self._cache

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)
        self._cache = data

    # ------------------------------------------------------------- public
    def list(self, include_secrets: bool = False) -> list[dict]:
        data = self._read()
        out = []
        for k in data["keys"]:
            row = dict(k)
            if not include_secrets:
                # show first/last 4 chars only — avoid splattering secrets in logs
                full = row.pop("key", "")
                row["key_preview"] = (full[:6] + "…" + full[-4:]) if len(full) >= 12 else "…"
            out.append(row)
        return out

    def create(self, name: str) -> dict:
        with self._lock:
            data = self._read()
            rec = {
                "id": _new_id(),
                "name": name or "unnamed",
                "key": _new_key(),
                "created_at": time.time(),
                "last_used_at": None,
                "enabled": True,
            }
            data["keys"].append(rec)
            self._write(data)
        log.info("created API key %s name=%s", rec["id"], rec["name"])
        return rec

    def revoke(self, kid: str) -> bool:
        with self._lock:
            data = self._read()
            before = len(data["keys"])
            data["keys"] = [k for k in data["keys"] if k.get("id") != kid]
            removed = len(data["keys"]) < before
            if removed:
                self._write(data)
                log.info("revoked API key %s", kid)
        return removed

    def set_enabled(self, kid: str, enabled: bool) -> bool:
        with self._lock:
            data = self._read()
            for k in data["keys"]:
                if k.get("id") == kid:
                    k["enabled"] = bool(enabled)
                    self._write(data)
                    return True
        return False

    def has_keys(self) -> bool:
        return any(k.get("enabled") for k in self._read()["keys"])

    def verify(self, presented: str | None) -> dict | None:
        """Return the record if the presented key matches an enabled one.

        `presented` may be the raw key (e.g. from `Authorization: Bearer …`)
        or None (returns None). On match, last_used_at is updated.
        """
        if not presented:
            return None
        with self._lock:
            data = self._read()
            for k in data["keys"]:
                if k.get("enabled") and secrets.compare_digest(k.get("key", ""), presented):
                    k["last_used_at"] = time.time()
                    self._write(data)
                    return k
        return None
