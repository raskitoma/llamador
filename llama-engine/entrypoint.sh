#!/usr/bin/env bash
# entrypoint.sh — translate runtime config into llama-server flags.
#
# Config sources, in order of precedence (later overrides earlier):
#   1. Defaults baked into the Dockerfile ENV
#   2. /config/engine.json — written by the backend control plane (if mounted)
#   3. Environment variables passed by docker-compose (.env)
#
# The JSON file lets the UI mutate parameters without recreating the container:
# the backend writes engine.json + restarts the container, and we re-read here.

set -euo pipefail

CONFIG_JSON="${CONFIG_JSON:-/config/engine.json}"

# ---------- 1. read JSON config if present --------------------------------
if [[ -f "${CONFIG_JSON}" ]]; then
  echo "[engine] applying config from ${CONFIG_JSON}"
  # Tiny inline JSON reader — avoids dragging jq into the runtime image.
  # Pulls one key at a time and exports it as the matching env var.
  read_json() {
    python3 -c "
import json, sys
try:
    d = json.load(open('${CONFIG_JSON}'))
    v = d.get('$1')
    if v is None: sys.exit(0)
    if isinstance(v, bool): print('on' if v else 'off')
    else: print(v)
except Exception as e:
    print('', file=sys.stderr); sys.exit(0)
" 2>/dev/null
  }
  # Map JSON keys → engine env vars. Empty results keep the existing value.
  for k in model_file alias n_cpu_moe ctx kv_type flash_attn threads batch ubatch extra_args; do
    v="$(read_json "$k" || true)"
    if [[ -n "$v" ]]; then
      case "$k" in
        model_file) MODEL_FILE_NAME="$v" ;;
        alias)      ALIAS="$v" ;;
        n_cpu_moe)  N_CPU_MOE="$v" ;;
        ctx)        CTX="$v" ;;
        kv_type)    KV_TYPE="$v" ;;
        flash_attn) FLASH_ATTN="$v" ;;
        threads)    THREADS="$v" ;;
        batch)      BATCH="$v" ;;
        ubatch)     UBATCH="$v" ;;
        extra_args) EXTRA_ARGS="$v" ;;
      esac
    fi
  done

  # If the JSON only gave us a filename, prefix with /models.
  if [[ -n "${MODEL_FILE_NAME:-}" ]]; then
    if [[ "${MODEL_FILE_NAME}" == /* ]]; then
      MODEL_FILE="${MODEL_FILE_NAME}"
    else
      MODEL_FILE="/models/${MODEL_FILE_NAME}"
    fi
  fi
fi

# ---------- 2. resolve auto values ----------------------------------------
if [[ "${THREADS}" == "auto" || "${THREADS}" == "0" ]]; then
  THREADS="$(nproc)"
fi

# ---------- 3. preflight ---------------------------------------------------
if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "[engine] FATAL: model not found: ${MODEL_FILE}" >&2
  echo "[engine] mount your models dir at /models and set MODEL_FILE accordingly." >&2
  echo "[engine] available files in /models:" >&2
  ls -lh /models 2>&1 | sed 's/^/  /' >&2 || true
  exit 2
fi

echo "[engine] booting TurboQuant llama-server"
echo "[engine] commit:   $(cat /etc/turboquant.sha 2>/dev/null || echo unknown)"
echo "[engine] model:    ${MODEL_FILE}"
echo "[engine] knobs:    ngl=${NGL} n_cpu_moe=${N_CPU_MOE} ctx=${CTX} kv=${KV_TYPE} fa=${FLASH_ATTN} threads=${THREADS}"
echo "[engine] extra:    ${EXTRA_ARGS:-(none)}"

# ---------- 4. exec --------------------------------------------------------
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
exec llama-server \
  --model        "${MODEL_FILE}" \
  --alias        "${ALIAS}" \
  --host         "${HOST}" \
  --port         "${PORT}" \
  -ngl           "${NGL}" \
  --n-cpu-moe    "${N_CPU_MOE}" \
  -c             "${CTX}" \
  -fa            "${FLASH_ATTN}" \
  --cache-type-k "${KV_TYPE}" \
  --cache-type-v "${KV_TYPE}" \
  -t             "${THREADS}" \
  -b             "${BATCH}" \
  -ub            "${UBATCH}" \
  --mlock \
  --metrics \
  ${EXTRA_ARGS}
