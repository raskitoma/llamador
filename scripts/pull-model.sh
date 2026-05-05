#!/usr/bin/env bash
# pull-model.sh — fetch a GGUF from Hugging Face into ./data/models
# Usage:
#   scripts/pull-model.sh <repo> <filename>
#   scripts/pull-model.sh unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf
set -euo pipefail

REPO="${1:-unsloth/Qwen3.6-35B-A3B-GGUF}"
FILE="${2:-Qwen3.6-35B-A3B-Q4_K_M.gguf}"
OUT="${MODELS_DIR:-./data/models}"

mkdir -p "$OUT"

# Prefer the running backend container so we don't need Python on the host.
if docker compose ps --status running --services 2>/dev/null | grep -q '^backend$'; then
  echo "[pull] using running backend container"
  docker compose exec -T backend python -c "
import os
from huggingface_hub import hf_hub_download
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
p = hf_hub_download(repo_id='$REPO', filename='$FILE',
                    local_dir='/models', local_dir_use_symlinks=False,
                    token=os.environ.get('HF_TOKEN') or None)
print('downloaded to', p)
"
  exit 0
fi

# Fallback: host-side download — needs python3 + huggingface_hub.
echo "[pull] backend not up; falling back to host python"
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
python3 -m pip install --user --quiet --upgrade "huggingface_hub[cli]" hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -m huggingface_hub.commands.huggingface_cli download \
    "$REPO" "$FILE" --local-dir "$OUT" --local-dir-use-symlinks False
ls -lh "$OUT/$FILE"
