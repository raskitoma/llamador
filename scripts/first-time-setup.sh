#!/usr/bin/env bash
# first-time-setup.sh — one-shot bootstrap for a fresh Nevermind clone.
# Idempotent: re-running is safe.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[setup] preflight"
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker compose version >/dev/null || { echo "docker compose plugin not found"; exit 1; }

echo "[setup] checking nvidia container toolkit"
if ! docker info 2>/dev/null | grep -qi 'nvidia'; then
  cat <<'EOF'
[setup] WARNING: NVIDIA container runtime not detected.
        Install it on Ubuntu:
          curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
          curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
          sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
          sudo nvidia-ctk runtime configure --runtime=docker
          sudo systemctl restart docker
EOF
fi

echo "[setup] writing .env if missing"
[[ -f .env ]] || cp .env.example .env

echo "[setup] creating data dirs"
mkdir -p data/models data/config

echo "[setup] checking GPU access from a test container"
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
  echo "[setup] WARNING: 'docker run --gpus all' did not return a GPU."
  echo "[setup] Check 'sudo systemctl status docker' and 'nvidia-ctk runtime configure'."
fi

echo "[setup] building images (this takes a while — engine compiles llama.cpp + CUDA)"
docker compose build

echo
echo "[setup] done. Next:"
echo "  1) ./scripts/pull-model.sh                            # download default Qwen3.6 GGUF"
echo "  2) docker compose up -d                                # start the stack"
echo "  3) open http://<this-host>:\${CADDY_PORT:-8088}/         # control panel"
