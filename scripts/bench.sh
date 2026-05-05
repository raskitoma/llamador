#!/usr/bin/env bash
# bench.sh — sweep --n-cpu-moe with llama-bench inside the engine container
# Usage: scripts/bench.sh [n_cpu_moe values...]
#   scripts/bench.sh                  # 24 28 32 36
#   scripts/bench.sh 20 24 28 32 36
set -euo pipefail
SWEEP=("$@")
[[ ${#SWEEP[@]} -eq 0 ]] && SWEEP=(24 28 32 36)

MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-Q4_K_M.gguf}"

for n in "${SWEEP[@]}"; do
  echo "============================================================"
  echo "  --n-cpu-moe $n"
  echo "============================================================"
  docker compose exec -T llama-engine \
    llama-bench -m "/models/$MODEL_FILE" -ngl 999 --n-cpu-moe "$n" \
                -fa 1 -p 512 -n 128 -t "$(nproc)" \
    || echo "(failed at n=$n — likely OOM)"
done
