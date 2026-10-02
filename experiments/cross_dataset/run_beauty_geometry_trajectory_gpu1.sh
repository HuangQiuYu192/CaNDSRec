#!/usr/bin/env bash
set -euo pipefail

# P1: training-time mechanism probe.  The two score functions share the
# encoder, data split, optimiser and seed; only final score geometry changes.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-60}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/beauty_geometry_trajectory}"
LOG_ROOT="${LOG_ROOT:-$ROOT/log_runs/beauty_geometry_trajectory_gpu1}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

for variant in dot joint; do
  out="$OUT_ROOT/${variant}_seed${SEED}.csv"
  log="$LOG_ROOT/${variant}_seed${SEED}.log"
  if [ -s "$out" ] && [ "${RERUN:-false}" != true ]; then
    echo "SKIP variant=$variant existing=$out" | tee -a "$LOG_ROOT/driver.log"
    continue
  fi
  echo "START P1 variant=$variant seed=$SEED epochs=$EPOCHS gpu=$GPU_ID" | tee -a "$LOG_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python \
    experiments/cross_dataset/analyze_beauty_geometry_trajectory.py \
    --variant "$variant" --dataset Beauty --gpu_id "$GPU_ID" --seed "$SEED" \
    --epochs "$EPOCHS" --out "$out" > "$log" 2>&1
  tail -1 "$log" | tee -a "$LOG_ROOT/driver.log" || true
done

echo ALL_DONE | tee -a "$LOG_ROOT/driver.log"
