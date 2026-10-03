#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
CHECKPOINT="${CHECKPOINT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12/seed2025/dot_tau1/SASRec-Sep-28-2026_09-23-14.pth}"
OUT="${OUT:-$ROOT/analysis_results/beauty_frozen_radial_power_sweep/seed2025.csv}"

source "$CONDA_SH"
cd "$ROOT"
if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing CHECKPOINT=$CHECKPOINT" >&2
  exit 2
fi
conda run --no-capture-output -n "$CONDA_ENV" python \
  experiments/cross_dataset/analyze_beauty_frozen_radial_power_sweep.py \
  --checkpoint "$CHECKPOINT" --gpu_id "$GPU_ID" --seed 2025 --out "$OUT"
