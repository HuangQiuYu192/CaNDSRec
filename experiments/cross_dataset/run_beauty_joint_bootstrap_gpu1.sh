#!/usr/bin/env bash
set -euo pipefail

# Read-only uncertainty analysis for the existing three-seed dot vs joint runs.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEEDS="${SEEDS:-2023 2024 2025}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/beauty_joint_bootstrap}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$OUT_ROOT"
for seed in $SEEDS; do
  out="$OUT_ROOT/seed${seed}"
  if [ "$RERUN" != true ] && [ -s "$out.csv" ]; then echo "SKIP seed=$seed"; continue; fi
  dot=$(find "$CKPT_ROOT/seed${seed}/dot_tau1" -maxdepth 1 -type f -name '*.pth' | head -n 1)
  joint=$(find "$CKPT_ROOT/seed${seed}/both_tau10" -maxdepth 1 -type f -name '*.pth' | head -n 1)
  if [ -z "$dot" ] || [ -z "$joint" ]; then echo "Missing checkpoint seed=$seed" >&2; exit 2; fi
  echo "START seed=$seed gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_www27_paired_bootstrap.py \
    --dataset Beauty --seed "$seed" --base_checkpoint "$dot" --cands_checkpoint "$joint" \
    --base_model SASRec --cands_model GeometrySASRec --score_geometry both --sequence_norm_power 1 --item_norm_power 1 \
    --temperature 10 --gpu_id "$GPU_ID" --bootstrap_samples "$BOOTSTRAP_SAMPLES" --out "$out" \
    > "$OUT_ROOT/seed${seed}.log" 2>&1
  echo "DONE seed=$seed" | tee -a "$OUT_ROOT/driver.log"
done
conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/collect_beauty_joint_bootstrap.py \
  --input_root "$OUT_ROOT" --out "$OUT_ROOT/summary.csv"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
