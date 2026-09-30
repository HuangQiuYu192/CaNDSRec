#!/usr/bin/env bash
set -euo pipefail

# Read-only paired rank-transition analysis; GPU 1 only.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEEDS="${SEEDS:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/beauty_tail_rank_transitions}"
OLD_CKPT_ROOT="${OLD_CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
TAU4_CKPT_ROOT="${TAU4_CKPT_ROOT:-$ROOT/ckpt/beauty_sequence_tau4_confirmation_gpu1}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$OUT_ROOT"

checkpoint() { find "$1" -maxdepth 1 -type f -name '*.pth' | head -n 1; }
for seed in $SEEDS; do
  out="$OUT_ROOT/seed${seed}/summary.csv"
  if [ "$RERUN" != true ] && [ -s "$out" ]; then echo "SKIP seed=$seed" | tee -a "$OUT_ROOT/driver.log"; continue; fi
  dot=$(checkpoint "$OLD_CKPT_ROOT/seed${seed}/dot_tau1")
  sequence=$(checkpoint "$TAU4_CKPT_ROOT/seed${seed}/sequence_tau4")
  both=$(checkpoint "$OLD_CKPT_ROOT/seed${seed}/both_tau10")
  if [ -z "$dot" ] || [ -z "$sequence" ] || [ -z "$both" ]; then echo "Missing checkpoint for seed=$seed" >&2; exit 2; fi
  echo "START seed=$seed gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_beauty_tail_rank_transitions.py \
    --seed "$seed" --gpu_id "$GPU_ID" --dot_checkpoint "$dot" --sequence_checkpoint "$sequence" \
    --both_checkpoint "$both" --out_dir "$OUT_ROOT/seed${seed}" > "$OUT_ROOT/seed${seed}.log" 2>&1
  echo "DONE seed=$seed" | tee -a "$OUT_ROOT/driver.log"
done
conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/collect_beauty_tail_rank_transitions.py \
  --input_root "$OUT_ROOT" --out "$OUT_ROOT/summary.csv"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
