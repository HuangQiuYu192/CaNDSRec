#!/usr/bin/env bash
set -euo pipefail

# Read-only checkpoint analysis. Uses GPU 3 only and never trains a model.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-3}"
SEEDS="${SEEDS:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/beauty_score_geometry_gradients}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
MAX_BATCHES="${MAX_BATCHES:-32}"

source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local seed="$1" tag="$2" model="$3" tau="$4" geometry="$5" seq_power="$6" item_power="$7"
  local checkpoint out
  checkpoint=$(find "$CKPT_ROOT/seed${seed}/${tag}_tau${tau}" -maxdepth 1 -type f -name '*.pth' | head -n 1)
  out="$OUT_ROOT/seed${seed}_${tag}_tau${tau}.csv"
  if [ -z "$checkpoint" ]; then echo "Missing checkpoint seed=$seed tag=$tag tau=$tau" >&2; exit 2; fi
  if [ -f "$out" ]; then echo "SKIP $out"; return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python \
    experiments/cross_dataset/analyze_beauty_score_geometry_gradients.py \
    --checkpoint "$checkpoint" --model "$model" --geometry "$geometry" --temperature "$tau" \
    --sequence_norm_power "$seq_power" --item_norm_power "$item_power" --seed "$seed" \
    --gpu_id "$GPU_ID" --max_batches "$MAX_BATCHES" --out "$out"
}

for seed in $SEEDS; do
  run_one "$seed" dot SASRec 1 dot 0 0
  run_one "$seed" sequence GeometrySASRec 5 sequence 1 0
  run_one "$seed" both GeometrySASRec 10 both 1 1
done
# Item-only was a one-seed screening control, not part of the seed replication.
run_one 2025 item GeometrySASRec 10 item 0 1
echo ALL_DONE
