#!/usr/bin/env bash
set -euo pipefail

# Recompute frozen diagnostics with explicit scale statistics for every
# completed capacity checkpoint. This is read-only: no checkpoint is trained
# or selected here.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/www27_bsarec_scale_diagnostics}"

checkpoint_for() {
  local seed="$1" dim="$2" model="$3" tag="${model}_d${dim}_a0.5_c9_h2_lr0.001" root
  if [ "$seed" = 2025 ]; then
    root="$ROOT/ckpt/www27_bsarec_dimension_validation/Beauty/seed2025/$tag"
  elif [ "$dim" = 64 ]; then
    root="$ROOT/ckpt/www27_bsarec_geometry_replication/Beauty/seed${seed}/$tag"
  else
    root="$ROOT/ckpt/www27_bsarec_dimension_replication/Beauty/seed${seed}/$tag"
  fi
  find "$root" -maxdepth 1 -type f -name '*.pth' | head -n 1
}

for seed in 2023 2024 2025; do
  for dim in 64 128 256; do
    for model in BSARec CANDSBSARec; do
      out="$OUT_DIR/${model}_d${dim}_seed${seed}.csv"
      test -s "$out" && continue
      ckpt=$(checkpoint_for "$seed" "$dim" "$model")
      test -n "$ckpt"
      echo "START seed=$seed dimension=$dim model=$model"
      conda run --no-capture-output -n recbole python experiments/cross_dataset/analyze_www27_bsarec_dimension_geometry.py \
        --model "$model" --checkpoint "$ckpt" --dimension "$dim" --seed "$seed" --gpu_id "$GPU_ID" --out "$out"
    done
  done
done
echo ALL_DONE
