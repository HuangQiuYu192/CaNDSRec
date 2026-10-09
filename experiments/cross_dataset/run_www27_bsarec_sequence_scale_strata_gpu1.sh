#!/usr/bin/env bash
set -euo pipefail

# Read-only quintile diagnostics for every already-completed d=256 checkpoint.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/www27_bsarec_sequence_scale_strata}"

checkpoint_for() {
  local seed="$1" variant="$2" tag root
  case "$variant" in
    dot|joint)
      if [ "$variant" = dot ]; then tag="BSARec_d256_a0.5_c9_h2_lr0.001"; else tag="CANDSBSARec_d256_a0.5_c9_h2_lr0.001"; fi
      if [ "$seed" = 2025 ]; then root="$ROOT/ckpt/www27_bsarec_dimension_validation/Beauty/seed2025/$tag"; else root="$ROOT/ckpt/www27_bsarec_dimension_replication/Beauty/seed${seed}/$tag"; fi
      ;;
    scaled_dot) tag="ScaledDotBSARec_d256_tau0.04_a0.5_c9_h2_lr0.001"; root="$ROOT/ckpt/www27_bsarec_scaled_dot_confirmation/Beauty/seed${seed}/$tag" ;;
    sequence) tag="GeometryBSARec_sequence_d256_tau4_a0.5_c9_h2_lr0.001"; root="$ROOT/ckpt/www27_bsarec_sequence_mechanism_confirmation/Beauty/seed${seed}/$tag" ;;
    stopgrad_sequence) tag="StopGradSequenceGeometryBSARec_sequence_d256_tau4_a0.5_c9_h2_lr0.001"; root="$ROOT/ckpt/www27_bsarec_sequence_mechanism_confirmation/Beauty/seed${seed}/$tag" ;;
  esac
  find "$root" -maxdepth 1 -type f -name '*.pth' | head -n 1
}

model_for() { case "$1" in dot) echo BSARec;; scaled_dot) echo ScaledDotBSARec;; sequence) echo GeometryBSARec;; stopgrad_sequence) echo StopGradSequenceGeometryBSARec;; joint) echo CANDSBSARec;; esac; }
temp_for() { case "$1" in dot) echo 1;; scaled_dot) echo 0.04;; sequence|stopgrad_sequence) echo 4;; joint) echo 10;; esac; }

for seed in 2023 2024 2025; do
  for variant in dot scaled_dot sequence stopgrad_sequence joint; do
    out="$OUT_DIR/${variant}_seed${seed}.csv"
    test -s "$out" && continue
    ckpt="$(checkpoint_for "$seed" "$variant")"
    test -n "$ckpt"
    echo "START seed=$seed variant=$variant"
    conda run --no-capture-output -n recbole python experiments/cross_dataset/analyze_www27_bsarec_sequence_scale_strata.py \
      --model "$(model_for "$variant")" --variant "$variant" --temperature "$(temp_for "$variant")" \
      --checkpoint "$ckpt" --seed "$seed" --gpu_id "$GPU_ID" --out "$out"
  done
done
conda run --no-capture-output -n recbole python experiments/cross_dataset/collect_www27_bsarec_sequence_scale_strata.py
echo ALL_DONE
