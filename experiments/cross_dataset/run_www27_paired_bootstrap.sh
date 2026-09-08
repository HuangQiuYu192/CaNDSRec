#!/usr/bin/env bash
set -euo pipefail

# CPU/GPU-0 inference only; this never trains or selects a model. GPU 1 and
# GPUs 2--3 are not referenced.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/www27_paired_bootstrap}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_matched_seed_grid}"

source "$CONDA_SH"
cd "$ROOT"
for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    out="$OUT_ROOT/$dataset/seed$seed"
    if [ -f "$out.csv" ]; then echo "SKIP $dataset seed=$seed"; continue; fi
    base=$(find "$CKPT_ROOT/$dataset/seed$seed/SASRec" -type f -name '*.pth' | head -n 1)
    cands=$(find "$CKPT_ROOT/$dataset/seed$seed/CANDSSASRec" -type f -name '*.pth' | head -n 1)
    if [ -z "$base" ] || [ -z "$cands" ]; then echo "missing checkpoint $dataset seed=$seed" >&2; exit 2; fi
    conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_www27_paired_bootstrap.py \
      --dataset "$dataset" --seed "$seed" --base_checkpoint "$base" --cands_checkpoint "$cands" \
      --gpu_id 0 --bootstrap_samples "$BOOTSTRAP_SAMPLES" --out "$out"
  done
done
echo ALL_DONE
