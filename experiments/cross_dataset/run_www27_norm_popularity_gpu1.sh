#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
OUT="analysis_results/www27_norm_popularity"
mkdir -p "$OUT"
for d in ${DATASETS_STR:-Beauty Sports}; do
  for s in ${SEEDS_STR:-2023 2024 2025}; do
    for v in dot joint; do
      f="$OUT/${d}_seed${s}_${v}.csv"
      [ -s "$f" ] && continue
      conda run --no-capture-output -n recbole python experiments/cross_dataset/analyze_www27_norm_popularity.py \
        --dataset "$d" --seed "$s" --variant "$v" --checkpoint_root ckpt/www27_gradient_causality_gpu1 \
        --out "$f" --gpu_id 1
    done
  done
done
conda run --no-capture-output -n recbole python experiments/cross_dataset/collect_www27_norm_popularity.py \
  --input_dir "$OUT" --out "$OUT/summary.csv"
echo ALL_DONE
