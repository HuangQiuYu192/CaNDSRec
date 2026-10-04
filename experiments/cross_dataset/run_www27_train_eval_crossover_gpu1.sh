#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"; source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"; cd "$ROOT"
for d in ${DATASETS_STR:-Beauty Sports}; do conda run --no-capture-output -n recbole python experiments/cross_dataset/analyze_www27_train_eval_crossover.py --dataset "$d" --checkpoint_root ckpt/www27_gradient_causality_gpu1 --out "analysis_results/www27_train_eval_crossover/${d}.csv" --gpu_id 1; done
echo ALL_DONE
