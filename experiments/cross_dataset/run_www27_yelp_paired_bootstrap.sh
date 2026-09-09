#!/usr/bin/env bash
set -euo pipefail

# Paired user-level uncertainty for the external Yelp geometry comparison.
# This loads completed checkpoints only; it does not train or tune a model.
# Only GPUs 0 and 1 are used.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_geometry_ablation/Yelp-S3Rec}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/www27_yelp_paired_bootstrap}"

source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local gpu="$1" seed="$2" out="$OUT_ROOT/seed${seed}"
  local base cands
  base=$(find "$CKPT_ROOT/seed${seed}/dot" -maxdepth 1 -type f -name 'SASRec-*.pth' | head -n 1)
  cands=$(find "$CKPT_ROOT/seed${seed}/both" -maxdepth 1 -type f -name 'GeometrySASRec-*.pth' | head -n 1)
  if [ -z "$base" ] || [ -z "$cands" ]; then
    echo "Missing checkpoint for Yelp seed=$seed" >&2; return 2
  fi
  if [ -s "$out.csv" ]; then echo "SKIP Yelp seed=$seed"; return 0; fi
  mkdir -p "$OUT_ROOT"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_www27_paired_bootstrap.py \
    --dataset Yelp-S3Rec --seed "$seed" --base_model SASRec --cands_model GeometrySASRec \
    --base_checkpoint "$base" --cands_checkpoint "$cands" --gpu_id "$gpu" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 --max_item_list_length 50 \
    --train_batch_size 1024 --eval_batch_size 512 --temperature 10 --bootstrap_samples "$BOOTSTRAP_SAMPLES" --out "$out"
}

seeds=($SEEDS_STR)
for ((idx=0; idx<${#seeds[@]}; idx+=2)); do
  run_one 0 "${seeds[$idx]}" & p0=$!
  if [ $((idx + 1)) -lt ${#seeds[@]} ]; then run_one 1 "${seeds[$((idx + 1))]}" & p1=$!; wait "$p0"; wait "$p1"; else wait "$p0"; fi
done
echo ALL_DONE
