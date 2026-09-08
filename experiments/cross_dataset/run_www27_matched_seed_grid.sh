#!/usr/bin/env bash
set -euo pipefail

# Matched-seed endpoint grid. Each pair uses only GPU 0 (SASRec) and GPU 1
# (CaNDS); pairs run sequentially so no other GPU is ever requested.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_matched_seed_grid}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_matched_seed_grid}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
# shellcheck source=/dev/null
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local gpu="$1" model="$2" dataset="$3" seed="$4"
  local run_dir="$OUT_ROOT/$dataset/seed$seed"
  local ckpt_dir="$CKPT_ROOT/$dataset/seed$seed"
  local log="$run_dir/${model}.log"
  mkdir -p "$run_dir" "$ckpt_dir"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
    echo "SKIP completed dataset=$dataset seed=$seed model=$model"
    return 0
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model "$model" --dataset "$dataset" --gpu_id "$gpu" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature 10 --checkpoint_dir "$ckpt_dir" \
    --verbose True --show_progress True > "$log" 2>&1
}

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    echo "START dataset=$dataset seed=$seed (SASRec: GPU 0, CaNDS: GPU 1)"
    run_one 0 SASRec "$dataset" "$seed" & pid_sasrec=$!
    run_one 1 CANDSSASRec "$dataset" "$seed" & pid_cands=$!
    wait "$pid_sasrec"
    wait "$pid_cands"
    echo "DONE dataset=$dataset seed=$seed"
  done
done
echo "ALL_DONE"
