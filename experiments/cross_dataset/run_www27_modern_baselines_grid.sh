#!/usr/bin/env bash
set -euo pipefail

# Modern architectural baselines. FMLPRec (frequency filtering) runs on GPU 0
# and BSARec (frequency-aware self-attention) runs on GPU 1. No other GPU is
# addressed. This is a matched-capacity comparison, not a test-set tuning grid.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_modern_baselines}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_modern_baselines}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"

run_fmlp() {
  local dataset="$1" seed="$2" log="$OUT_ROOT/$dataset/seed$seed/FMLPRec.log" ckpt="$CKPT_ROOT/$dataset/seed$seed/FMLPRec"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model FMLPRec --dataset "$dataset" --gpu_id 0 --seed "$seed" \
    --hidden_size 256 --n_layers 2 --inner_size 1024 --hidden_dropout_prob 0.5 \
    --learning_rate 0.001 --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}

run_bsarec() {
  local dataset="$1" seed="$2" log="$OUT_ROOT/$dataset/seed$seed/BSARec.log" ckpt="$CKPT_ROOT/$dataset/seed$seed/BSARec"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model BSARec --dataset "$dataset" --gpu_id 1 --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 \
    --c 4 --alpha 0.5 --learning_rate 0.001 --epochs 300 --stopping_step 10 \
    --train_batch_size 1024 --eval_batch_size 512 --max_item_list_length 50 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    echo "START dataset=$dataset seed=$seed (FMLPRec GPU 0; BSARec GPU 1)"
    run_fmlp "$dataset" "$seed" & pid_fmlp=$!
    run_bsarec "$dataset" "$seed" & pid_bsa=$!
    wait "$pid_fmlp"
    wait "$pid_bsa"
    echo "DONE dataset=$dataset seed=$seed"
  done
done
echo ALL_DONE
