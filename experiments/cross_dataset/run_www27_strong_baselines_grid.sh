#!/usr/bin/env bash
set -euo pipefail

# Strong sequential baselines for the WWW paper. GPU 0 runs GRU4Rec and GPU 1
# runs BERT4Rec; no other GPU is referenced. Jobs are paired by data set and
# seed so each card has at most one training process.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_strong_baselines}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_strong_baselines}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then
  echo "Missing CONDA_SH=$CONDA_SH" >&2
  exit 127
fi
source "$CONDA_SH"
cd "$ROOT"

run_gru() {
  local dataset="$1" seed="$2" log="$OUT_ROOT/$dataset/seed$seed/GRU4Rec.log"
  local ckpt="$CKPT_ROOT/$dataset/seed$seed/GRU4Rec"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GRU4Rec --dataset "$dataset" --gpu_id 0 --seed "$seed" \
    --hidden_size 256 --n_layers 2 --hidden_dropout_prob 0.5 \
    --learning_rate 0.001 --epochs 300 --stopping_step 10 \
    --train_batch_size 1024 --eval_batch_size 512 --max_item_list_length 50 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}

run_bert() {
  local dataset="$1" seed="$2" log="$OUT_ROOT/$dataset/seed$seed/BERT4Rec.log"
  local ckpt="$CKPT_ROOT/$dataset/seed$seed/BERT4Rec"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model BERT4Rec --dataset "$dataset" --gpu_id 1 --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --bert_mask_ratio 0.2 \
    --learning_rate 0.001 --epochs 300 --stopping_step 10 \
    --train_batch_size 512 --eval_batch_size 512 --max_item_list_length 50 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    echo "START dataset=$dataset seed=$seed (GRU4Rec GPU 0; BERT4Rec GPU 1)"
    run_gru "$dataset" "$seed" & pid_gru=$!
    run_bert "$dataset" "$seed" & pid_bert=$!
    wait "$pid_gru"
    wait "$pid_bert"
    echo "DONE dataset=$dataset seed=$seed"
  done
done
echo "ALL_DONE"
