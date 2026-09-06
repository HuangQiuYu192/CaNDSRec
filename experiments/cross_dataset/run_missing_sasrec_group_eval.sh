#!/usr/bin/env bash
set -euo pipefail

# Evaluate missing SASRec head/mid/tail group metrics from existing checkpoints.
# This fills the grouped table for datasets where earlier AngularSmooth scripts
# only evaluated CaNDS and AngularSmooth.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
SEED="${SEED:-2025}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
INNER_SIZE="${INNER_SIZE:-$(( HIDDEN_SIZE * 4 ))}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
CUTOFFS="${CUTOFFS:-5,10,20,50,100}"

BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT/ckpt/main_benchmark_grid}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/sasrec_group_eval}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/sasrec_group_eval}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

latest_checkpoint_in_dir() {
  local ckpt_root="$1"
  local run_name="$2"
  find "$ckpt_root/$run_name" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | sort | tail -n 1
}

group_eval_sasrec() {
  local dataset="$1"
  local ckpt
  local run_name="${dataset}_SASRec_h${HIDDEN_SIZE}_len${MAX_LEN}"
  local tag="${dataset}_SASRec_h${HIDDEN_SIZE}"
  local out_prefix="$OUT_DIR/$tag"
  local log_file="$LOG_DIR/${tag}.log"

  ckpt="$(latest_checkpoint_in_dir "$BASE_CKPT_DIR" "$run_name")"
  if [ -z "$ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING checkpoint for $run_name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START $tag gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --dataset "$dataset" \
    --model SASRec \
    --checkpoint "$ckpt" \
    --tag "$tag" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$INNER_SIZE" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature 10 \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOG_DIR/master.log"
}

for dataset in $DATASETS_STR; do
  group_eval_sasrec "$dataset"
done

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
