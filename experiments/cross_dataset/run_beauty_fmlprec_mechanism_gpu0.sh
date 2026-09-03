#!/usr/bin/env bash
set -euo pipefail

# Mechanism diagnostics for Beauty FMLPRec vs CANDSFMLPRec.
# It assumes run_beauty_fmlprec_cands_grid_gpu0.sh has already produced checkpoints.
# Default uses only GPU 0.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-64 128 256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
N_LAYERS="${N_LAYERS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
MAX_BATCHES="${MAX_BATCHES:-}"

TAG="${TAG:-beauty_fmlprec_cands_grid_gpu0}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/beauty_fmlprec_mechanism}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/beauty_fmlprec_mechanism_gpu0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

inner_size_for_hidden() {
  echo $(( "$1" * 4 ))
}

latest_checkpoint() {
  local run_name="$1"
  find "$CKPT_DIR/$run_name" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | sort | tail -n 1
}

run_one() {
  local hidden_size="$1"
  local inner_size
  inner_size="$(inner_size_for_hidden "$hidden_size")"
  local fmlprec_name="${DATASET}_FMLPRec_h${hidden_size}_len${MAX_LEN}"
  local cands_name="${DATASET}_CANDSFMLPRec_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}"
  local fmlprec_ckpt
  local cands_ckpt
  fmlprec_ckpt="$(latest_checkpoint "$fmlprec_name")"
  cands_ckpt="$(latest_checkpoint "$cands_name")"

  if [ -z "$fmlprec_ckpt" ] || [ -z "$cands_ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING h${hidden_size}: FMLPRec=$fmlprec_ckpt CANDS=$cands_ckpt" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  local out_csv="$OUT_DIR/${DATASET}_FMLPRec_vs_CANDS_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}.csv"
  local log_file="$LOG_DIR/${DATASET}_FMLPRec_vs_CANDS_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}.log"
  if [ -s "$out_csv" ]; then
    echo "[$(date '+%F %T')] SKIP h${hidden_size}" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START h${hidden_size} gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  extra_args=()
  if [ -n "$MAX_BATCHES" ]; then
    extra_args+=(--max_batches "$MAX_BATCHES")
  fi

  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_fmlprec_normalization_mechanism.py \
    --dataset "$DATASET" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$hidden_size" \
    --n_layers "$N_LAYERS" \
    --inner_size "$inner_size" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --fmlprec_checkpoint "$fmlprec_ckpt" \
    --cands_checkpoint "$cands_ckpt" \
    --output "$out_csv" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE h${hidden_size}" | tee -a "$LOG_DIR/master.log"
}

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] hidden_sizes=$HIDDEN_SIZES_STR gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"

for hidden_size in $HIDDEN_SIZES_STR; do
  run_one "$hidden_size"
done

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
