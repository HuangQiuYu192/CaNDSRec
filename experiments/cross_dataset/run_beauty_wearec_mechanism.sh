#!/usr/bin/env bash
set -euo pipefail

# Mechanism diagnostics for Beauty WEARec vs CANDSWEARec.
# It assumes run_beauty_wearec_cands_grid.sh has already produced checkpoints.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPUS_STR="${GPUS_STR:-0 1}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-64 128 256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
N_LAYERS="${N_LAYERS:-2}"
WEAREC_NUM_HEADS="${WEAREC_NUM_HEADS:-1}"
WEAREC_ALPHA="${WEAREC_ALPHA:-0.8}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
MAX_BATCHES="${MAX_BATCHES:-}"

TAG="${TAG:-beauty_wearec_cands_grid}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/beauty_wearec_mechanism}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/beauty_wearec_mechanism}"

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
  local gpu="$1"
  local hidden_size="$2"
  local inner_size
  inner_size="$(inner_size_for_hidden "$hidden_size")"
  local wearec_name="${DATASET}_WEARec_h${hidden_size}_len${MAX_LEN}"
  local cands_name="${DATASET}_CANDSWEARec_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}"
  local wearec_ckpt
  local cands_ckpt
  wearec_ckpt="$(latest_checkpoint "$wearec_name")"
  cands_ckpt="$(latest_checkpoint "$cands_name")"

  if [ -z "$wearec_ckpt" ] || [ -z "$cands_ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING h${hidden_size}: WEARec=$wearec_ckpt CANDS=$cands_ckpt" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  local out_csv="$OUT_DIR/${DATASET}_WEARec_vs_CANDS_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}.csv"
  local log_file="$LOG_DIR/${DATASET}_WEARec_vs_CANDS_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}.log"
  if [ -s "$out_csv" ]; then
    echo "[$(date '+%F %T')] SKIP h${hidden_size}" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START h${hidden_size} gpu=$gpu" | tee -a "$LOG_DIR/master.log"
  extra_args=()
  if [ -n "$MAX_BATCHES" ]; then
    extra_args+=(--max_batches "$MAX_BATCHES")
  fi

  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_wearec_normalization_mechanism.py \
    --dataset "$DATASET" \
    --gpu_id "$gpu" \
    --seed "$SEED" \
    --hidden_size "$hidden_size" \
    --n_layers "$N_LAYERS" \
    --num_heads "$WEAREC_NUM_HEADS" \
    --inner_size "$inner_size" \
    --alpha "$WEAREC_ALPHA" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --wearec_checkpoint "$wearec_ckpt" \
    --cands_checkpoint "$cands_ckpt" \
    --output "$out_csv" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE h${hidden_size}" | tee -a "$LOG_DIR/master.log"
}

worker() {
  local gpu="$1"
  local shard="$2"
  local shards="$3"
  local index=0
  for hidden_size in $HIDDEN_SIZES_STR; do
    if [ $(( index % shards )) -eq "$shard" ]; then
      run_one "$gpu" "$hidden_size"
    fi
    index=$(( index + 1 ))
  done
}

mapfile -t GPUS < <(printf "%s\n" $GPUS_STR)
if [ "${#GPUS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -gt 2 ]; then
  echo "ERROR: GPUS_STR must contain one or two GPUs, e.g. '0 1'." >&2
  exit 1
fi

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] hidden_sizes=$HIDDEN_SIZES_STR gpus=${GPUS[*]}" | tee -a "$LOG_DIR/master.log"

for shard in "${!GPUS[@]}"; do
  worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  echo $! > "$LOG_DIR/worker_gpu${GPUS[$shard]}.pid"
done

wait
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
