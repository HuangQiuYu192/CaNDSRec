#!/usr/bin/env bash
set -euo pipefail

# Main benchmark grid for CaNDSRec.
# It runs SASRec and CANDSSASRec with hidden sizes 64/256 by default, and
# assigns jobs only to GPU 0 and GPU 1.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
SEED="${SEED:-2025}"

DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec ML-1M LastFM-S3Rec}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-64 256}"
TEMPS_STR="${TEMPS_STR:-2 5 7.5 10 15 20 30 40}"
GPUS_STR="${GPUS_STR:-0 1}"

DEFAULT_MAX_LEN="${DEFAULT_MAX_LEN:-50}"
ML1M_MAX_LEN="${ML1M_MAX_LEN:-50}"
LASTFM_MAX_LEN="${LASTFM_MAX_LEN:-200}"

EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

TAG="${TAG:-main_benchmark_grid}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
TASK_FILE="$LOG_DIR/tasks.tsv"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
  echo "Set CONDA_SH=/path/to/conda.sh if conda is not already available." | tee -a "$LOG_DIR/master.log"
fi

max_len_for_dataset() {
  case "$1" in
    ML-1M) echo "$ML1M_MAX_LEN" ;;
    LastFM-S3Rec) echo "$LASTFM_MAX_LEN" ;;
    *) echo "$DEFAULT_MAX_LEN" ;;
  esac
}

inner_size_for_hidden() {
  echo $(( "$1" * 4 ))
}

write_task() {
  local dataset="$1"
  local hidden_size="$2"
  local model="$3"
  local temperature="$4"
  local max_len="$5"
  local inner_size="$6"
  local name="${dataset}_${model}_h${hidden_size}_len${max_len}"
  if [ "$model" = "CANDSSASRec" ]; then
    name="${name}_temp${temperature}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$dataset" "$hidden_size" "$model" "$temperature" "$max_len" "$inner_size" >> "$TASK_FILE"
}

build_tasks() {
  : > "$TASK_FILE"
  for hidden_size in $HIDDEN_SIZES_STR; do
    inner_size="$(inner_size_for_hidden "$hidden_size")"
    for dataset in $DATASETS_STR; do
      max_len="$(max_len_for_dataset "$dataset")"
      write_task "$dataset" "$hidden_size" SASRec 10.0 "$max_len" "$inner_size"
      for temp in $TEMPS_STR; do
        write_task "$dataset" "$hidden_size" CANDSSASRec "$temp" "$max_len" "$inner_size"
      done
    done
  done
}

run_task() {
  local gpu="$1"
  local name="$2"
  local dataset="$3"
  local hidden_size="$4"
  local model="$5"
  local temperature="$6"
  local max_len="$7"
  local inner_size="$8"
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/${name}"

  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP $name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  mkdir -p "$ckpt_path"
  echo "[$(date '+%F %T')] START $name gpu=$gpu" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --dataset "$dataset" \
    --model "$model" \
    --gpu_id "$gpu" \
    --seed "$SEED" \
    --hidden_size "$hidden_size" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$inner_size" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --stopping_step "$STOPPING_STEP" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_item_list_length "$max_len" \
    --temperature "$temperature" \
    --checkpoint_dir "$ckpt_path" \
    --verbose True \
    --show_progress True \
    > "$log_file" 2>&1
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$LOG_DIR/master.log"
}

worker() {
  local gpu="$1"
  local shard="$2"
  local shards="$3"
  local index=0

  while IFS=$'\t' read -r name dataset hidden_size model temperature max_len inner_size; do
    if [ $(( index % shards )) -eq "$shard" ]; then
      run_task "$gpu" "$name" "$dataset" "$hidden_size" "$model" "$temperature" "$max_len" "$inner_size"
    fi
    index=$(( index + 1 ))
  done < "$TASK_FILE"
}

build_tasks
mapfile -t GPUS < <(printf "%s\n" $GPUS_STR)
if [ "${#GPUS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -gt 2 ]; then
  echo "ERROR: GPUS_STR must contain one or two GPUs, e.g. '0 1'." >&2
  exit 1
fi

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] tasks=$(wc -l < "$TASK_FILE") gpus=${GPUS[*]}" | tee -a "$LOG_DIR/master.log"

for shard in "${!GPUS[@]}"; do
  worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  echo $! > "$LOG_DIR/worker_gpu${GPUS[$shard]}.pid"
done

wait
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
