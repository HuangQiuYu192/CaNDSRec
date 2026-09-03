#!/usr/bin/env bash
set -euo pipefail

# Beauty-only FMLPRec backbone benchmark.
# Runs FMLPRec dot-product baseline and CANDSFMLPRec cosine-temperature grid.
# Default uses only GPU 0, as requested.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
SEED="${SEED:-2025}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-64 128 256}"
TEMPS_STR="${TEMPS_STR:-2 5 7.5 10 15 20 30 40}"
GPU_ID="${GPU_ID:-0}"

MAX_LEN="${MAX_LEN:-50}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

TAG="${TAG:-beauty_fmlprec_cands_grid_gpu0}"
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

inner_size_for_hidden() {
  echo $(( "$1" * 4 ))
}

write_task() {
  local hidden_size="$1"
  local model="$2"
  local temperature="$3"
  local name="${DATASET}_${model}_h${hidden_size}_len${MAX_LEN}"
  if [ "$model" = "CANDSFMLPRec" ]; then
    name="${name}_temp${temperature}"
  fi
  printf "%s\t%s\t%s\t%s\n" "$name" "$hidden_size" "$model" "$temperature" >> "$TASK_FILE"
}

build_tasks() {
  : > "$TASK_FILE"
  for hidden_size in $HIDDEN_SIZES_STR; do
    write_task "$hidden_size" FMLPRec 10.0
    for temp in $TEMPS_STR; do
      write_task "$hidden_size" CANDSFMLPRec "$temp"
    done
  done
}

run_task() {
  local name="$1"
  local hidden_size="$2"
  local model="$3"
  local temperature="$4"
  local inner_size
  inner_size="$(inner_size_for_hidden "$hidden_size")"
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/${name}"

  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP $name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  mkdir -p "$ckpt_path"
  echo "[$(date '+%F %T')] START $name gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --dataset "$DATASET" \
    --model "$model" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$hidden_size" \
    --n_layers "$N_LAYERS" \
    --inner_size "$inner_size" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --stopping_step "$STOPPING_STEP" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_item_list_length "$MAX_LEN" \
    --temperature "$temperature" \
    --checkpoint_dir "$ckpt_path" \
    --verbose True \
    --show_progress True \
    > "$log_file" 2>&1
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$LOG_DIR/master.log"
}

build_tasks

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] tasks=$(wc -l < "$TASK_FILE") gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"

while IFS=$'\t' read -r name hidden_size model temperature; do
  run_task "$name" "$hidden_size" "$model" "$temperature"
done < "$TASK_FILE"

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
