#!/usr/bin/env bash
set -euo pipefail

# Angular neighbor positive smoothing on top of CaNDS.
# Default: Beauty, hidden size 256, temperature 10, GPU 0.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

SMOOTH_WEIGHTS_STR="${SMOOTH_WEIGHTS_STR:-0.01 0.03 0.05 0.1}"
SMOOTH_KS_STR="${SMOOTH_KS_STR:-5 10 20}"
SMOOTH_TEMPS_STR="${SMOOTH_TEMPS_STR:-0.1 0.2}"
SMOOTH_QUANTILES_STR="${SMOOTH_QUANTILES_STR:-0.67}"
SIM_THRESHOLDS_STR="${SIM_THRESHOLDS_STR:-0.0 0.2}"
POP_WEIGHT="${POP_WEIGHT:-True}"

TAG="${TAG:-beauty_angular_smooth_cands_gpu0}"
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

build_tasks() {
  : > "$TASK_FILE"
  for hidden_size in $HIDDEN_SIZES_STR; do
    inner_size="$(inner_size_for_hidden "$hidden_size")"
    for smooth_weight in $SMOOTH_WEIGHTS_STR; do
      for smooth_k in $SMOOTH_KS_STR; do
        for smooth_temp in $SMOOTH_TEMPS_STR; do
          for smooth_quantile in $SMOOTH_QUANTILES_STR; do
            for sim_threshold in $SIM_THRESHOLDS_STR; do
              name="${DATASET}_AngularSmoothCANDSSASRec_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}_w${smooth_weight}_k${smooth_k}_st${smooth_temp}_q${smooth_quantile}_thr${sim_threshold}"
              printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$name" "$hidden_size" "$inner_size" "$smooth_weight" "$smooth_k" "$smooth_temp" "$smooth_quantile" "$sim_threshold" >> "$TASK_FILE"
            done
          done
        done
      done
    done
  done
}

run_task() {
  local name="$1"
  local hidden_size="$2"
  local inner_size="$3"
  local smooth_weight="$4"
  local smooth_k="$5"
  local smooth_temp="$6"
  local smooth_quantile="$7"
  local sim_threshold="$8"
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
    --model AngularSmoothCANDSSASRec \
    --gpu_id "$GPU_ID" \
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
    --max_item_list_length "$MAX_LEN" \
    --temperature "$TEMPERATURE" \
    --angular_smooth_weight "$smooth_weight" \
    --angular_smooth_k "$smooth_k" \
    --angular_smooth_temperature "$smooth_temp" \
    --angular_smooth_pop_quantile "$smooth_quantile" \
    --angular_smooth_sim_threshold "$sim_threshold" \
    --angular_smooth_pop_weight "$POP_WEIGHT" \
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

while IFS=$'\t' read -r name hidden_size inner_size smooth_weight smooth_k smooth_temp smooth_quantile sim_threshold; do
  run_task "$name" "$hidden_size" "$inner_size" "$smooth_weight" "$smooth_k" "$smooth_temp" "$smooth_quantile" "$sim_threshold"
done < "$TASK_FILE"

python experiments/cross_dataset/collect_main_benchmark_results.py --log_dir "$LOG_DIR" || true
python experiments/cross_dataset/collect_angular_smooth_results.py --log_dir "$LOG_DIR" || true
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
