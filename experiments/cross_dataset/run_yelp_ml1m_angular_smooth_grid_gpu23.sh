#!/usr/bin/env bash
set -euo pipefail

# Coarse hyperparameter grid for AngularSmooth on Yelp-S3Rec and ML-1M.
# It tunes only the new AngularSmooth module and keeps CaNDS temperatures fixed.
# Default uses GPU 2 and 3 as requested.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPUS_STR="${GPUS_STR:-2 3}"

DATASETS_STR="${DATASETS_STR:-Yelp-S3Rec ML-1M}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
INNER_SIZE="${INNER_SIZE:-$(( HIDDEN_SIZE * 4 ))}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

YELP_MAX_LEN="${YELP_MAX_LEN:-50}"
ML1M_MAX_LEN="${ML1M_MAX_LEN:-50}"
YELP_TEMP="${YELP_TEMP:-10}"
ML1M_TEMP="${ML1M_TEMP:-20}"

# Coarse grid: 3 * 3 * 2 * 3 = 54 settings per dataset.
SMOOTH_WEIGHTS_STR="${SMOOTH_WEIGHTS_STR:-0.03 0.1 0.2}"
SMOOTH_KS_STR="${SMOOTH_KS_STR:-5 10 20}"
SMOOTH_TEMPS_STR="${SMOOTH_TEMPS_STR:-0.1}"
SMOOTH_QUANTILES_STR="${SMOOTH_QUANTILES_STR:-0.33 0.67 1.0}"
SIM_THRESHOLDS_STR="${SIM_THRESHOLDS_STR:-0.0 0.2}"

TAG="${TAG:-yelp_ml1m_angular_smooth_grid_gpu23}"
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
fi

max_len_for_dataset() {
  case "$1" in
    Yelp-S3Rec) echo "$YELP_MAX_LEN" ;;
    ML-1M) echo "$ML1M_MAX_LEN" ;;
    *) echo "50" ;;
  esac
}

temperature_for_dataset() {
  case "$1" in
    Yelp-S3Rec) echo "$YELP_TEMP" ;;
    ML-1M) echo "$ML1M_TEMP" ;;
    *) echo "10" ;;
  esac
}

build_tasks() {
  : > "$TASK_FILE"
  for dataset in $DATASETS_STR; do
    max_len="$(max_len_for_dataset "$dataset")"
    temperature="$(temperature_for_dataset "$dataset")"
    for smooth_weight in $SMOOTH_WEIGHTS_STR; do
      for smooth_k in $SMOOTH_KS_STR; do
        for smooth_temp in $SMOOTH_TEMPS_STR; do
          for smooth_quantile in $SMOOTH_QUANTILES_STR; do
            for sim_threshold in $SIM_THRESHOLDS_STR; do
              name="${dataset}_AngularSmoothCANDSSASRec_h${HIDDEN_SIZE}_len${max_len}_temp${temperature}_w${smooth_weight}_k${smooth_k}_st${smooth_temp}_q${smooth_quantile}_thr${sim_threshold}"
              printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$name" "$dataset" "$max_len" "$temperature" "$smooth_weight" "$smooth_k" "$smooth_temp" "$smooth_quantile" "$sim_threshold" >> "$TASK_FILE"
            done
          done
        done
      done
    done
  done
}

run_task() {
  local gpu="$1"
  local name="$2"
  local dataset="$3"
  local max_len="$4"
  local temperature="$5"
  local smooth_weight="$6"
  local smooth_k="$7"
  local smooth_temp="$8"
  local smooth_quantile="$9"
  local sim_threshold="${10}"
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
    --model AngularSmoothCANDSSASRec \
    --gpu_id "$gpu" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$INNER_SIZE" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --stopping_step "$STOPPING_STEP" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_item_list_length "$max_len" \
    --temperature "$temperature" \
    --angular_smooth_weight "$smooth_weight" \
    --angular_smooth_k "$smooth_k" \
    --angular_smooth_temperature "$smooth_temp" \
    --angular_smooth_pop_quantile "$smooth_quantile" \
    --angular_smooth_sim_threshold "$sim_threshold" \
    --angular_smooth_pop_weight False \
    --checkpoint_dir "$ckpt_path" \
    --verbose True \
    --show_progress True \
    > "$log_file" 2>&1

  if ! grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR missing test result: $name. Check $log_file" | tee -a "$LOG_DIR/master.log"
    return 1
  fi
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$LOG_DIR/master.log"
}

worker() {
  local gpu="$1"
  local shard="$2"
  local shards="$3"
  local index=0

  while IFS=$'\t' read -r name dataset max_len temperature smooth_weight smooth_k smooth_temp smooth_quantile sim_threshold; do
    if [ $(( index % shards )) -eq "$shard" ]; then
      run_task "$gpu" "$name" "$dataset" "$max_len" "$temperature" "$smooth_weight" "$smooth_k" "$smooth_temp" "$smooth_quantile" "$sim_threshold"
    fi
    index=$(( index + 1 ))
  done < "$TASK_FILE"
}

build_tasks
mapfile -t GPUS < <(printf "%s\n" $GPUS_STR)
if [ "${#GPUS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -gt 2 ]; then
  echo "ERROR: GPUS_STR must contain one or two GPUs, e.g. '2 3'." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [ "$gpu" != "2" ] && [ "$gpu" != "3" ]; then
    echo "ERROR: this tuning script is restricted to GPU 2/3. Got gpu=$gpu from GPUS_STR='$GPUS_STR'." >&2
    exit 1
  fi
done

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] tasks=$(wc -l < "$TASK_FILE") datasets=$DATASETS_STR gpus=${GPUS[*]}" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] weights=$SMOOTH_WEIGHTS_STR ks=$SMOOTH_KS_STR smooth_temps=$SMOOTH_TEMPS_STR quantiles=$SMOOTH_QUANTILES_STR thresholds=$SIM_THRESHOLDS_STR" | tee -a "$LOG_DIR/master.log"

for shard in "${!GPUS[@]}"; do
  worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  echo $! > "$LOG_DIR/worker_gpu${GPUS[$shard]}.pid"
done
wait

missing=0
while IFS=$'\t' read -r name dataset max_len temperature smooth_weight smooth_k smooth_temp smooth_quantile sim_threshold; do
  if ! grep -q "test result" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR missing test result for $name. Check $LOG_DIR/${name}.log" | tee -a "$LOG_DIR/master.log"
    missing=1
  fi
done < "$TASK_FILE"
if [ "$missing" -ne 0 ]; then
  exit 1
fi

python experiments/cross_dataset/collect_angular_smooth_results.py --log_dir "$LOG_DIR" || true
python experiments/cross_dataset/summarize_angular_smooth_grid.py \
  --input_csv "$LOG_DIR/angular_smooth_summary.csv" \
  --out_md "$LOG_DIR/best_by_dataset.md" \
  --topk "${TOPK_PER_DATASET:-5}" || true
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
echo
cat "$LOG_DIR/angular_smooth_summary.md" 2>/dev/null || true
echo
cat "$LOG_DIR/best_by_dataset.md" 2>/dev/null || true
