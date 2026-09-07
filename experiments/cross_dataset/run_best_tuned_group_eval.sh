#!/usr/bin/env bash
set -euo pipefail

# Evaluate head/mid/tail groups for the true best-tuned SASRec/CaNDS/AngularSmooth
# settings selected by collect_best_tuned_table.py.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/best_tuned_group_eval}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/best_tuned_group_eval}"
BEST_OUT_DIR="${BEST_OUT_DIR:-$ROOT/analysis_results/best_tuned_table}"
TASK_FILE="$OUT_DIR/tasks.tsv"

SEED="${SEED:-2025}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
CUTOFFS="${CUTOFFS:-5,10,20,50,100}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

python experiments/cross_dataset/collect_best_tuned_table.py \
  --datasets ${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec LastFM-S3Rec} \
  --out_dir "$BEST_OUT_DIR"

python experiments/cross_dataset/prepare_best_tuned_group_eval_tasks.py \
  --best_csv "$BEST_OUT_DIR/best_tuned_table.csv" \
  --out_tsv "$TASK_FILE"

eval_one() {
  local dataset="$1"
  local method="$2"
  local model="$3"
  local hidden="$4"
  local max_len="$5"
  local inner_size="$6"
  local temperature="$7"
  local weight="$8"
  local smooth_k="$9"
  local smooth_temp="${10}"
  local quantile="${11}"
  local threshold="${12}"
  local run_name="${13}"
  local checkpoint="${14}"
  local tag="${dataset}_${method}_best"
  local out_prefix="$OUT_DIR/$tag"
  local log_file="$LOG_DIR/${tag}.log"

  if [ -z "$checkpoint" ]; then
    echo "[$(date '+%F %T')] MISSING checkpoint for $dataset $method $run_name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START $tag gpu=$GPU_ID ckpt=$checkpoint" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --dataset "$dataset" \
    --model "$model" \
    --checkpoint "$checkpoint" \
    --tag "$tag" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$hidden" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$inner_size" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$max_len" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$temperature" \
    --angular_smooth_weight "$weight" \
    --angular_smooth_k "$smooth_k" \
    --angular_smooth_temperature "$smooth_temp" \
    --angular_smooth_pop_quantile "$quantile" \
    --angular_smooth_sim_threshold "$threshold" \
    --angular_smooth_pop_weight False \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOG_DIR/master.log"
}

tail -n +2 "$TASK_FILE" | while IFS=$'\t' read -r dataset method model hidden max_len inner_size temperature weight smooth_k smooth_temp quantile threshold run_name checkpoint; do
  eval_one "$dataset" "$method" "$model" "$hidden" "$max_len" "$inner_size" "$temperature" "$weight" "$smooth_k" "$smooth_temp" "$quantile" "$threshold" "$run_name" "$checkpoint"
done

python experiments/cross_dataset/collect_best_tuned_tail_table.py \
  --group_dir "$OUT_DIR" \
  --out_csv "$OUT_DIR/best_tuned_group_table.csv" \
  --out_md "$OUT_DIR/best_tuned_group_table.md"

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
echo
cat "$OUT_DIR/best_tuned_group_table.md"
