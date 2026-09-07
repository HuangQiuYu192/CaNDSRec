#!/usr/bin/env bash
set -euo pipefail

# Second-layer semantic anchor test: small pretrained text encoder.
# Default: sentence-transformers/all-MiniLM-L6-v2 on Beauty h256 temp=10.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
TEXT_FIELDS="${TEXT_FIELDS:-title,categories}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_MODEL_TAG="${TEXT_MODEL_TAG:-allminilml6v2}"
TEXT_DIM="${TEXT_DIM:-384}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"
TEXT_DEVICE="${TEXT_DEVICE:-cuda:0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
INNER_SIZE="${INNER_SIZE:-1024}"
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
TOPK_STR="${TOPK_STR:-5 10 20 50 100}"

# Compact default grid from TF-IDF evidence:
# - score/constant/0.1: best overall NDCG@10
# - anchor/tail/0.1: best tail-oriented behavior
SETTINGS_STR="${SETTINGS_STR:-score:constant:0.1 anchor:tail:0.1}"
SEMANTIC_FREEZE="${SEMANTIC_FREEZE:-True}"

TAG="${TAG:-beauty_pretrained_semantic_anchor_gpu0}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
TASK_FILE="$LOG_DIR/tasks.tsv"
SEM_PATH="${SEM_PATH:-$ROOT/dataset/$DATASET/${DATASET}.${TEXT_MODEL_TAG}.npy}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] build pretrained semantic embeddings dataset=$DATASET model=$TEXT_MODEL_NAME" | tee -a "$LOG_DIR/master.log"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_ENDPOINT="$HF_ENDPOINT"
conda run --no-capture-output -n "$CONDA_ENV" python experiments/semantic_anchor/build_pretrained_item_embeddings.py \
  --dataset "$DATASET" \
  --data_path "$ROOT/dataset" \
  --fields "$TEXT_FIELDS" \
  --model_name "$TEXT_MODEL_NAME" \
  --batch_size "$TEXT_BATCH_SIZE" \
  --device "$TEXT_DEVICE" \
  --out "$SEM_PATH" \
  > "$LOG_DIR/build_pretrained_embeddings.log" 2>&1

build_tasks() {
  : > "$TASK_FILE"
  printf "%s\t%s\t%s\t%s\n" \
    "${DATASET}_CANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}" \
    "CANDSSASRec" "" "" >> "$TASK_FILE"
  for setting in $SETTINGS_STR; do
    local mode="${setting%%:*}"
    local rest="${setting#*:}"
    local gate="${rest%%:*}"
    local weight="${rest##*:}"
    local name="${DATASET}_SemanticCANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}_text${TEXT_MODEL_TAG}_dim${TEXT_DIM}_mode${mode}_gate${gate}_w${weight}"
    printf "%s\t%s\t%s\t%s\n" "$name" "SemanticCANDSSASRec" "$mode" "$gate:$weight" >> "$TASK_FILE"
  done
}

run_task() {
  local name="$1"
  local model="$2"
  local mode="$3"
  local gate_weight="$4"
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/${name}"

  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP $name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  local extra_args=()
  if [ "$model" = "SemanticCANDSSASRec" ]; then
    local gate="${gate_weight%%:*}"
    local weight="${gate_weight##*:}"
    extra_args=(
      --semantic_embedding_path "$SEM_PATH"
      --semantic_fusion_mode "$mode"
      --semantic_gate "$gate"
      --semantic_weight "$weight"
      --semantic_freeze "$SEMANTIC_FREEZE"
    )
  fi

  mkdir -p "$ckpt_path"
  echo "[$(date '+%F %T')] START $name gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --dataset "$DATASET" \
    --model "$model" \
    --gpu_id "$GPU_ID" \
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
    --max_item_list_length "$MAX_LEN" \
    --temperature "$TEMPERATURE" \
    --checkpoint_dir "$ckpt_path" \
    --topk $TOPK_STR \
    --verbose True \
    --show_progress True \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$LOG_DIR/master.log"
}

build_tasks
echo "[$(date '+%F %T')] tasks=$(wc -l < "$TASK_FILE") gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"

while IFS=$'\t' read -r name model mode gate_weight; do
  run_task "$name" "$model" "$mode" "$gate_weight"
done < "$TASK_FILE"

python experiments/semantic_anchor/collect_semantic_anchor_results.py --log_dir "$LOG_DIR"

echo
cat "$LOG_DIR/semantic_anchor_summary.md"
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
