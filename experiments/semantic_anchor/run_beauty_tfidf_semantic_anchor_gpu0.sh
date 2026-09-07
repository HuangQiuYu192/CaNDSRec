#!/usr/bin/env bash
set -euo pipefail

# First-layer semantic anchor test: TF-IDF + SVD item text embeddings.
# Default runs Beauty h256 temp=10 on GPU 0.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
TEXT_FIELDS="${TEXT_FIELDS:-title,categories}"
SVD_DIM="${SVD_DIM:-128}"
MAX_FEATURES="${MAX_FEATURES:-50000}"
MIN_DF="${MIN_DF:-2}"

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

FUSION_MODES_STR="${FUSION_MODES_STR:-score anchor}"
GATES_STR="${GATES_STR:-constant tail}"
WEIGHTS_STR="${WEIGHTS_STR:-0.05 0.1 0.2}"
SEMANTIC_FREEZE="${SEMANTIC_FREEZE:-True}"

TAG="${TAG:-beauty_tfidf_semantic_anchor_gpu0}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
TASK_FILE="$LOG_DIR/tasks.tsv"
SEM_PATH="${SEM_PATH:-$ROOT/dataset/$DATASET/${DATASET}.tfidf_svd${SVD_DIM}.npy}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
  echo "Set CONDA_SH=/path/to/conda.sh if conda is not already available." | tee -a "$LOG_DIR/master.log"
fi

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] build semantic embeddings dataset=$DATASET fields=$TEXT_FIELDS dim=$SVD_DIM" | tee -a "$LOG_DIR/master.log"
conda run --no-capture-output -n "$CONDA_ENV" python experiments/semantic_anchor/build_tfidf_svd_item_embeddings.py \
  --dataset "$DATASET" \
  --data_path "$ROOT/dataset" \
  --fields "$TEXT_FIELDS" \
  --dim "$SVD_DIM" \
  --max_features "$MAX_FEATURES" \
  --min_df "$MIN_DF" \
  --out "$SEM_PATH" \
  > "$LOG_DIR/build_tfidf_svd.log" 2>&1

build_tasks() {
  : > "$TASK_FILE"
  printf "%s\t%s\t%s\t%s\n" \
    "${DATASET}_CANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}" \
    "CANDSSASRec" "" "" >> "$TASK_FILE"
  for mode in $FUSION_MODES_STR; do
    for gate in $GATES_STR; do
      for weight in $WEIGHTS_STR; do
        name="${DATASET}_SemanticCANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}_svd${SVD_DIM}_mode${mode}_gate${gate}_w${weight}"
        printf "%s\t%s\t%s\t%s\n" "$name" "SemanticCANDSSASRec" "$mode" "$gate:$weight" >> "$TASK_FILE"
      done
    done
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
