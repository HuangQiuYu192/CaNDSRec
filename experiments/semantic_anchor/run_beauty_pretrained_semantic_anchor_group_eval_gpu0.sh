#!/usr/bin/env bash
set -euo pipefail

# Head/mid/tail group evaluation for pretrained semantic anchor experiments.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

TAG="${TAG:-beauty_pretrained_semantic_anchor_gpu0}"
TEXT_MODEL_TAG="${TEXT_MODEL_TAG:-allminilml6v2}"
SUMMARY_CSV="${SUMMARY_CSV:-$ROOT/log_runs/$TAG/semantic_anchor_summary.csv}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
SEMANTIC_EMBEDDING_PATH="${SEMANTIC_EMBEDDING_PATH:-$ROOT/dataset/Beauty/Beauty.${TEXT_MODEL_TAG}.npy}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/beauty_pretrained_semantic_anchor_group_eval}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/beauty_pretrained_semantic_anchor_group_eval}"
TASK_FILE="$OUT_DIR/tasks.tsv"

TOP_N="${TOP_N:-2}"
RANK_METRIC="${RANK_METRIC:-ndcg@10}"
CUTOFFS="${CUTOFFS:-5,10,20,50,100}"

SEED="${SEED:-2025}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

python experiments/semantic_anchor/prepare_semantic_anchor_group_eval_tasks.py \
  --summary_csv "$SUMMARY_CSV" \
  --ckpt_dir "$CKPT_DIR" \
  --semantic_embedding_path "$SEMANTIC_EMBEDDING_PATH" \
  --top_n "$TOP_N" \
  --rank_metric "$RANK_METRIC" \
  --out_tsv "$TASK_FILE"

eval_one() {
  local run_name="$1"
  local dataset="$2"
  local method="$3"
  local model="$4"
  local hidden="$5"
  local max_len="$6"
  local inner_size="$7"
  local temp="$8"
  local svd_dim="$9"
  local text_model="${10}"
  local text_dim="${11}"
  local mode="${12}"
  local gate="${13}"
  local weight="${14}"
  local semantic_embedding_path="${15}"
  local checkpoint="${16}"

  checkpoint="${checkpoint%$'\r'}"
  local tag="${method}_${run_name}"
  local out_prefix="$OUT_DIR/$tag"
  local log_file="$LOG_DIR/${tag}.log"

  if [ -z "$checkpoint" ]; then
    echo "[$(date '+%F %T')] MISSING checkpoint for $run_name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  local extra_args=()
  if [ "$model" = "SemanticCANDSSASRec" ]; then
    extra_args=(
      --semantic_embedding_path "$semantic_embedding_path"
      --semantic_fusion_mode "$mode"
      --semantic_gate "$gate"
      --semantic_weight "$weight"
      --semantic_freeze True
    )
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
    --temperature "$temp" \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOG_DIR/master.log"
}

tail -n +2 "$TASK_FILE" | while IFS=$'\t' read -r run_name dataset method model hidden max_len inner_size temp svd_dim text_model text_dim mode gate weight semantic_embedding_path checkpoint; do
  eval_one "$run_name" "$dataset" "$method" "$model" "$hidden" "$max_len" "$inner_size" "$temp" "$svd_dim" "$text_model" "$text_dim" "$mode" "$gate" "$weight" "$semantic_embedding_path" "$checkpoint"
done

python experiments/semantic_anchor/collect_semantic_anchor_group_results.py \
  --group_dir "$OUT_DIR"

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
echo
cat "$OUT_DIR/semantic_anchor_group_table.md"
