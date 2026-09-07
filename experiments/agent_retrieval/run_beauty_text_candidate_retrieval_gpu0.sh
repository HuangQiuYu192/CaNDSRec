#!/usr/bin/env bash
set -euo pipefail

# Pre-Agent validation: can static catalog text supply tail candidates that
# CaNDS misses?  The script never indexes review text in this first stage.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/agent_retrieval/beauty_text_candidate_retrieval}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/agent_retrieval/beauty_text_candidate_retrieval}"

DATASET="${DATASET:-Beauty}"
SEED="${SEED:-2025}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
INNER_SIZE="${INNER_SIZE:-256}"
TEMPERATURE="${TEMPERATURE:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
TOPK="${TOPK:-100}"
CANDS_QUOTA="${CANDS_QUOTA:-75}"
TEXT_QUOTA="${TEXT_QUOTA:-25}"
MAX_BATCHES="${MAX_BATCHES:-}"

# Override CHECKPOINT when evaluating another trained CaNDS configuration.
CHECKPOINT="${CHECKPOINT:-$ROOT/ckpt/main_benchmark_grid/Beauty_CANDSSASRec_h256_len50_temp10/CANDSSASRec-Sep-02-2026_19-55-43.pth}"
if [ ! -f "$CHECKPOINT" ]; then
  CHECKPOINT="$(find "$ROOT/ckpt" -type f -path '*Beauty_CANDSSASRec_h256_len50_temp10*' -name '*.pth' | head -n 1 || true)"
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: no Beauty h256/len50/temp10 CaNDS checkpoint found. Set CHECKPOINT=/absolute/path/to/file.pth" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi

extra_args=()
if [ -n "$MAX_BATCHES" ]; then
  extra_args=(--max_batches "$MAX_BATCHES")
fi
tag="${DATASET}_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}"
echo "START $tag checkpoint=$CHECKPOINT"
conda run --no-capture-output -n "$CONDA_ENV" python experiments/agent_retrieval/analyze_beauty_text_candidate_retrieval.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --gpu_id "$GPU_ID" --seed "$SEED" \
  --hidden_size "$HIDDEN_SIZE" --max_item_list_length "$MAX_LEN" --inner_size "$INNER_SIZE" \
  --temperature "$TEMPERATURE" --n_layers 2 --n_heads 2 --hidden_dropout_prob 0.5 \
  --attn_dropout_prob 0.5 --learning_rate 0.001 --train_batch_size 1024 \
  --eval_batch_size "$EVAL_BATCH_SIZE" --topk "$TOPK" --cands_quota "$CANDS_QUOTA" \
  --text_quota "$TEXT_QUOTA" --out_prefix "$OUT_DIR/$tag" "${extra_args[@]}" \
  | tee "$LOG_DIR/${tag}.log"
echo "ALL_DONE"
