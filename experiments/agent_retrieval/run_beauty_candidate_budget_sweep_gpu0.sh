#!/usr/bin/env bash
set -euo pipefail

# Fair candidate-budget analysis for metadata and split-safe review evidence.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/agent_retrieval/beauty_candidate_budget_sweep}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/agent_retrieval/beauty_candidate_budget_sweep}"
EVIDENCE_SOURCES="${EVIDENCE_SOURCES:-metadata reviews}"
QUOTAS="${QUOTAS:-5,10,25,50}"
MAX_BATCHES="${MAX_BATCHES:-}"
CHECKPOINT="${CHECKPOINT:-$ROOT/ckpt/main_benchmark_grid/Beauty_CANDSSASRec_h256_len50_temp10/CANDSSASRec-Sep-02-2026_19-55-43.pth}"
if [ ! -f "$CHECKPOINT" ]; then CHECKPOINT="$(find "$ROOT/ckpt" -type f -path '*Beauty_CANDSSASRec_h256_len50_temp10*' -name '*.pth' | head -n 1 || true)"; fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then echo "ERROR: set CHECKPOINT to a Beauty CaNDS .pth file" >&2; exit 1; fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi
extra_args=()
if [ -n "$MAX_BATCHES" ]; then extra_args=(--max_batches "$MAX_BATCHES"); fi
for source in $EVIDENCE_SOURCES; do
  if [ "$source" = "reviews" ] && [ ! -f "$ROOT/dataset/process/raw/reviews_Beauty_5.json" ]; then
    echo "ERROR: missing dataset/process/raw/reviews_Beauty_5.json" >&2; exit 1
  fi
  tag="Beauty_h256_len50_temp10_${source}"
  echo "START $tag checkpoint=$CHECKPOINT"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/agent_retrieval/analyze_beauty_candidate_budget_sweep.py \
    --checkpoint "$CHECKPOINT" --evidence_source "$source" --gpu_id "$GPU_ID" --hidden_size 256 \
    --max_item_list_length 50 --inner_size 1024 --temperature 10 --n_layers 2 --n_heads 2 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 --train_batch_size 1024 \
    --eval_batch_size 256 --budget 100 --quotas "$QUOTAS" --out_prefix "$OUT_DIR/$tag" "${extra_args[@]}" \
    | tee "$LOG_DIR/${tag}.log"
done
echo "ALL_DONE"
