#!/usr/bin/env bash
set -euo pipefail

# This is NOT post-hoc score scaling. It routes among separately trained CaNDS
# checkpoints whose training temperatures differ.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
TEMPERATURES="${TEMPERATURES:-5 10 20}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/agent_retrieval/beauty_temperature_expert_routing}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/agent_retrieval/beauty_temperature_expert_routing}"
MAX_BATCHES="${MAX_BATCHES:-}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi
spec=()
for temperature in $TEMPERATURES; do
  checkpoint="$(find "$ROOT/ckpt" -type f -path "*Beauty_CANDSSASRec_h256_len50_temp${temperature}/*.pth" | head -n 1 || true)"
  if [ -z "$checkpoint" ]; then
    echo "ERROR: no trained Beauty CaNDS checkpoint found for temperature=$temperature. Set TEMPERATURES to available trained values." >&2
    exit 1
  fi
  spec+=("${temperature}=${checkpoint}")
done
checkpoints="$(IFS=,; echo "${spec[*]}")"
extra_args=()
if [ -n "$MAX_BATCHES" ]; then extra_args=(--max_batches "$MAX_BATCHES"); fi
echo "START experts: $checkpoints"
conda run --no-capture-output -n "$CONDA_ENV" python experiments/agent_retrieval/analyze_temperature_expert_routing.py \
  --checkpoints "$checkpoints" --dataset Beauty --gpu_id "$GPU_ID" --seed 2025 --hidden_size 256 \
  --max_item_list_length 50 --inner_size 1024 --n_layers 2 --n_heads 2 --hidden_dropout_prob 0.5 \
  --attn_dropout_prob 0.5 --learning_rate 0.001 --train_batch_size 1024 --eval_batch_size 256 \
  --reference_temperature 10 --recent_window 5 --select_k 10 \
  --out_prefix "$OUT_DIR/Beauty_h256_len50_temperature_experts" "${extra_args[@]}" \
  | tee "$LOG_DIR/Beauty_h256_len50_temperature_experts.log"
echo "ALL_DONE"
