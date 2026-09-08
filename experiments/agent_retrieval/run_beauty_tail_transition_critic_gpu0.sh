#!/usr/bin/env bash
set -euo pipefail

# Stage 2: choose a behaviour-only Critic threshold on validation and report a
# single frozen test check. This does not train on pseudo interactions yet.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/agent_simulator/beauty_tail_transition_critic}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/agent_simulator/beauty_tail_transition_critic}"
CHECKPOINT="${CHECKPOINT:-$ROOT/ckpt/main_benchmark_grid/Beauty_CANDSSASRec_h256_len50_temp10/CANDSSASRec-Sep-02-2026_19-55-43.pth}"
if [ ! -f "$CHECKPOINT" ]; then CHECKPOINT="$(find "$ROOT/ckpt" -type f -path '*Beauty_CANDSSASRec_h256_len50_temp10/*.pth' | head -n 1 || true)"; fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then echo "ERROR: set CHECKPOINT to a Beauty T=10 CaNDS checkpoint" >&2; exit 1; fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi
tag="Beauty_h256_len50_temp10"
echo "START Stage-2 behaviour Critic validation checkpoint=$CHECKPOINT"
conda run --no-capture-output -n "$CONDA_ENV" python experiments/agent_retrieval/analyze_tail_transition_critic.py \
  --checkpoint "$CHECKPOINT" --dataset Beauty --gpu_id "$GPU_ID" --seed 2025 --hidden_size 256 \
  --max_item_list_length 50 --inner_size 1024 --temperature 10 --n_layers 2 --n_heads 2 \
  --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 --train_batch_size 1024 \
  --eval_batch_size 512 --min_contexts 3 --min_predecessors 2 --recent_window 5 --transition_decay 0.7 \
  --max_candidates 20 --quantiles 0,0.25,0.5,0.75 --min_retention 0.10 --calibration_bins 5 \
  --out_prefix "$OUT_DIR/$tag" | tee "$LOG_DIR/${tag}.log"
echo "ALL_DONE"
