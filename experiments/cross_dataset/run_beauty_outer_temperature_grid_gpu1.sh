#!/usr/bin/env bash
set -euo pipefail

# Wide, validation-driven temperature screen for the three score geometries.
# This is deliberately a selection experiment, not a test-set comparison:
# choose tau from each log's best validation NDCG@10 before any final reruns.
# It uses GPU 1 only and trains jobs sequentially to avoid resource contention.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
TUNING_SEED="${TUNING_SEED:-2026}"
TEMPERATURES="${TEMPERATURES:-0.5 1 2 5 10 20 40}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_outer_temperature_grid_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_outer_temperature_grid_gpu1}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local tag="$1" geometry="$2" tau="$3" seq_power="$4" item_power="$5"
  local log="$OUT_ROOT/seed${TUNING_SEED}/${tag}_tau${tau}.log"
  local ckpt="$CKPT_ROOT/seed${TUNING_SEED}/${tag}_tau${tau}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'test result:' "$log" 2>/dev/null; then
    echo "SKIP seed=$TUNING_SEED tag=$tag tau=$tau"; return 0
  fi
  echo "START seed=$TUNING_SEED tag=$tag tau=$tau gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$TUNING_SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature "$tau" --score_geometry "$geometry" \
    --sequence_norm_power "$seq_power" --item_norm_power "$item_power" \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  grep 'best valid result:' "$log" | tail -1 | tee -a "$OUT_ROOT/driver.log" || true
}

for tau in $TEMPERATURES; do
  run_one sequence sequence "$tau" 1 0
  run_one item item "$tau" 0 1
  run_one both both "$tau" 1 1
done
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
