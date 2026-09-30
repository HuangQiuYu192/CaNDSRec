#!/usr/bin/env bash
set -euo pipefail

# The decisive scale-only control: GeometrySASRec partial with p_h=p_e=0 is
# exactly tau * h^T e.  It changes global logit scale but retains all radial
# degrees of freedom and dot-product gradient directions.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
TUNING_SEED="${TUNING_SEED:-2026}"
TEMPERATURES="${TEMPERATURES:-0.125 0.25 0.5 1 2 4}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_scaled_dot_temperature_grid_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_scaled_dot_temperature_grid_gpu1}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$OUT_ROOT/seed${TUNING_SEED}"
for tau in $TEMPERATURES; do
  log="$OUT_ROOT/seed${TUNING_SEED}/scaled_dot_tau${tau}.log"
  ckpt="$CKPT_ROOT/seed${TUNING_SEED}/scaled_dot_tau${tau}"
  mkdir -p "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'test result:' "$log" 2>/dev/null; then echo "SKIP tau=$tau"; continue; fi
  echo "START seed=$TUNING_SEED scaled_dot tau=$tau gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$TUNING_SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature "$tau" --score_geometry partial \
    --sequence_norm_power 0 --item_norm_power 0 --checkpoint_dir "$ckpt" \
    --verbose True --show_progress True > "$log" 2>&1
  grep 'best valid result:' "$log" | tail -1 | tee -a "$OUT_ROOT/driver.log" || true
done
conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/collect_beauty_scaled_dot_grid.py \
  --log_root "$OUT_ROOT/seed${TUNING_SEED}" \
  --out "$ROOT/analysis_results/beauty_scaled_dot_temperature_grid/seed${TUNING_SEED}_summary.csv"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
