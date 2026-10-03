#!/usr/bin/env bash
set -euo pipefail

# Development-only temperature screen for the scale-only control.  Geometry
# ``partial`` with (p_h, p_e)=(0,0) is exactly a trainable scaled dot product.
# No test metric is used to choose a temperature.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
DATASETS_STR="${DATASETS_STR:-Sports Toys}"
SEED="${SEED:-2026}"
TEMPERATURES_STR="${TEMPERATURES_STR:-0.0625 0.125 0.25 0.5 1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_scaled_dot_temperature_screen_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_scaled_dot_temperature_screen_gpu1}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
for dataset in $DATASETS_STR; do
  for tau in $TEMPERATURES_STR; do
    log="$OUT_ROOT/$dataset/seed$SEED/scaled_dot_tau${tau}.log"
    ckpt="$CKPT_ROOT/$dataset/seed$SEED/scaled_dot_tau${tau}"
    mkdir -p "$(dirname "$log")" "$ckpt"
    if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
      echo "SKIP completed dataset=$dataset tau=$tau" | tee -a "$OUT_ROOT/driver.log"
      continue
    fi
    echo "START P6-screen dataset=$dataset seed=$SEED scaled-dot tau=$tau gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --model GeometrySASRec --dataset "$dataset" --gpu_id "$GPU_ID" --seed "$SEED" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --temperature "$tau" --score_geometry partial \
      --sequence_norm_power 0 --item_norm_power 0 --checkpoint_dir "$ckpt" \
      --verbose True --show_progress True > "$log" 2>&1
    grep -A2 'test result' "$log" | tee -a "$OUT_ROOT/driver.log" || true
  done
done
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
