#!/usr/bin/env bash
set -euo pipefail

# Confirm the P0 scale-only control on seeds independent of the temperature
# development seed. The temperature is fixed before these test runs.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
TEMPERATURE="${TEMPERATURE:-0.125}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_scaled_dot_confirmation_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_scaled_dot_confirmation_gpu1}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"

for seed in $SEEDS_STR; do
  log="$OUT_ROOT/seed${seed}.log"
  ckpt="$CKPT_ROOT/seed${seed}"
  if [ "$RERUN" != true ] && grep -q 'test result:' "$log" 2>/dev/null; then
    echo "SKIP seed=$seed" | tee -a "$OUT_ROOT/driver.log"
    continue
  fi
  echo "START scaled-dot confirmation seed=$seed tau=$TEMPERATURE gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature "$TEMPERATURE" --score_geometry partial \
    --sequence_norm_power 0 --item_norm_power 0 --checkpoint_dir "$ckpt" \
    --verbose True --show_progress True > "$log" 2>&1
  grep 'best valid result:' "$log" | tail -1 | tee -a "$OUT_ROOT/driver.log" || true
done

conda run --no-capture-output -n "$CONDA_ENV" python \
  experiments/cross_dataset/collect_beauty_scaled_dot_confirmation.py \
  --log_root "$OUT_ROOT" --temperature "$TEMPERATURE" \
  --out "$ROOT/analysis_results/beauty_scaled_dot_confirmation/summary.csv"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
