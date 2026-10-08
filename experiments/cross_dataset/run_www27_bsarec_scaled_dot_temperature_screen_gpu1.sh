#!/usr/bin/env bash
set -euo pipefail

# Validation-only calibration pilot for the high-dimensional dot-product head.
# The model always evaluates test at the end because of the shared runner, but
# the companion collector reads only `best valid result` and must be used for
# temperature selection. Seed 2026 is reserved for this development stage and
# is excluded from all reported three-seed test averages.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2026}"
DIM="${DIM:-256}"
TEMPS_STR="${TEMPS_STR:-0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_scaled_dot_temperature_screen}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_scaled_dot_temperature_screen}"

for tau in $TEMPS_STR; do
  tag="ScaledDotBSARec_d${DIM}_tau${tau}_a0.5_c9_h2_lr0.001"
  log="$OUT_ROOT/Beauty/seed${SEED}/${tag}.log"
  ckpt="$CKPT_ROOT/Beauty/seed${SEED}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && continue
  echo "START seed=$SEED dimension=$DIM tau=$tau gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model ScaledDotBSARec --dataset Beauty --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size "$DIM" --n_layers 2 --n_heads 2 --inner_size "$((4 * DIM))" \
    --alpha 0.5 --c 9 --temperature "$tau" --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    > "$log" 2>&1
done
echo ALL_DONE
