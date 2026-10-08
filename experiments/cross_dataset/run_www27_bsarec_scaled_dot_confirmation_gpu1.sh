#!/usr/bin/env bash
set -euo pipefail

# Formal, held-out-seed test of the scale-only alternative to joint
# normalization.  tau=0.04 was selected once using seed 2026 validation only;
# no metric from the three runs below is used for model selection.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
DIM="${DIM:-256}"
TEMPERATURE="${TEMPERATURE:-0.04}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_scaled_dot_confirmation}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_scaled_dot_confirmation}"

for seed in $SEEDS_STR; do
  tag="ScaledDotBSARec_d${DIM}_tau${TEMPERATURE}_a0.5_c9_h2_lr0.001"
  log="$OUT_ROOT/Beauty/seed${seed}/${tag}.log"
  ckpt="$CKPT_ROOT/Beauty/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && continue
  echo "START seed=$seed dimension=$DIM tau=$TEMPERATURE gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model ScaledDotBSARec --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size "$DIM" --n_layers 2 --n_heads 2 --inner_size "$((4 * DIM))" \
    --alpha 0.5 --c 9 --temperature "$TEMPERATURE" --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    > "$log" 2>&1
done

conda run --no-capture-output -n recbole python \
  experiments/cross_dataset/collect_www27_bsarec_scaled_dot_confirmation.py \
  --temperature "$TEMPERATURE"
echo ALL_DONE
