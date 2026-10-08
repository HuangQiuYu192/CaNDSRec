#!/usr/bin/env bash
set -euo pipefail

# Development-only selection of the sequence-only score scale.  The model
# normalizes h but not e, so its frozen within-user ranking is still the dot
# ranking.  Seed 2026 is never included in final test reporting.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2026}"
DIM="${DIM:-256}"
TEMPS_STR="${TEMPS_STR:-2 4 6 8 10 15 20}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_sequence_temperature_screen}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_sequence_temperature_screen}"

for tau in $TEMPS_STR; do
  tag="GeometryBSARec_sequence_d${DIM}_tau${tau}_a0.5_c9_h2_lr0.001"
  log="$OUT_ROOT/Beauty/seed${SEED}/${tag}.log"
  ckpt="$CKPT_ROOT/Beauty/seed${SEED}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && continue
  echo "START seed=$SEED d=$DIM geometry=sequence tau=$tau gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model GeometryBSARec --score_geometry sequence --dataset Beauty --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size "$DIM" --n_layers 2 --n_heads 2 --inner_size "$((4 * DIM))" --alpha 0.5 --c 9 \
    --temperature "$tau" --learning_rate 0.001 --weight_decay 0 --epochs 300 --stopping_step 10 \
    --train_batch_size 1024 --eval_batch_size 512 --max_item_list_length 50 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
done
echo ALL_DONE
