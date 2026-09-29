#!/usr/bin/env bash
set -euo pipefail

# Validation-only local refinement after the wide screen.  The outer screen
# located sequence near tau=5 and joint normalization near tau=10, so this
# runner evaluates only the neighbouring interior values.  Item-only is
# excluded because it was consistently inferior in the outer screen.
#
# This is a tuning run on seed 2026.  Any temperature adopted for a paper
# comparison must subsequently be confirmed on held-out random seeds.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
TUNING_SEED="${TUNING_SEED:-2026}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_outer_temperature_refinement_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_outer_temperature_refinement_gpu1}"
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

# The outer-grid optima are deliberately omitted; they remain in the outer
# table.  These values test whether a narrower interior peak exists.
for tau in 3 4 6 7; do
  run_one sequence sequence "$tau" 1 0
done
for tau in 7 8 12 15; do
  run_one both both "$tau" 1 1
done

python experiments/cross_dataset/collect_beauty_outer_temperature_grid.py \
  --log_root "$OUT_ROOT/seed${TUNING_SEED}" \
  --out "$ROOT/analysis_results/beauty_outer_temperature_refinement/seed${TUNING_SEED}_summary.csv"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
