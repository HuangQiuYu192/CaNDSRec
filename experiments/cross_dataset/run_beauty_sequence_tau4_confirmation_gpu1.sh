#!/usr/bin/env bash
set -euo pipefail

# Independent-seed confirmation for the small tuning-seed improvement from
# sequence-only tau=5 to tau=4.  Tau=5 is deliberately *not* rerun: the exact
# same configuration already has completed logs for seeds 2023--2025.  This
# script produces only the missing tau=4 arm on GPU 1, then invokes a paired
# collector against those preserved tau=5 results.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEEDS="${SEEDS:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_sequence_tau4_confirmation_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_sequence_tau4_confirmation_gpu1}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

for seed in $SEEDS; do
  log="$OUT_ROOT/seed${seed}/sequence_tau4.log"
  ckpt="$CKPT_ROOT/seed${seed}/sequence_tau4"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'test result:' "$log" 2>/dev/null; then
    echo "SKIP seed=$seed tau=4" | tee -a "$OUT_ROOT/driver.log"
    continue
  fi
  echo "START seed=$seed sequence_tau4 gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature 4 --score_geometry sequence \
    --sequence_norm_power 1 --item_norm_power 0 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  grep 'best valid result:' "$log" | tail -1 | tee -a "$OUT_ROOT/driver.log" || true
done

python experiments/cross_dataset/collect_beauty_sequence_temperature_confirmation.py
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
