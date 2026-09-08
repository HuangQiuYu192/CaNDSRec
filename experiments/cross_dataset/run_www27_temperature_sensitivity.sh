#!/usr/bin/env bash
set -euo pipefail

# Validation-only temperature sensitivity. Runs serially on the explicitly
# selected GPU (default: 1) so it never overlaps another task on that card. The test set is only read by main.py's
# final report; no test metric selects a temperature.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2025}"
TEMPS_STR="${TEMPS_STR:-5 10 20}"
GPU_ID="${GPU_ID:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_temperature_sensitivity}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_temperature_sensitivity}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    for temp in $TEMPS_STR; do
      log="$OUT_ROOT/$dataset/seed$seed/temp$temp.log"
      ckpt="$CKPT_ROOT/$dataset/seed$seed/temp$temp"
      mkdir -p "$(dirname "$log")" "$ckpt"
      if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
        echo "SKIP dataset=$dataset seed=$seed temp=$temp"
        continue
      fi
      echo "START dataset=$dataset seed=$seed temp=$temp (GPU $GPU_ID)"
      conda run --no-capture-output -n "$CONDA_ENV" python main.py \
        --model CANDSSASRec --dataset "$dataset" --gpu_id "$GPU_ID" --seed "$seed" \
        --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
        --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
        --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
        --max_item_list_length 50 --temperature "$temp" --checkpoint_dir "$ckpt" \
        --verbose True --show_progress True > "$log" 2>&1
      echo "DONE dataset=$dataset seed=$seed temp=$temp"
    done
  done
done
echo "ALL_DONE"
