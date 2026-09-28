#!/usr/bin/env bash
set -euo pipefail

# Final seed replication for the controlled Beauty score-geometry study.
# This launcher is intentionally sequential and uses GPU 3 only. It reuses
# completed same-configuration logs for seed 2025 and fills the missing seeds.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-3}"
SEEDS="${SEEDS:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_radial_power_grid_gpu12}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local seed="$1" tag="$2" model="$3" tau="$4" geometry="$5" seq_power="$6" item_power="$7"
  local log="$OUT_ROOT/seed${seed}/${tag}_tau${tau}.log"
  local ckpt="$CKPT_ROOT/seed${seed}/${tag}_tau${tau}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'test result:' "$log" 2>/dev/null; then
    echo "SKIP seed=$seed tag=$tag tau=$tau (completed log)"; return 0
  fi

  echo "START seed=$seed tag=$tag tau=$tau gpu=$GPU_ID" | tee -a "$OUT_ROOT/replication_gpu3.log"
  if [ "$model" = SASRec ]; then
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --model SASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  else
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --model GeometrySASRec --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --temperature "$tau" --score_geometry "$geometry" \
      --sequence_norm_power "$seq_power" --item_norm_power "$item_power" \
      --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  fi
  grep 'test result:' "$log" | tail -1 | tee -a "$OUT_ROOT/replication_gpu3.log" || true
}

for seed in $SEEDS; do
  run_one "$seed" dot SASRec 1 dot 0 0
  run_one "$seed" sequence GeometrySASRec 5 sequence 1 0
  run_one "$seed" both GeometrySASRec 10 both 1 1
done

echo ALL_DONE | tee -a "$OUT_ROOT/replication_gpu3.log"
