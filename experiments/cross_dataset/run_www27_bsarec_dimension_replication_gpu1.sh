#!/usr/bin/env bash
set -euo pipefail

# Confirm the pre-specified d=128/256 capacity--geometry interaction across
# two additional seeds. d=64 has already been completed for these seeds, and
# all 2025 checkpoints are reused from the original validation grid.  There is
# no test-time choice of dimension, temperature, or BSARec backbone setting.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024}"
DIMS_STR="${DIMS_STR:-128 256}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_dimension_replication}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_dimension_replication}"

run_one() {
  local seed="$1" dim="$2" model="$3"
  local tag="${model}_d${dim}_a0.5_c9_h2_lr0.001"
  local log="$OUT_ROOT/Beauty/seed${seed}/${tag}.log"
  local ckpt="$CKPT_ROOT/Beauty/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && return 0
  local extra=()
  if [ "$model" = "CANDSBSARec" ]; then extra=(--temperature 10); fi
  echo "START seed=$seed dimension=$dim model=$model gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model "$model" --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size "$dim" --n_layers 2 --n_heads 2 --inner_size "$((4 * dim))" \
    --alpha 0.5 --c 9 --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    "${extra[@]}" > "$log" 2>&1
}

for seed in $SEEDS_STR; do
  for dim in $DIMS_STR; do
    run_one "$seed" "$dim" BSARec
    run_one "$seed" "$dim" CANDSBSARec
  done
done
echo ALL_DONE
