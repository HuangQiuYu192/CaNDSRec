#!/usr/bin/env bash
set -euo pipefail

# Cross-encoder validation for the score-geometry claim.  The BSARec backbone
# configuration (d=64, alpha=.5, c=9, two heads, lr=.001) was selected by the
# BSARec-only validation grid before this script is run.  Both models use this
# identical configuration; only the recommendation head differs.  Seed 2025
# is deliberately reused from the earlier capacity validation rather than
# retrained.  This script adds the two independent held-out seeds on GPU 1.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_geometry_replication}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_geometry_replication}"

run_one() {
  local seed="$1" model="$2"
  local tag="${model}_d64_a0.5_c9_h2_lr0.001"
  local log="$OUT_ROOT/Beauty/seed${seed}/${tag}.log"
  local ckpt="$CKPT_ROOT/Beauty/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && return 0

  local extra=()
  if [ "$model" = "CANDSBSARec" ]; then
    extra=(--temperature 10)
  fi
  echo "START seed=${seed} model=${model} gpu=${GPU_ID}"
  conda run --no-capture-output -n recbole python main.py \
    --model "$model" --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 64 --n_layers 2 --n_heads 2 --inner_size 256 \
    --alpha 0.5 --c 9 --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    "${extra[@]}" > "$log" 2>&1
}

for seed in $SEEDS_STR; do
  run_one "$seed" BSARec
  run_one "$seed" CANDSBSARec
done
echo ALL_DONE
