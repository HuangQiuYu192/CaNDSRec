#!/usr/bin/env bash
set -euo pipefail

# Held-out-seed confirmation of the pre-specified positive scale intervention.
# sigma=0.5 is the largest pre-defined pilot condition; it is not selected by
# development-test performance. Unjittered baselines are reused from completed
# same-seed BSARec and sequence-only runs.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
SIGMA="${SIGMA:-0.5}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_scale_jitter_confirmation}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_scale_jitter_confirmation}"

run_one() {
  local seed="$1" variant="$2" model extra=()
  if [ "$variant" = dot_jitter ]; then model=ScaleJitterBSARec; else model=ScaleJitterSequenceGeometryBSARec; extra=(--score_geometry sequence --temperature 4); fi
  local tag="${variant}_sigma${SIGMA}_d256_a0.5_c9_h2_lr0.001"
  local log="$OUT_ROOT/Beauty/seed${seed}/${tag}.log"
  local ckpt="$CKPT_ROOT/Beauty/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && return 0
  echo "START seed=$seed variant=$variant sigma=$SIGMA gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model "$model" --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 --alpha 0.5 --c 9 \
    --sequence_scale_jitter_sigma "$SIGMA" --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    "${extra[@]}" > "$log" 2>&1
}

for seed in $SEEDS_STR; do
  run_one "$seed" dot_jitter
  run_one "$seed" sequence_jitter
done
conda run --no-capture-output -n recbole python experiments/cross_dataset/collect_www27_bsarec_scale_jitter_confirmation.py --sigma "$SIGMA"
echo ALL_DONE
