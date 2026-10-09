#!/usr/bin/env bash
set -euo pipefail

# Development-only causal intervention. Positive per-sequence jitter changes a
# dot score's sample temperature but cancels exactly in a sequence-normalized
# score. No development-seed test value is used in later reporting.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2026}"
SIGMAS_STR="${SIGMAS_STR:-0 0.25 0.5}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_scale_jitter_pilot}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_scale_jitter_pilot}"

run_one() {
  local variant="$1" sigma="$2" model tau extra=()
  if [ "$variant" = dot ]; then model=ScaleJitterBSARec; else model=ScaleJitterSequenceGeometryBSARec; tau=4; extra=(--score_geometry sequence --temperature "$tau"); fi
  local tag="${variant}_sigma${sigma}_d256_a0.5_c9_h2_lr0.001"
  local log="$OUT_ROOT/Beauty/seed${SEED}/${tag}.log"
  local ckpt="$CKPT_ROOT/Beauty/seed${SEED}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && return 0
  echo "START seed=$SEED variant=$variant sigma=$sigma gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model "$model" --dataset Beauty --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 --alpha 0.5 --c 9 \
    --sequence_scale_jitter_sigma "$sigma" --learning_rate 0.001 --weight_decay 0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True \
    "${extra[@]}" > "$log" 2>&1
}

for sigma in $SIGMAS_STR; do
  run_one dot "$sigma"
  run_one sequence "$sigma"
done
conda run --no-capture-output -n recbole python experiments/cross_dataset/collect_www27_bsarec_scale_jitter_pilot.py
echo ALL_DONE
