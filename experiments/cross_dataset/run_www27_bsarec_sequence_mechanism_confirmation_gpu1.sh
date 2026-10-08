#!/usr/bin/env bash
set -euo pipefail

# Held-out-seed causal comparison. tau=4 is fixed by the independent seed-2026
# sequence-only validation screen.  GeometryBSARec and its stop-gradient
# counterpart have the same forward score at a fixed parameter state; only the
# gradient through the sequence norm differs.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
DIM="${DIM:-256}"
TEMPERATURE="${TEMPERATURE:-4}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_sequence_mechanism_confirmation}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_sequence_mechanism_confirmation}"

run_one() {
  local seed="$1" model="$2"
  local tag="${model}_sequence_d${DIM}_tau${TEMPERATURE}_a0.5_c9_h2_lr0.001"
  local log="$OUT_ROOT/Beauty/seed${seed}/${tag}.log"
  local ckpt="$CKPT_ROOT/Beauty/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && return 0
  echo "START seed=$seed d=$DIM model=$model geometry=sequence tau=$TEMPERATURE gpu=$GPU_ID"
  conda run --no-capture-output -n recbole python main.py \
    --model "$model" --score_geometry sequence --dataset Beauty --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size "$DIM" --n_layers 2 --n_heads 2 --inner_size "$((4 * DIM))" --alpha 0.5 --c 9 \
    --temperature "$TEMPERATURE" --learning_rate 0.001 --weight_decay 0 --epochs 300 --stopping_step 10 \
    --train_batch_size 1024 --eval_batch_size 512 --max_item_list_length 50 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}

for seed in $SEEDS_STR; do
  run_one "$seed" GeometryBSARec
  run_one "$seed" StopGradSequenceGeometryBSARec
done

conda run --no-capture-output -n recbole python \
  experiments/cross_dataset/collect_www27_bsarec_sequence_mechanism_confirmation.py \
  --temperature "$TEMPERATURE"
echo ALL_DONE
