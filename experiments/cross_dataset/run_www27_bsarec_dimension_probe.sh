#!/usr/bin/env bash
set -euo pipefail

# Stage 1: isolate representation dimension before searching BSARec's other
# published hyperparameters. All runs use the same BSARec centre setting and
# differ only in D and the final dot-versus-cosine scoring geometry.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Yelp-S3Rec}"
DIMS_STR="${DIMS_STR:-64 128 256}"
SEED="${SEED:-2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_bsarec_dimension_probe}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_dimension_probe}"

source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local gpu="$1" dataset="$2" dim="$3" model="$4"
  local out="$OUT_ROOT/$dataset/seed$SEED/${model}_d$dim.log"
  local ckpt="$CKPT_ROOT/$dataset/seed$SEED/${model}_d$dim"
  mkdir -p "$(dirname "$out")" "$ckpt"
  if grep -q 'MODEL TEST' "$out" 2>/dev/null; then echo "SKIP $dataset $model D=$dim"; return 0; fi
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model "$model" --dataset "$dataset" --gpu_id "$gpu" --seed "$SEED" \
    --hidden_size "$dim" --n_layers 2 --n_heads 2 --inner_size $((dim * 4)) \
    --c 5 --alpha 0.5 --temperature 10 --learning_rate 0.001 --weight_decay 0.0 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$out" 2>&1
}

for dataset in $DATASETS_STR; do
  for dim in $DIMS_STR; do
    run_one 0 "$dataset" "$dim" BSARec & p0=$!
    run_one 1 "$dataset" "$dim" CANDSBSARec & p1=$!
    wait "$p0"; wait "$p1"
  done
done
echo ALL_DONE
