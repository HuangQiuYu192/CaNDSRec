#!/usr/bin/env bash
set -euo pipefail

# Complete paper evaluation from already trained matched checkpoints.  This
# reports global training-popularity item tertiles, not equal test-target bins.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_matched_seed_grid}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/www27_full_evaluation}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
# shellcheck source=/dev/null
source "$CONDA_SH"
cd "$ROOT"

latest_ckpt() {
  local dir="$1" model="$2"
  find "$dir" -maxdepth 1 -type f -name "${model}-*.pth" 2>/dev/null | sort | tail -n 1
}

group_eval() {
  local gpu="$1" model="$2" dataset="$3" seed="$4" checkpoint="$5"
  local prefix="$OUT_ROOT/group_metrics/$dataset/seed$seed/$model"
  if [ "$RERUN" != true ] && [ -s "$prefix.csv" ]; then return 0; fi
  mkdir -p "$(dirname "$prefix")"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --model "$model" --checkpoint "$checkpoint" --tag "${dataset}_seed${seed}_${model}" \
    --dataset "$dataset" --gpu_id "$gpu" --seed "$seed" --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 --max_item_list_length 50 \
    --train_batch_size 1024 --eval_batch_size 512 --temperature 10 --cutoffs 5,10,20,50,100 \
    --group_mode global_item_tertile --out_prefix "$prefix"
}

geometry_eval() {
  local dataset="$1" seed="$2" ckpt_dir="$3"
  local out="$OUT_ROOT/directional_calibration/${dataset}_seed${seed}_h256_len50_temp10.csv"
  if [ "$RERUN" != true ] && [ -s "$out" ]; then return 0; fi
  mkdir -p "$(dirname "$out")"
  GPU_ID=0 DATASET="$dataset" SEED="$seed" HIDDEN_SIZE=256 MAX_LEN=50 INNER_SIZE=1024 TEMPERATURE=10 \
    CKPT_DIR="$ckpt_dir" OUT_DIR="$(dirname "$out")" \
    bash experiments/cross_dataset/run_directional_calibration.sh
}

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    ckpt_dir="$CKPT_ROOT/$dataset/seed$seed"
    sasrec="$(latest_ckpt "$ckpt_dir" SASRec)"
    cands="$(latest_ckpt "$ckpt_dir" CANDSSASRec)"
    if [ -z "$sasrec" ] || [ -z "$cands" ]; then
      echo "MISSING dataset=$dataset seed=$seed SASRec=$sasrec CaNDS=$cands" >&2
      continue
    fi
    echo "EVAL dataset=$dataset seed=$seed"
    group_eval 0 SASRec "$dataset" "$seed" "$sasrec" & p0=$!
    group_eval 1 CANDSSASRec "$dataset" "$seed" "$cands" & p1=$!
    wait "$p0"; wait "$p1"
    geometry_eval "$dataset" "$seed" "$ckpt_dir"
  done
done
echo "ALL_DONE"
