#!/usr/bin/env bash
set -euo pipefail

# Read-only follow-up to the geometry replication. The `both` GeometrySASRec
# checkpoint has the same parameters and score function as CANDSSASRec, so it
# is deliberately loaded by CANDSSASRec to keep the proposed-model comparison
# explicit. This runner never trains or selects a checkpoint.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-LastFM-S3Rec Yelp-S3Rec}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_geometry_ablation}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/www27_radial_displacement}"

source "$CONDA_SH"
cd "$ROOT"
max_len_for() { [ "$1" = "LastFM-S3Rec" ] && echo 200 || echo 50; }
batch_for() { [ "$1" = "LastFM-S3Rec" ] && echo 512 || echo 1024; }
# Two checkpoints are simultaneously resident during the paired analysis.
# Long LastFM histories therefore need a smaller inference batch than training.
eval_batch_for() { [ "$1" = "LastFM-S3Rec" ] && echo 128 || echo 256; }

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    out="$OUT_ROOT/$dataset/seed$seed"
    [ -f "$out.csv" ] && { echo "SKIP $dataset seed=$seed"; continue; }
    base=$(find "$CKPT_ROOT/$dataset/seed$seed/dot" -maxdepth 1 -type f -name 'SASRec-*.pth' | head -n 1)
    both=$(find "$CKPT_ROOT/$dataset/seed$seed/both" -maxdepth 1 -type f -name 'GeometrySASRec-*.pth' | head -n 1)
    if [ -z "$base" ] || [ -z "$both" ]; then echo "Missing checkpoints for $dataset seed=$seed" >&2; exit 2; fi
    mkdir -p "$(dirname "$out")"
    conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_radial_displacement_subgroups.py \
      --dataset "$dataset" --seed "$seed" --gpu_id 0 --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --max_item_list_length "$(max_len_for "$dataset")" --train_batch_size "$(batch_for "$dataset")" \
      --eval_batch_size "$(eval_batch_for "$dataset")" --temperature 10 --base_checkpoint "$base" --cands_checkpoint "$both" --out "$out"
  done
done
echo ALL_DONE
