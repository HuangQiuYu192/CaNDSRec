#!/usr/bin/env bash
set -euo pipefail

# Replicate the externally observed score-geometry pattern before using it in
# the paper. Each child invocation uses only GPUs 0 and 1; seeds run
# sequentially to keep the allocation unambiguous.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
SEEDS_STR="${SEEDS_STR:-2023 2024}"
DATASETS_STR="${DATASETS_STR:-LastFM-S3Rec Yelp-S3Rec}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_geometry_ablation}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_geometry_ablation}"

cd "$ROOT"
for seed in $SEEDS_STR; do
  echo "START replication seed=$seed"
  SEED="$seed" DATASETS_STR="$DATASETS_STR" OUT_ROOT="$OUT_ROOT" CKPT_ROOT="$CKPT_ROOT" \
    RUN_L2_CONTROL=false bash experiments/cross_dataset/run_www27_geometry_ablation.sh
  echo "DONE replication seed=$seed"
done
echo ALL_DONE
