#!/usr/bin/env bash
set -euo pipefail

# Build a true best-tuned table from existing logs.
# Default datasets exclude ML-1M.

python experiments/cross_dataset/collect_best_tuned_table.py \
  --datasets ${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec LastFM-S3Rec} \
  --out_dir "${OUT_DIR:-analysis_results/best_tuned_table}"

echo
cat "${OUT_DIR:-analysis_results/best_tuned_table}/best_tuned_table.md"
