#!/usr/bin/env bash
set -euo pipefail

# Build unified paper tables from existing logs and grouped evaluation outputs.
# Default excludes ML-1M, because its AngularSmooth behavior is a boundary case.

python experiments/cross_dataset/collect_unified_main_table.py \
  --datasets ${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec LastFM-S3Rec} \
  --hidden "${HIDDEN_SIZE:-256}" \
  --out_dir "${OUT_DIR:-analysis_results/unified_main_table}"

echo
echo "Overall table:"
cat "${OUT_DIR:-analysis_results/unified_main_table}/overall_table.md"

echo
echo "Grouped table:"
cat "${OUT_DIR:-analysis_results/unified_main_table}/group_table.md"
