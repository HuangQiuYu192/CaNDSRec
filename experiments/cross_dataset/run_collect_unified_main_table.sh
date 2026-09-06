#!/usr/bin/env bash
set -euo pipefail

# Build unified paper tables from existing logs and grouped evaluation outputs.
# Default excludes ML-1M, because its AngularSmooth behavior is a boundary case.

python experiments/cross_dataset/collect_unified_main_table.py \
  --datasets ${DATASETS_STR:-Beauty Sports Toys Yelp-S3Rec LastFM-S3Rec} \
  --hidden "${HIDDEN_SIZE:-256}" \
  --out_dir "${OUT_DIR:-analysis_results/unified_main_table}"

echo
echo "Compact overall table:"
cat "${OUT_DIR:-analysis_results/unified_main_table}/compact_overall_table.md"

echo
echo "Compact tail delta table:"
cat "${OUT_DIR:-analysis_results/unified_main_table}/compact_tail_delta_table.md"

echo
echo "Full tables are saved under ${OUT_DIR:-analysis_results/unified_main_table}/"
