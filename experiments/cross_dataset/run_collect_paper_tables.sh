#!/usr/bin/env bash
set -euo pipefail

python experiments/cross_dataset/collect_backbone_hidden_trend.py \
  --dataset "${DATASET:-Beauty}" \
  --out_dir "${TREND_OUT_DIR:-analysis_results/backbone_hidden_trend}"

python experiments/cross_dataset/collect_backbone_mechanism_summary.py \
  --out_dir "${MECHANISM_OUT_DIR:-analysis_results/backbone_mechanism_summary}"

echo
echo "Backbone hidden trend:"
cat "${TREND_OUT_DIR:-analysis_results/backbone_hidden_trend}/summary.md"

echo
echo "Backbone mechanism summary:"
cat "${MECHANISM_OUT_DIR:-analysis_results/backbone_mechanism_summary}/summary.md"
