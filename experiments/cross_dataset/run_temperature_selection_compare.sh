#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_CSV="${BENCHMARK_CSV:-log_runs/main_benchmark_grid/summary.csv}"
CALIBRATION_SUMMARY="${CALIBRATION_SUMMARY:-analysis_results/temperature_calibration_adaptau/summary.csv}"
OUT_DIR="${OUT_DIR:-analysis_results/temperature_selection}"
FIXED_TEMP="${FIXED_TEMP:-10}"
METHODS="${METHODS:-std margin pos_neg_gap adaptau_all adaptau_all_x0.5 adaptau_neg adaptau_neg_x0.5}"

python experiments/cross_dataset/compare_temperature_selection_methods.py \
  --benchmark_csv "$BENCHMARK_CSV" \
  --calibration_summary "$CALIBRATION_SUMMARY" \
  --fixed_temp "$FIXED_TEMP" \
  --methods "$METHODS" \
  --out_dir "$OUT_DIR"

echo
echo "Aggregate:"
cat "$OUT_DIR/aggregate.md"
