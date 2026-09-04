#!/usr/bin/env bash
set -euo pipefail

# Summarize @5/@10/@20/@50/@100 metrics from existing rank-transition samples.
# This does not reload models and can be run after run_beauty_rank_transition_gpu0.sh.

python experiments/cross_dataset/collect_rank_cutoff_metrics.py \
  --rank_transition_dir "${RANK_TRANSITION_DIR:-analysis_results/beauty_rank_transition}" \
  --out_dir "${OUT_DIR:-analysis_results/rank_cutoff_metrics}" \
  --cutoffs "${CUTOFFS:-5,10,20,50,100}"

echo
cat "${OUT_DIR:-analysis_results/rank_cutoff_metrics}/summary.md"
