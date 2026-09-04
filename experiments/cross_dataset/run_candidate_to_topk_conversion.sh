#!/usr/bin/env bash
set -euo pipefail

# Offline candidate-to-top-k conversion diagnostics from rank-transition samples.
# This script does not train or load models.

python experiments/cross_dataset/analyze_candidate_to_topk_conversion.py \
  --rank_transition_dir "${RANK_TRANSITION_DIR:-analysis_results/beauty_rank_transition}" \
  --out_dir "${OUT_DIR:-analysis_results/candidate_to_topk_conversion}" \
  --cutoffs "${CUTOFFS:-10,20,50,100}" \
  --candidate_cutoffs "${CANDIDATE_CUTOFFS:-20,50,100}" \
  --target_cutoff "${TARGET_CUTOFF:-10}" \
  --promoted_rank "${PROMOTED_RANK:-10}" \
  --shifts "${SHIFTS:-5,10,20,50,100}" \
  --scales "${SCALES:-0.25,0.5,0.75}" \
  --eligible_sets "${ELIGIBLE_SETS:-tail:tail,mid_tail:mid+tail}"

echo
cat "${OUT_DIR:-analysis_results/candidate_to_topk_conversion}/summary.md"
