#!/usr/bin/env bash
set -euo pipefail

# ML-1M-only AngularSmooth grid. Uses max_len 200 and a conservative batch size.

DATASETS_STR="${DATASETS_STR:-ML-1M}" \
ML1M_MAX_LEN="${ML1M_MAX_LEN:-200}" \
ML1M_TEMP="${ML1M_TEMP:-20}" \
GPUS_STR="${GPUS_STR:-2 3}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}" \
TAG="${TAG:-ml1m_len200_angular_smooth_grid_gpu23_bs512}" \
bash experiments/cross_dataset/run_yelp_ml1m_angular_smooth_grid_gpu23.sh
