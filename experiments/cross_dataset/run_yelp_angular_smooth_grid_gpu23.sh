#!/usr/bin/env bash
set -euo pipefail

# Yelp-S3Rec-only AngularSmooth grid. Uses batch size 1024 as requested.

DATASETS_STR="${DATASETS_STR:-Yelp-S3Rec}" \
YELP_MAX_LEN="${YELP_MAX_LEN:-50}" \
YELP_TEMP="${YELP_TEMP:-10}" \
GPUS_STR="${GPUS_STR:-2 3}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}" \
TAG="${TAG:-yelp_angular_smooth_grid_gpu23_bs1024}" \
bash experiments/cross_dataset/run_yelp_ml1m_angular_smooth_grid_gpu23.sh
