#!/usr/bin/env bash
set -euo pipefail

ROOT="/code/25-huangqiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_DIR="${ROOT}/experiments/ablation_h256"
LOG_DIR="${ROOT}/log_runs/ablation_h256"
WAIT_LOG="${LOG_DIR}/ablation_h256_wait_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${EXP_DIR}" "${LOG_DIR}" "${ROOT}/ckpt/ablation_h256"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$1 != 0 && $2 < 4000 && $3 < 35 { print $1, $2, $3 }' \
    | sort -k2,2n -k3,3n \
    | awk 'NR == 1 { print $1 }'
}

GPU_ID="$(pick_gpu || true)"
while [[ -z "${GPU_ID}" ]]; do
  {
    echo "[$(date '+%F %T')] waiting for GPU 1/2; current status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  } >> "${WAIT_LOG}"
  sleep 120
  GPU_ID="$(pick_gpu || true)"
done

echo "[$(date '+%F %T')] selected gpu ${GPU_ID}" | tee -a "${WAIT_LOG}"

run_one() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] start ${name} on gpu ${GPU_ID}" | tee -a "${WAIT_LOG}"
  cd "${ROOT}"
  source "${ENV_SETUP}"
  conda run -n recbole python main.py \
    --dataset Beauty \
    --gpu_id "${GPU_ID}" \
    --hidden_size 256 \
    --inner_size 1024 \
    --eval_batch_size 1024 \
    --train_batch_size 1024 \
    "$@" \
    2>&1 | tee "${log}"
  echo "[$(date '+%F %T')] finish ${name}; log=${log}" | tee -a "${WAIT_LOG}"
}

run_one "sasrec_beauty_h256" \
  --model SASRec \
  --checkpoint_dir ./ckpt/ablation_h256/Beauty_SASRec_h256

run_one "cosine_beauty_h256" \
  --model CANDSSASRec \
  --checkpoint_dir ./ckpt/ablation_h256/Beauty_Cosine_h256 \
  --temperature 10.0 \
  --use_norm_residual False \
  --use_weighted_ce False \
  --use_cands_smoothing False

run_one "cosine_normres_beauty_h256" \
  --model CANDSSASRec \
  --checkpoint_dir ./ckpt/ablation_h256/Beauty_Cosine_NormResidual_h256 \
  --temperature 10.0 \
  --use_norm_residual True \
  --norm_beta 1.0 \
  --use_weighted_ce False \
  --use_cands_smoothing False

run_one "cosine_weightedce_beauty_h256" \
  --model CANDSSASRec \
  --checkpoint_dir ./ckpt/ablation_h256/Beauty_Cosine_WeightedCE_h256 \
  --temperature 10.0 \
  --use_norm_residual False \
  --use_weighted_ce True \
  --weight_gamma 0.10 \
  --use_cands_smoothing False

run_one "cosine_normres_weightedce_beauty_h256" \
  --model CANDSSASRec \
  --checkpoint_dir ./ckpt/ablation_h256/Beauty_Cosine_NormResidual_WeightedCE_h256 \
  --temperature 10.0 \
  --use_norm_residual True \
  --norm_beta 1.0 \
  --use_weighted_ce True \
  --weight_gamma 0.10 \
  --use_cands_smoothing False

echo "[$(date '+%F %T')] all ablation_h256 runs finished" | tee -a "${WAIT_LOG}"
