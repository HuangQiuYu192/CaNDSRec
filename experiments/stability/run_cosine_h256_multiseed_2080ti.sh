#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_NAME="cosine_h256_multiseed_2080ti"
LOG_DIR="${ROOT}/log_runs/${EXP_NAME}"
CKPT_ROOT="${ROOT}/ckpt/${EXP_NAME}"

mkdir -p "${LOG_DIR}" "${CKPT_ROOT}" "${ROOT}/experiments/stability"

run_one() {
  local gpu="$1"
  local seed="$2"
  local log="${LOG_DIR}/cosine_seed${seed}.log"
  local ckpt="${CKPT_ROOT}/seed${seed}"
  mkdir -p "${ckpt}"
  {
    echo "[$(date '+%F %T')] START seed=${seed} gpu=${gpu}"
    cd "${ROOT}"
    source "${ENV_SETUP}"
    conda run --no-capture-output -n recbole python main.py \
      --model CANDSSASRec \
      --dataset Beauty \
      --gpu_id "${gpu}" \
      --seed "${seed}" \
      --hidden_size 256 \
      --inner_size 1024 \
      --train_batch_size 1024 \
      --eval_batch_size 1024 \
      --temperature 10.0 \
      --checkpoint_dir "${ckpt}"
    echo "[$(date '+%F %T')] END seed=${seed}"
  } > "${log}" 2>&1
}

run_one 0 2026 &
pid0=$!
run_one 1 2027 &
pid1=$!
run_one 3 2028 &
pid3=$!
wait "${pid0}" "${pid1}" "${pid3}"

echo "[$(date '+%F %T')] all cosine multiseed runs finished" > "${LOG_DIR}/finished.txt"
