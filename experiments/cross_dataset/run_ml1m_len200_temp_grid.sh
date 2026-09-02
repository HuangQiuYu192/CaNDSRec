#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
cd "$ROOT"

source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID="${GPU_ID:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
MAX_LEN="${MAX_LEN:-200}"
SEED="${SEED:-2025}"
LOG_DIR="log_runs/ml1m_len${MAX_LEN}_temp_grid_h${HIDDEN_SIZE}"
CKPT_DIR="ckpt/ml1m_len${MAX_LEN}_temp_grid_h${HIDDEN_SIZE}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

temps=(2 5 7.5 15 20 30 40)

for temp in "${temps[@]}"; do
  run_name="ML-1M_len${MAX_LEN}_CANDSSASRec_h${HIDDEN_SIZE}_temp${temp}"
  log_file="${LOG_DIR}/${run_name}.log"
  echo "===== $(date '+%F %T') START ${run_name} gpu=${GPU_ID} =====" | tee -a "${LOG_DIR}/master.log"
  conda run --no-capture-output -n recbole python main.py \
    --dataset ML-1M \
    --model CANDSSASRec \
    --gpu_id "${GPU_ID}" \
    --seed "${SEED}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --temperature "${temp}" \
    --max_item_list_length "${MAX_LEN}" \
    --train_batch_size 1024 \
    --eval_batch_size 1024 \
    --epochs 300 \
    --stopping_step 10 \
    --show_progress True \
    --verbose True \
    --checkpoint_dir "./${CKPT_DIR}/${run_name}" \
    > "${log_file}" 2>&1
  grep -E "best valid result|test result" "${log_file}" >> "${LOG_DIR}/summary.raw" || true
  echo "===== $(date '+%F %T') DONE ${run_name} =====" | tee -a "${LOG_DIR}/master.log"
done

echo "all_done $(date '+%F %T')" | tee -a "${LOG_DIR}/master.log"
