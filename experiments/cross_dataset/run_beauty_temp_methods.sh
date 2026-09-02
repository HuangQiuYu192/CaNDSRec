#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
cd "$ROOT"

source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
SEED="${SEED:-2025}"
MAX_LEN="${MAX_LEN:-50}"
LOG_DIR="log_runs/beauty_temp_methods_h${HIDDEN_SIZE}"
CKPT_DIR="ckpt/beauty_temp_methods_h${HIDDEN_SIZE}"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

pick_gpu() {
  while true; do
    for gpu in 2 3; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      if [ "$used" -lt 1000 ] && [ "$util" -lt 20 ]; then
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date '+%F %T')] waiting for GPU 2/3 to become idle" | tee -a "${LOG_DIR}/master.log" >&2
    sleep 300
  done
}

run_one() {
  local model="$1"
  local tag="$2"
  shift 2
  local gpu
  gpu=$(pick_gpu)
  local run_name="Beauty_${tag}_h${HIDDEN_SIZE}"
  local log_file="${LOG_DIR}/${run_name}.log"
  echo "===== $(date '+%F %T') START ${run_name} gpu=${gpu} =====" | tee -a "${LOG_DIR}/master.log"
  conda run --no-capture-output -n recbole python main.py \
    --dataset Beauty \
    --model "${model}" \
    --gpu_id "${gpu}" \
    --seed "${SEED}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --max_item_list_length "${MAX_LEN}" \
    --train_batch_size 1024 \
    --eval_batch_size 1024 \
    --epochs 300 \
    --stopping_step 10 \
    --show_progress True \
    --verbose True \
    --checkpoint_dir "./${CKPT_DIR}/${run_name}" \
    "$@" \
    > "${log_file}" 2>&1
  grep -E "best valid result|test result" "${log_file}" >> "${LOG_DIR}/summary.raw" || true
  echo "===== $(date '+%F %T') DONE ${run_name} =====" | tee -a "${LOG_DIR}/master.log"
}

run_one CANDSSASRec fixed_temp10 --temperature 10
run_one LearnableTempCANDSSASRec learnable_init10 --temperature 10 --temp_reg_weight 0.0
run_one DataAwareTempCANDSSASRec dataaware_learnable --temperature 10 --temp_reg_weight 0.0 --data_temp_scale 3.0 --data_temp_min 1.0 --data_temp_max 100.0

echo "all_done $(date '+%F %T')" | tee -a "${LOG_DIR}/master.log"
