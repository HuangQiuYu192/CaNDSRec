#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
cd "$ROOT"

source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID="${GPU_ID:-0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEED="${SEED:-2025}"
TAG="${TAG:-small}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys}"
TEMPS_STR="${TEMPS_STR:-5 7.5 10 15 20 30 40}"
LOG_DIR="log_runs/temp_law_${TAG}_h${HIDDEN_SIZE}"
CKPT_DIR="ckpt/temp_law_${TAG}_h${HIDDEN_SIZE}"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

for dataset in ${DATASETS_STR}; do
  if [ "$dataset" = "ML-1M" ]; then
    max_len="${MAX_LEN:-200}"
  else
    max_len="${MAX_LEN:-50}"
  fi

  base_name="${dataset}_SASRec_h${HIDDEN_SIZE}_len${max_len}"
  base_log="${LOG_DIR}/${base_name}.log"
  if ! grep -q "test result" "$base_log" 2>/dev/null; then
    echo "===== $(date '+%F %T') START ${base_name} gpu=${GPU_ID} =====" | tee -a "${LOG_DIR}/master.log"
    conda run --no-capture-output -n recbole python main.py \
      --dataset "$dataset" \
      --model SASRec \
      --gpu_id "$GPU_ID" \
      --seed "$SEED" \
      --hidden_size "$HIDDEN_SIZE" \
      --max_item_list_length "$max_len" \
      --train_batch_size 1024 \
      --eval_batch_size 1024 \
      --epochs 300 \
      --stopping_step 10 \
      --show_progress True \
      --verbose True \
      --checkpoint_dir "./${CKPT_DIR}/${base_name}" \
      > "$base_log" 2>&1
    grep -E "best valid result|test result" "$base_log" >> "${LOG_DIR}/summary.raw" || true
    echo "===== $(date '+%F %T') DONE ${base_name} =====" | tee -a "${LOG_DIR}/master.log"
  fi

  for temp in ${TEMPS_STR}; do
    run_name="${dataset}_CANDSSASRec_h${HIDDEN_SIZE}_len${max_len}_temp${temp}"
    log_file="${LOG_DIR}/${run_name}.log"
    if grep -q "test result" "$log_file" 2>/dev/null; then
      echo "===== $(date '+%F %T') SKIP ${run_name} =====" | tee -a "${LOG_DIR}/master.log"
      continue
    fi
    echo "===== $(date '+%F %T') START ${run_name} gpu=${GPU_ID} =====" | tee -a "${LOG_DIR}/master.log"
    conda run --no-capture-output -n recbole python main.py \
      --dataset "$dataset" \
      --model CANDSSASRec \
      --gpu_id "$GPU_ID" \
      --seed "$SEED" \
      --hidden_size "$HIDDEN_SIZE" \
      --temperature "$temp" \
      --max_item_list_length "$max_len" \
      --train_batch_size 1024 \
      --eval_batch_size 1024 \
      --epochs 300 \
      --stopping_step 10 \
      --show_progress True \
      --verbose True \
      --checkpoint_dir "./${CKPT_DIR}/${run_name}" \
      > "$log_file" 2>&1
    grep -E "best valid result|test result" "$log_file" >> "${LOG_DIR}/summary.raw" || true
    echo "===== $(date '+%F %T') DONE ${run_name} =====" | tee -a "${LOG_DIR}/master.log"
  done
done

echo "all_done $(date '+%F %T')" | tee -a "${LOG_DIR}/master.log"
