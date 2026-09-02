#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
cd "$ROOT"

source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

OUT="log_runs/ml1m_len50"
mkdir -p "$OUT" ckpt/ml1m_len50

GPU_ID="${GPU_ID:-2}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"

run_one() {
  local model="$1"
  local hidden="$2"
  local tag="ML-1M_len50_${model}_h${hidden}"
  local log="$OUT/${tag}.log"

  echo "===== $(date '+%F %T') START ${tag} =====" | tee -a "$OUT/master.log"
  conda run --no-capture-output -n recbole python main.py \
    --dataset "ML-1M" \
    --model "$model" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$hidden" \
    --max_item_list_length 50 \
    --train_batch_size 1024 \
    --eval_batch_size 1024 \
    --epochs "$EPOCHS" \
    --stopping_step 10 \
    --show_progress True \
    --verbose True \
    --checkpoint_dir "./ckpt/ml1m_len50/${tag}" \
    > "$log" 2>&1
  echo "===== $(date '+%F %T') DONE ${tag} =====" | tee -a "$OUT/master.log"
  grep -E "best valid result|test result" "$log" | tail -2 | tee -a "$OUT/summary.raw"
}

run_one "SASRec" 64
run_one "CANDSSASRec" 64
run_one "SASRec" 256
run_one "CANDSSASRec" 256

echo "all_done $(date '+%F %T')" | tee -a "$OUT/master.log"
