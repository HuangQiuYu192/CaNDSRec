#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
cd "$ROOT"

source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

HIDDEN_SIZE="${HIDDEN_SIZE:?set HIDDEN_SIZE, e.g. 64 or 256}"
GPU_ID="${GPU_ID:?set GPU_ID}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"
TEMPS="${TEMPS:-2 5 7.5 15 20 30 40}"

OUT="log_runs/ml1m_len50_temp_grid_h${HIDDEN_SIZE}"
mkdir -p "$OUT" "ckpt/ml1m_len50_temp_grid_h${HIDDEN_SIZE}"

for temp in $TEMPS; do
  tag="ML-1M_len50_CANDSSASRec_h${HIDDEN_SIZE}_temp${temp}"
  log="$OUT/${tag}.log"
  echo "===== $(date '+%F %T') START ${tag} gpu=${GPU_ID} =====" | tee -a "$OUT/master.log"
  conda run --no-capture-output -n recbole python main.py \
    --dataset "ML-1M" \
    --model "CANDSSASRec" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --temperature "$temp" \
    --max_item_list_length 50 \
    --train_batch_size 1024 \
    --eval_batch_size 1024 \
    --epochs "$EPOCHS" \
    --stopping_step 10 \
    --show_progress True \
    --verbose True \
    --checkpoint_dir "./ckpt/ml1m_len50_temp_grid_h${HIDDEN_SIZE}/${tag}" \
    > "$log" 2>&1
  echo "===== $(date '+%F %T') DONE ${tag} =====" | tee -a "$OUT/master.log"
  grep -E "best valid result|test result" "$log" | tail -2 | tee -a "$OUT/summary.raw"
done

echo "all_done $(date '+%F %T')" | tee -a "$OUT/master.log"
