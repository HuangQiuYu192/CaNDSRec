#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ssh_user/code/25-HuangQiuyu/LongTailRec
ENV=/home/ssh_user/miniconda3/etc/profile.d/conda.sh
OUT="$ROOT/log_runs/lastfm_s3rec_len200_temp_fine_h64"
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local temp="$1"
  local gpu="$2"
  local tag="LastFM-S3Rec_CANDSSASRec_h64_len200_temp${temp}"
  local log="$OUT/${tag}.log"
  echo "[$(date '+%F %T')] START $tag gpu=$gpu" | tee -a "$OUT/master.log"
  source "$ENV"
  conda run --no-capture-output -n recbole python main.py \
    --dataset LastFM-S3Rec \
    --model CANDSSASRec \
    --gpu_id "$gpu" \
    --seed 2025 \
    --hidden_size 64 \
    --n_layers 2 \
    --n_heads 2 \
    --inner_size 256 \
    --hidden_dropout_prob 0.5 \
    --attn_dropout_prob 0.5 \
    --learning_rate 0.001 \
    --epochs 300 \
    --stopping_step 10 \
    --eval_batch_size 1024 \
    --train_batch_size 1024 \
    --max_item_list_length 200 \
    --temperature "$temp" \
    --verbose True \
    --show_progress True > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$OUT/master.log"
  grep -A2 'test result' "$log" | tee -a "$OUT/summary.raw" || true
}

run_low() {
  for temp in 12.0 13.0 14.0 16.0 17.0 18.0; do
    run_one "$temp" 0
  done
}

run_high() {
  for temp in 22.0 24.0 26.0 28.0 32.0 35.0; do
    run_one "$temp" 1
  done
}

run_low &
echo $! > "$OUT/low_queue.pid"
run_high &
echo $! > "$OUT/high_queue.pid"
wait
