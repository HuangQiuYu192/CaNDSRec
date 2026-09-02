#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ssh_user/code/25-HuangQiuyu/LongTailRec
ENV=/home/ssh_user/miniconda3/etc/profile.d/conda.sh
OUT="$ROOT/log_runs/lastfm_s3rec_len200_h64"
mkdir -p "$OUT"
cd "$ROOT"

source "$ENV"

run_one() {
  local model="$1"
  local temp="$2"
  local tag="LastFM-S3Rec_${model}_h64_len200"
  if [ "$model" = "CANDSSASRec" ]; then
    tag="${tag}_temp${temp}"
  fi
  local log="$OUT/${tag}.log"
  echo "[$(date '+%F %T')] START $tag gpu=0" | tee -a "$OUT/master.log"
  conda run --no-capture-output -n recbole python main.py \
    --dataset LastFM-S3Rec \
    --model "$model" \
    --gpu_id 0 \
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

run_one SASRec 10.0
run_one CANDSSASRec 10.0
run_one CANDSSASRec 15.0
run_one CANDSSASRec 20.0
run_one CANDSSASRec 30.0
