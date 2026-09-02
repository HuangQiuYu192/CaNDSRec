#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ssh_user/code/25-HuangQiuyu/LongTailRec
ENV=/home/ssh_user/miniconda3/etc/profile.d/conda.sh
OUT="$ROOT/log_runs/s3rec_yelp_lastfm_cands_h64"
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local dataset="$1"
  local model="$2"
  local gpu="$3"
  local temp="$4"
  local tag="${dataset}_${model}_h64_len50"
  if [ "$model" = "CANDSSASRec" ]; then
    tag="${tag}_temp${temp}"
  fi
  local log="$OUT/${tag}.log"
  echo "[$(date '+%F %T')] START $tag gpu=$gpu" | tee -a "$OUT/master.log"
  source "$ENV"
  conda run --no-capture-output -n recbole python main.py \
    --dataset "$dataset" \
    --model "$model" \
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
    --max_item_list_length 50 \
    --temperature "$temp" \
    --verbose True \
    --show_progress True > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$OUT/master.log"
  grep -A2 'test result' "$log" | tee -a "$OUT/summary.raw" || true
}

run_yelp() {
  run_one Yelp-S3Rec SASRec 0 10.0
  run_one Yelp-S3Rec CANDSSASRec 0 10.0
  run_one Yelp-S3Rec CANDSSASRec 0 20.0
}

run_lastfm() {
  run_one LastFM-S3Rec SASRec 1 10.0
  run_one LastFM-S3Rec CANDSSASRec 1 10.0
  run_one LastFM-S3Rec CANDSSASRec 1 20.0
}

run_yelp &
echo $! > "$OUT/yelp_queue.pid"
run_lastfm &
echo $! > "$OUT/lastfm_queue.pid"
wait
