#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ssh_user/code/25-HuangQiuyu/LongTailRec
ENV=/home/ssh_user/miniconda3/etc/profile.d/conda.sh
OUT="$ROOT/analysis_results/rank_distribution_cross_dataset"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"
cd "$ROOT"
source "$ENV"

run_rank() {
  local label="$1"
  local checkpoint="$2"
  local dataset="$3"
  local model="$4"
  local max_len="$5"
  local temp="$6"
  local hidden="$7"
  echo "[$(date '+%F %T')] START $label" | tee -a "$OUT/master.log"
  conda run --no-capture-output -n recbole python analyze_rank_distribution.py \
    --checkpoint "$checkpoint" \
    --label "$label" \
    --output_dir "$OUT" \
    --dataset "$dataset" \
    --model "$model" \
    --gpu_id 0 \
    --seed 2025 \
    --hidden_size "$hidden" \
    --n_layers 2 \
    --n_heads 2 \
    --inner_size 256 \
    --hidden_dropout_prob 0.5 \
    --attn_dropout_prob 0.5 \
    --learning_rate 0.001 \
    --eval_batch_size 1024 \
    --train_batch_size 1024 \
    --max_item_list_length "$max_len" \
    --temperature "$temp" \
    --verbose True \
    --show_progress False > "$LOG/${label}.log" 2>&1
  echo "[$(date '+%F %T')] DONE $label" | tee -a "$OUT/master.log"
}

run_rank Beauty_sasrec_h64_len50 \
  ckpt/temp_law_small_h64/Beauty_SASRec_h64_len50/SASRec-Aug-31-2026_11-29-43.pth \
  Beauty SASRec 50 10.0 64
run_rank Beauty_cands_h64_len50_temp10 \
  ckpt/temp_law_small_h64/Beauty_CANDSSASRec_h64_len50_temp10/CANDSSASRec-Aug-31-2026_11-46-56.pth \
  Beauty CANDSSASRec 50 10.0 64
run_rank YelpS3_sasrec_h64_len50 \
  ckpt/Yelp-S3Rec/SASRec-Sep-02-2026_09-44-10.pth \
  Yelp-S3Rec SASRec 50 10.0 64
run_rank YelpS3_cands_h64_len50_temp10 \
  ckpt/Yelp-S3Rec/CANDSSASRec-Sep-02-2026_09-47-45.pth \
  Yelp-S3Rec CANDSSASRec 50 10.0 64
run_rank ML1M_sasrec_h64_len50 \
  ckpt/ml1m_len50/ML-1M_len50_SASRec_h64/SASRec-Aug-29-2026_09-35-17.pth \
  ML-1M SASRec 50 10.0 64
run_rank ML1M_cands_h64_len50_temp20 \
  ckpt/ml1m_len50_temp_grid_h64/ML-1M_len50_CANDSSASRec_h64_temp20/CANDSSASRec-Aug-30-2026_16-17-27.pth \
  ML-1M CANDSSASRec 50 20.0 64

cp analysis_results/lastfm_rank_distribution_len200/sasrec_len200.summary.json "$OUT/LastFM_sasrec_h64_len200.summary.json"
cp analysis_results/lastfm_rank_distribution_len200/cands_temp30_len200.summary.json "$OUT/LastFM_cands_h64_len200_temp30.summary.json"
cp analysis_results/lastfm_rank_distribution_len200/sasrec_len200.ranks.tsv "$OUT/LastFM_sasrec_h64_len200.ranks.tsv"
cp analysis_results/lastfm_rank_distribution_len200/cands_temp30_len200.ranks.tsv "$OUT/LastFM_cands_h64_len200_temp30.ranks.tsv"

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$OUT/master.log"
