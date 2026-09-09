#!/usr/bin/env bash
set -euo pipefail

# Targeted replication of the LastFM temperature discrepancy.  The prior
# screen used seed 2025 and found that validation chose tau=5 while test
# NDCG@10 was higher at tau=10.  This script adds only seeds 2023 and 2024,
# keeping the comparison paired across score geometries and temperatures.
# It deliberately uses GPUs 0 and 1 only.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
SEEDS_STR="${SEEDS_STR:-2023 2024}"
TEMPS_STR="${TEMPS_STR:-5 10}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_lastfm_temperature_replication}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_lastfm_temperature_replication}"

source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local gpu="$1" seed="$2" geometry="$3" temp="$4"
  local tag="${geometry}_tau${temp}"
  local log="$OUT_ROOT/LastFM-S3Rec/seed${seed}/${tag}.log"
  local ckpt="$CKPT_ROOT/LastFM-S3Rec/seed${seed}/${tag}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if grep -q 'MODEL TEST' "$log" 2>/dev/null; then echo "SKIP seed=$seed $tag"; return 0; fi
  echo "START seed=$seed geometry=$geometry tau=$temp gpu=$gpu" | tee -a "$OUT_ROOT/launcher.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --score_geometry "$geometry" --temperature "$temp" \
    --dataset LastFM-S3Rec --gpu_id "$gpu" --seed "$seed" --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 --weight_decay 0.0 \
    --epochs 300 --stopping_step 10 --train_batch_size 512 --eval_batch_size 512 --max_item_list_length 200 \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  grep -E 'best valid result:|test result:' "$log" | tail -n 2 | tee -a "$OUT_ROOT/summary.raw" || true
}

for seed in $SEEDS_STR; do
  for temp in $TEMPS_STR; do
    run_one 0 "$seed" sequence "$temp" & p0=$!
    run_one 1 "$seed" both "$temp" & p1=$!
    wait "$p0"; wait "$p1"
  done
done
echo ALL_DONE
