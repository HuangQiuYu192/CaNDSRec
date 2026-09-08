#!/usr/bin/env bash
set -euo pipefail

# Validation-driven temperature screen for the two mechanisms that remained
# competitive in the three-seed decomposition. This is a diagnostic screen,
# not a test-set selection procedure: each run chooses its checkpoint by
# validation NDCG@10 and the collector must select tau using only that value.
# Only GPUs 0 and 1 are referenced.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-LastFM-S3Rec Yelp-S3Rec}"
TEMPS_STR="${TEMPS_STR:-5 10 20}"
SEED="${SEED:-2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_geometry_temperature_screen}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_geometry_temperature_screen}"

source "$CONDA_SH"
cd "$ROOT"
max_len_for() { [ "$1" = "LastFM-S3Rec" ] && echo 200 || echo 50; }
batch_for() { [ "$1" = "LastFM-S3Rec" ] && echo 512 || echo 1024; }

run_one() {
  local gpu="$1" dataset="$2" geometry="$3" temp="$4"
  local tag="${geometry}_tau${temp}" max_len batch log ckpt
  max_len="$(max_len_for "$dataset")"; batch="$(batch_for "$dataset")"
  log="$OUT_ROOT/$dataset/seed$SEED/$tag.log"; ckpt="$CKPT_ROOT/$dataset/seed$SEED/$tag"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if grep -q 'MODEL TEST' "$log" 2>/dev/null; then echo "SKIP $dataset $tag"; return 0; fi
  echo "START dataset=$dataset geometry=$geometry tau=$temp gpu=$gpu" | tee -a "$OUT_ROOT/launcher.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --score_geometry "$geometry" --temperature "$temp" \
    --dataset "$dataset" --gpu_id "$gpu" --seed "$SEED" --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 --weight_decay 0.0 \
    --epochs 300 --stopping_step 10 --train_batch_size "$batch" --eval_batch_size 512 \
    --max_item_list_length "$max_len" --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  grep -E 'best valid result:|test result:' "$log" | tail -n 2 | tee -a "$OUT_ROOT/summary.raw" || true
}

for dataset in $DATASETS_STR; do
  for temp in $TEMPS_STR; do
    run_one 0 "$dataset" sequence "$temp" & p0=$!
    run_one 1 "$dataset" both "$temp" & p1=$!
    wait "$p0"; wait "$p1"
  done
done
echo ALL_DONE
