#!/usr/bin/env bash
set -euo pipefail

# Targeted theory-validation ablation, not a large main-baseline grid. It
# deliberately uses only GPUs 0 and 1. LastFM uses its longer history, while
# Yelp supplies a distinct sparse commercial domain.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
DATASETS_STR="${DATASETS_STR:-LastFM-S3Rec Yelp-S3Rec}"
SEED="${SEED:-2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_geometry_ablation}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_geometry_ablation}"
RERUN="${RERUN:-false}"
RUN_L2_CONTROL="${RUN_L2_CONTROL:-true}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

max_len_for() { [ "$1" = "LastFM-S3Rec" ] && echo 200 || echo 50; }
batch_for() { [ "$1" = "LastFM-S3Rec" ] && echo 512 || echo 1024; }

run_one() {
  local gpu="$1" dataset="$2" tag="$3" model="$4" geometry="$5" weight_decay="$6"
  local max_len batch log ckpt extra=()
  max_len="$(max_len_for "$dataset")"; batch="$(batch_for "$dataset")"
  log="$OUT_ROOT/$dataset/seed$SEED/$tag.log"; ckpt="$CKPT_ROOT/$dataset/seed$SEED/$tag"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
    echo "SKIP $dataset $tag"; return 0
  fi
  if [ -n "$geometry" ]; then extra+=(--score_geometry "$geometry" --temperature 10); fi
  echo "START dataset=$dataset tag=$tag gpu=$gpu" | tee -a "$OUT_ROOT/launcher.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model "$model" --dataset "$dataset" --gpu_id "$gpu" --seed "$SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --weight_decay "$weight_decay" --epochs 300 --stopping_step 10 \
    --train_batch_size "$batch" --eval_batch_size 512 --max_item_list_length "$max_len" \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True "${extra[@]}" > "$log" 2>&1
  grep -A2 'test result' "$log" | tee -a "$OUT_ROOT/summary.raw" || true
}

for dataset in $DATASETS_STR; do
  run_one 0 "$dataset" dot SASRec '' 0.0 & p0=$!
  run_one 1 "$dataset" sequence GeometrySASRec sequence 0.0 & p1=$!
  wait "$p0"; wait "$p1"
  run_one 0 "$dataset" item GeometrySASRec item 0.0 & p0=$!
  run_one 1 "$dataset" both GeometrySASRec both 0.0 & p1=$!
  wait "$p0"; wait "$p1"
  if [ "$RUN_L2_CONTROL" = true ]; then
    run_one 0 "$dataset" dot_l2e4 SASRec '' 0.0001
  fi
done
echo ALL_DONE
