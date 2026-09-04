#!/usr/bin/env bash
set -euo pipefail

# Rank-transition diagnostics for Beauty. Default uses only GPU 0.
# It explains whether CaNDS moves targets across the top-10 boundary or only
# improves their ranks outside the metric cutoff.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
BACKBONES_STR="${BACKBONES_STR:-SASRec WEARec FMLPRec}"
HIDDEN_SIZES_STR="${HIDDEN_SIZES_STR:-64 128 256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
WEAREC_NUM_HEADS="${WEAREC_NUM_HEADS:-1}"
WEAREC_ALPHA="${WEAREC_ALPHA:-0.8}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
MAX_BATCHES="${MAX_BATCHES:-}"

MAIN_CKPT_DIR="${MAIN_CKPT_DIR:-$ROOT/ckpt/main_benchmark_grid}"
SASREC_H128_CKPT_DIR="${SASREC_H128_CKPT_DIR:-$ROOT/ckpt/beauty_sasrec_h128_grid_gpu0}"
WEAREC_CKPT_DIR="${WEAREC_CKPT_DIR:-$ROOT/ckpt/beauty_wearec_cands_grid}"
FMLPREC_CKPT_DIR="${FMLPREC_CKPT_DIR:-$ROOT/ckpt/beauty_fmlprec_cands_grid_gpu0}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/beauty_rank_transition}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/beauty_rank_transition_gpu0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

inner_size_for_hidden() {
  echo $(( "$1" * 4 ))
}

latest_checkpoint_in_dir() {
  local ckpt_root="$1"
  local run_name="$2"
  find "$ckpt_root/$run_name" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | sort | tail -n 1
}

base_model_for_backbone() {
  case "$1" in
    SASRec) echo "SASRec" ;;
    WEARec) echo "WEARec" ;;
    FMLPRec) echo "FMLPRec" ;;
    *) echo "ERROR: unsupported backbone $1" >&2; return 1 ;;
  esac
}

cands_model_for_backbone() {
  case "$1" in
    SASRec) echo "CANDSSASRec" ;;
    WEARec) echo "CANDSWEARec" ;;
    FMLPRec) echo "CANDSFMLPRec" ;;
    *) echo "ERROR: unsupported backbone $1" >&2; return 1 ;;
  esac
}

checkpoint_for_run() {
  local backbone="$1"
  local run_name="$2"
  local ckpt=""
  case "$backbone" in
    SASRec)
      ckpt="$(latest_checkpoint_in_dir "$MAIN_CKPT_DIR" "$run_name")"
      if [ -z "$ckpt" ]; then
        ckpt="$(latest_checkpoint_in_dir "$SASREC_H128_CKPT_DIR" "$run_name")"
      fi
      ;;
    WEARec)
      ckpt="$(latest_checkpoint_in_dir "$WEAREC_CKPT_DIR" "$run_name")"
      ;;
    FMLPRec)
      ckpt="$(latest_checkpoint_in_dir "$FMLPREC_CKPT_DIR" "$run_name")"
      ;;
  esac
  echo "$ckpt"
}

run_one() {
  local backbone="$1"
  local hidden_size="$2"
  local base_model
  local cands_model
  local inner_size
  base_model="$(base_model_for_backbone "$backbone")"
  cands_model="$(cands_model_for_backbone "$backbone")"
  inner_size="$(inner_size_for_hidden "$hidden_size")"

  local base_run="${DATASET}_${base_model}_h${hidden_size}_len${MAX_LEN}"
  local cands_run="${DATASET}_${cands_model}_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}"
  local base_ckpt
  local cands_ckpt
  base_ckpt="$(checkpoint_for_run "$backbone" "$base_run")"
  cands_ckpt="$(checkpoint_for_run "$backbone" "$cands_run")"

  if [ -z "$base_ckpt" ] || [ -z "$cands_ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING ${backbone} h${hidden_size}: base=$base_ckpt cands=$cands_ckpt" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  local out_prefix="$OUT_DIR/${DATASET}_${backbone}_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}"
  local log_file="$LOG_DIR/${DATASET}_${backbone}_h${hidden_size}_len${MAX_LEN}_temp${TEMPERATURE}.log"
  if [ -s "${out_prefix}.summary.csv" ]; then
    echo "[$(date '+%F %T')] SKIP ${backbone} h${hidden_size}" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START ${backbone} h${hidden_size} gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  extra_args=()
  if [ -n "$MAX_BATCHES" ]; then
    extra_args+=(--max_batches "$MAX_BATCHES")
  fi

  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_rank_transition.py \
    --dataset "$DATASET" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --base_model "$base_model" \
    --cands_model "$cands_model" \
    --base_checkpoint "$base_ckpt" \
    --cands_checkpoint "$cands_ckpt" \
    --hidden_size "$hidden_size" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --wearec_num_heads "$WEAREC_NUM_HEADS" \
    --wearec_alpha "$WEAREC_ALPHA" \
    --inner_size "$inner_size" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --out_prefix "$out_prefix" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE ${backbone} h${hidden_size}" | tee -a "$LOG_DIR/master.log"
}

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] backbones=$BACKBONES_STR hidden_sizes=$HIDDEN_SIZES_STR gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"

for backbone in $BACKBONES_STR; do
  for hidden_size in $HIDDEN_SIZES_STR; do
    run_one "$backbone" "$hidden_size"
  done
done

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
