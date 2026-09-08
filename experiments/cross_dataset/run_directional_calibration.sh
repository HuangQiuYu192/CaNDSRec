#!/usr/bin/env bash
set -euo pipefail

# Run after matched SASRec/CaNDS checkpoints exist.  This script runs one
# analysis at a time and honors GPU_ID, so it never claims an otherwise busy GPU.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_ENV="${CONDA_ENV:-recbole}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-Beauty}"
SEED="${SEED:-2025}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
INNER_SIZE="${INNER_SIZE:-1024}"
TEMPERATURE="${TEMPERATURE:-10}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/www27_${DATASET,,}_seed${SEED}}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/www27_directional_calibration}"

latest_ckpt() {
  local run_name="$1"
  local nested
  nested="$(find "$CKPT_DIR/$run_name" -type f -name '*.pth' 2>/dev/null | sort | tail -n 1)"
  if [ -n "$nested" ]; then
    echo "$nested"
    return 0
  fi
  # RecBole also permits a flat checkpoint directory.  Infer only the model
  # prefix here; the caller has already supplied an isolated experiment dir.
  local pattern='SASRec-*.pth'
  if [[ "$run_name" == *'_CANDSSASRec_'* ]]; then
    pattern='CANDSSASRec-*.pth'
  fi
  find "$CKPT_DIR" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | sort | tail -n 1
}

sasrec_name="${DATASET}_SASRec_h${HIDDEN_SIZE}_len${MAX_LEN}"
cands_name="${DATASET}_CANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}"
sasrec_ckpt="$(latest_ckpt "$sasrec_name")"
cands_ckpt="$(latest_ckpt "$cands_name")"
if [ -z "$sasrec_ckpt" ] || [ -z "$cands_ckpt" ]; then
  echo "Missing matched checkpoints: SASRec=$sasrec_ckpt CaNDS=$cands_ckpt" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
elif ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found; set CONDA_SH to the environment initialization script" >&2
  exit 127
fi
conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_directional_calibration.py \
  --dataset "$DATASET" --gpu_id "$GPU_ID" --seed "$SEED" \
  --hidden_size "$HIDDEN_SIZE" --n_layers 2 --n_heads 2 --inner_size "$INNER_SIZE" \
  --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
  --max_item_list_length "$MAX_LEN" --train_batch_size 1024 --eval_batch_size 512 --temperature "$TEMPERATURE" \
  --sasrec_checkpoint "$sasrec_ckpt" --cands_checkpoint "$cands_ckpt" \
  --output "$OUT_DIR/${DATASET}_seed${SEED}_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}.csv"
