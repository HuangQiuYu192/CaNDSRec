#!/usr/bin/env bash
set -euo pipefail

# P4 cross-domain mechanism screen.  SASRec and joint CaNDS endpoints for the
# same seed already exist; this adds only the missing sequence-only geometry.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
DATASETS_STR="${DATASETS_STR:-Sports Toys}"
SEED="${SEED:-2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_sequence_geometry_probe_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_sequence_geometry_probe_gpu1}"
RERUN="${RERUN:-false}"

source "$CONDA_SH"
cd "$ROOT"
for dataset in $DATASETS_STR; do
  log="$OUT_ROOT/$dataset/seed$SEED/sequence_tau10.log"
  ckpt="$CKPT_ROOT/$dataset/seed$SEED"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
    echo "SKIP completed dataset=$dataset seed=$SEED" | tee -a "$OUT_ROOT/driver.log"
    continue
  fi
  echo "START P4 dataset=$dataset seed=$SEED sequence tau=10 gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model GeometrySASRec --dataset "$dataset" --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature 10 --score_geometry sequence \
    --sequence_norm_power 1 --item_norm_power 0 --checkpoint_dir "$ckpt" \
    --verbose True --show_progress True > "$log" 2>&1
  grep -A2 'test result' "$log" | tee -a "$OUT_ROOT/driver.log" || true
done
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
