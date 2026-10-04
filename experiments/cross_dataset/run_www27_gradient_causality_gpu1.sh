#!/usr/bin/env bash
set -euo pipefail

# P0 causal mechanism check: true normalization and stop-gradient
# normalization have identical forward scores at fixed parameters, but only
# the former removes direct radial gradients.  GPU 1 only by design.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_gradient_causality_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_gradient_causality_gpu1}"
RERUN="${RERUN:-false}"

[ -f "$CONDA_SH" ] || { echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; }
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local dataset="$1" seed="$2" variant="$3" model="$4" geometry="$5"
  local log="$OUT_ROOT/$dataset/seed$seed/$variant.log"
  local ckpt="$CKPT_ROOT/$dataset/seed$seed/$variant"
  local geometry_args=()
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
    echo "SKIP $dataset seed=$seed $variant"; return
  fi
  if [ "$model" != "SASRec" ]; then geometry_args=(--score_geometry "$geometry" --temperature 10); fi
  echo "START $dataset seed=$seed $variant gpu=$GPU_ID" | tee -a "$OUT_ROOT/launcher.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model "$model" --dataset "$dataset" --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --weight_decay 0.0 --epochs 300 --stopping_step 10 --train_batch_size 1024 \
    --eval_batch_size 512 --max_item_list_length 50 --checkpoint_dir "$ckpt" \
    --verbose True --show_progress True "${geometry_args[@]}" \
    > "$log" 2>&1
}

for dataset in $DATASETS_STR; do
  for seed in $SEEDS_STR; do
    run_one "$dataset" "$seed" dot SASRec both
    run_one "$dataset" "$seed" joint GeometrySASRec both
    run_one "$dataset" "$seed" stopgrad_joint StopGradGeometrySASRec both
  done
done

python experiments/cross_dataset/collect_www27_gradient_causality.py \
  --log_root "$OUT_ROOT" --output_prefix "$ROOT/analysis_results/www27_gradient_causality/summary"
echo ALL_DONE
