#!/usr/bin/env bash
set -euo pipefail

# P0b: all pairs below have identical forward scores at a fixed parameter
# state.  They differ only in whether one normalized side propagates the
# denominator derivative. GPU 1 only; completed logs are never rerun.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports}"
SEEDS_STR="${SEEDS_STR:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/www27_selective_stopgrad_gpu1}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_selective_stopgrad_gpu1}"

source "$CONDA_SH"; cd "$ROOT"
run_one() {
  local dataset="$1" seed="$2" tag="$3" model="$4" geometry="$5"
  local log="$OUT_ROOT/$dataset/seed$seed/$tag.log" ckpt="$CKPT_ROOT/$dataset/seed$seed/$tag"
  mkdir -p "$(dirname "$log")" "$ckpt"
  grep -q 'MODEL TEST' "$log" 2>/dev/null && { echo "SKIP $dataset $seed $tag"; return; }
  echo "START $dataset seed=$seed $tag gpu=$GPU_ID" | tee -a "$OUT_ROOT/launcher.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --model "$model" --dataset "$dataset" --gpu_id "$GPU_ID" --seed "$seed" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
    --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
    --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
    --max_item_list_length 50 --temperature 10 --score_geometry "$geometry" \
    --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
}
for dataset in $DATASETS_STR; do for seed in $SEEDS_STR; do
  run_one "$dataset" "$seed" sequence GeometrySASRec sequence
  run_one "$dataset" "$seed" stopgrad_sequence StopGradSequenceGeometrySASRec sequence
  run_one "$dataset" "$seed" item GeometrySASRec item
  run_one "$dataset" "$seed" stopgrad_item StopGradItemGeometrySASRec item
done; done
python experiments/cross_dataset/collect_www27_gradient_causality.py --log_root "$OUT_ROOT" --output_prefix "$ROOT/analysis_results/www27_selective_stopgrad/summary"
echo ALL_DONE
