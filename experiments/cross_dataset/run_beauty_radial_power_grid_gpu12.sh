#!/usr/bin/env bash
set -euo pipefail

# Controlled SASRec score-geometry experiment on Beauty.  The only model
# difference is how much sequence/item radial scale is retained in the final
# score.  No GPU other than 1 and 2 is used.
#
# Modes:
#   screen    one seed, tau=10: dot, three discrete geometries, three partials
#   tune      one seed: tau in {5,20} for the promising normalized geometries
#   replicate seeds {2023,2024,2025} for SELECTED_TAGS (default: sequence,both)
#   all       screen, tune, then replicate (run phases sequentially)

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
MODE="${MODE:-screen}"
OUT_ROOT="${OUT_ROOT:-$ROOT/log_runs/beauty_radial_power_grid_gpu12}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
RERUN="${RERUN:-false}"
SELECTED_TAGS="${SELECTED_TAGS:-sequence both}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local gpu="$1" seed="$2" tag="$3" geometry="$4" tau="$5" seq_power="$6" item_power="$7"
  local log="$OUT_ROOT/seed${seed}/${tag}_tau${tau}.log"
  local ckpt="$CKPT_ROOT/seed${seed}/${tag}_tau${tau}"
  mkdir -p "$(dirname "$log")" "$ckpt"
  if [ "$RERUN" != true ] && grep -q 'MODEL TEST' "$log" 2>/dev/null; then
    echo "SKIP seed=$seed tag=$tag tau=$tau"; return 0
  fi
  echo "START seed=$seed tag=$tag geometry=$geometry tau=$tau powers=($seq_power,$item_power) gpu=$gpu" | tee -a "$OUT_ROOT/launcher.log"
  if [ "$geometry" = dot ]; then
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --model SASRec --dataset Beauty --gpu_id "$gpu" --seed "$seed" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  else
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --model GeometrySASRec --dataset Beauty --gpu_id "$gpu" --seed "$seed" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs 300 --stopping_step 10 --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --temperature "$tau" --score_geometry "$geometry" \
      --sequence_norm_power "$seq_power" --item_norm_power "$item_power" \
      --checkpoint_dir "$ckpt" --verbose True --show_progress True > "$log" 2>&1
  fi
  grep -A2 'test result' "$log" | tee -a "$OUT_ROOT/summary.raw" || true
}

jobs=()
start_one() { run_one "$@" & jobs+=("$!"); }
wait_all() {
  local pid
  for pid in "${jobs[@]}"; do wait "$pid"; done
  jobs=()
}

screen() {
  # Fixed tau first: isolates geometry before temperature becomes a confound.
  start_one 1 2025 dot dot 1 0 0
  start_one 2 2025 sequence sequence 10 1 0
  wait_all
  start_one 1 2025 item item 10 0 1
  start_one 2 2025 both both 10 1 1
  wait_all
  start_one 1 2025 partial_05_05 partial 10 0.5 0.5
  start_one 2 2025 partial_10_05 partial 10 1.0 0.5
  wait_all
  run_one 1 2025 partial_05_10 partial 10 0.5 1.0
}

tune() {
  # Tau=10 is already present from screen. Tune only the normalized variants.
  local tau tag geometry seq_power item_power
  for tau in 5 20; do
    start_one 1 2025 sequence sequence "$tau" 1 0
    start_one 2 2025 both both "$tau" 1 1
    wait_all
    start_one 1 2025 partial_05_05 partial "$tau" 0.5 0.5
    start_one 2 2025 partial_10_05 partial "$tau" 1.0 0.5
    wait_all
    run_one 1 2025 partial_05_10 partial "$tau" 0.5 1.0
  done
}

replicate() {
  # Set SELECTED_TAGS after reading the screen/tune table.  Default compares
  # the strongest observed one-sided geometry with joint normalization.
  local seed tag geometry seq_power item_power tau
  for seed in 2023 2024 2025; do
    start_one 1 "$seed" dot dot 1 0 0
    for tag in $SELECTED_TAGS; do
      case "$tag" in
        sequence) geometry=sequence; seq_power=1; item_power=0; tau=10 ;;
        item) geometry=item; seq_power=0; item_power=1; tau=10 ;;
        both) geometry=both; seq_power=1; item_power=1; tau=10 ;;
        partial_05_05) geometry=partial; seq_power=0.5; item_power=0.5; tau=10 ;;
        partial_10_05) geometry=partial; seq_power=1; item_power=0.5; tau=10 ;;
        partial_05_10) geometry=partial; seq_power=0.5; item_power=1; tau=10 ;;
        *) echo "Unknown SELECTED_TAGS entry: $tag" >&2; exit 2 ;;
      esac
      start_one 2 "$seed" "$tag" "$geometry" "$tau" "$seq_power" "$item_power"
      wait_all
    done
  done
}

case "$MODE" in
  screen) screen ;;
  tune) tune ;;
  replicate) replicate ;;
  all) screen; tune; replicate ;;
  *) echo "MODE must be screen, tune, replicate, or all" >&2; exit 2 ;;
esac
echo "ALL_DONE mode=$MODE"
