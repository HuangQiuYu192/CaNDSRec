#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-only validation of the AngularSmooth mechanism.  It asks whether
# tail gains occur specifically when angular neighbors carry transition evidence
# from the current sequence; no model is trained in this script.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
DATASETS_STR="${DATASETS_STR:-Beauty Sports Toys LastFM-S3Rec Yelp-S3Rec}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/angular_smooth_evidence}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/angular_smooth_evidence}"
TASK_FILE="$OUT_DIR/tasks.tsv"

SEED="${SEED:-2025}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
NEIGHBOR_K="${NEIGHBOR_K:-10}"
RECENT_WINDOW="${RECENT_WINDOW:-5}"
MAX_BATCHES="${MAX_BATCHES:-}"
FORCE="${FORCE:-False}"
RUN_COMPETITION="${RUN_COMPETITION:-False}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi

conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/collect_best_tuned_table.py \
  --datasets $DATASETS_STR \
  --out_dir "$ROOT/analysis_results/best_tuned_table" \
  > "$LOG_DIR/collect_best_tuned.log" 2>&1

conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/prepare_angular_smooth_evidence_tasks.py \
  --candidates_csv "$ROOT/analysis_results/best_tuned_table/all_candidates.csv" \
  --datasets $DATASETS_STR \
  --ckpt_root "$ROOT/ckpt" \
  --out_tsv "$TASK_FILE" \
  | tee "$LOG_DIR/prepare_tasks.log"

tail -n +2 "$TASK_FILE" | while IFS=$'\t' read -r dataset hidden max_len temp cands_inner_size smooth_inner_size weight smooth_k smooth_temp quantile threshold base_run smooth_run base_checkpoint smooth_checkpoint; do
  tag="${dataset}_h${hidden}_len${max_len}_temp${temp}"
  out_prefix="$OUT_DIR/$tag"
  if [ -z "$base_checkpoint" ] || [ -z "$smooth_checkpoint" ]; then
    echo "MISSING $tag base=$base_checkpoint smooth=$smooth_checkpoint" | tee -a "$LOG_DIR/master.log"
    continue
  fi
  extra_args=()
  if [ -n "$MAX_BATCHES" ]; then
    extra_args=(--max_batches "$MAX_BATCHES")
  fi
  if [ -s "${out_prefix}.csv" ] && [ "$FORCE" != "True" ]; then
    echo "SKIP evidence $tag (already exists)" | tee -a "$LOG_DIR/master.log"
  else
    echo "START evidence $tag" | tee -a "$LOG_DIR/master.log"
    conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_angular_smooth_evidence.py \
    --dataset "$dataset" \
    --cands_checkpoint "$base_checkpoint" \
    --smooth_checkpoint "$smooth_checkpoint" \
    --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size "$hidden" --max_item_list_length "$max_len" \
    --cands_inner_size "$cands_inner_size" --smooth_inner_size "$smooth_inner_size" --temperature "$temp" \
    --n_layers "$N_LAYERS" --n_heads "$N_HEADS" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" --train_batch_size "$TRAIN_BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
    --angular_smooth_weight "$weight" --angular_smooth_k "$smooth_k" \
    --angular_smooth_temperature "$smooth_temp" --angular_smooth_pop_quantile "$quantile" \
    --angular_smooth_sim_threshold "$threshold" --angular_smooth_pop_weight False \
    --neighbor_k "$NEIGHBOR_K" --recent_window "$RECENT_WINDOW" \
    --out_prefix "$out_prefix" "${extra_args[@]}" \
      > "$LOG_DIR/${tag}.log" 2>&1
    echo "DONE evidence $tag" | tee -a "$LOG_DIR/master.log"
  fi
  competition_prefix="${out_prefix}_neighbor_competition"
  if [ "$RUN_COMPETITION" = "True" ]; then
    if [ -s "${competition_prefix}_summary.csv" ] && [ "$FORCE" != "True" ]; then
      echo "SKIP competition $tag (already exists)" | tee -a "$LOG_DIR/master.log"
    else
      echo "START competition $tag" | tee -a "$LOG_DIR/master.log"
      conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_angular_smooth_neighbor_competition.py \
        --dataset "$dataset" --cands_checkpoint "$base_checkpoint" --smooth_checkpoint "$smooth_checkpoint" \
        --gpu_id "$GPU_ID" --seed "$SEED" --hidden_size "$hidden" --max_item_list_length "$max_len" \
        --cands_inner_size "$cands_inner_size" --smooth_inner_size "$smooth_inner_size" --temperature "$temp" \
        --n_layers "$N_LAYERS" --n_heads "$N_HEADS" --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
        --attn_dropout_prob "$ATTN_DROPOUT_PROB" --learning_rate "$LEARNING_RATE" \
        --train_batch_size "$TRAIN_BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
        --angular_smooth_weight "$weight" --angular_smooth_k "$smooth_k" \
        --angular_smooth_temperature "$smooth_temp" --angular_smooth_pop_quantile "$quantile" \
        --angular_smooth_sim_threshold "$threshold" --angular_smooth_pop_weight False \
        --neighbor_k "$NEIGHBOR_K" --recent_window "$RECENT_WINDOW" --out_prefix "$competition_prefix" "${extra_args[@]}" \
        > "$LOG_DIR/${tag}_neighbor_competition.log" 2>&1
      echo "DONE competition $tag" | tee -a "$LOG_DIR/master.log"
    fi
  fi
done

echo "ALL_DONE" | tee -a "$LOG_DIR/master.log"
