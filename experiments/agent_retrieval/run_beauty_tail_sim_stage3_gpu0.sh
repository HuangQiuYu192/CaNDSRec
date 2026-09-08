#!/usr/bin/env bash
set -euo pipefail

# Stage 3: a same-seed, low-weight replay pilot. The strict rule was frozen by
# Stage 2.5. It replays only observed targets recovered by leave-one-out graph
# evidence; no unverified synthetic label is used during training.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
SIM_WEIGHT="${SIM_WEIGHT:-0.03}"
MODES_STR="${MODES_STR:-baseline random raw strict}"
TAG="${TAG:-beauty_tail_sim_stage3_gpu0}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/agent_simulator/beauty_tail_sim_stage3}"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$OUT_DIR/group_metrics"
cd "$ROOT"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi

run_one() {
  local mode="$1"
  local name="Beauty_${mode}_h256_len50_temp10"
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/$name"
  local model="TailSimCANDSSASRec"
  local extra=(--tail_sim_weight "$SIM_WEIGHT" --tail_sim_mode "$mode")
  if [ "$mode" = "baseline" ]; then
    model="CANDSSASRec"
    extra=()
  fi
  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "SKIP $name (completed)"
  else
    mkdir -p "$ckpt_path"
    echo "START $name model=$model"
    conda run --no-capture-output -n "$CONDA_ENV" python main.py \
      --dataset Beauty --model "$model" --gpu_id "$GPU_ID" --seed "$SEED" \
      --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 \
      --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 --learning_rate 0.001 \
      --epochs "$EPOCHS" --stopping_step "$STOPPING_STEP" --train_batch_size 1024 --eval_batch_size 512 \
      --max_item_list_length 50 --temperature 10 --checkpoint_dir "$ckpt_path" \
      --verbose True --show_progress True "${extra[@]}" > "$log_file" 2>&1
  fi
  if ! grep -q "test result" "$log_file"; then echo "ERROR: failed $name; inspect $log_file" >&2; return 1; fi
  local checkpoint
  checkpoint="$(find "$ckpt_path" -type f -name '*.pth' | sort | tail -n 1)"
  if [ -z "$checkpoint" ]; then echo "ERROR: missing checkpoint for $name" >&2; return 1; fi
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --model "$model" --checkpoint "$checkpoint" --tag "$mode" --dataset Beauty --gpu_id "$GPU_ID" --seed "$SEED" \
    --hidden_size 256 --n_layers 2 --n_heads 2 --inner_size 1024 --hidden_dropout_prob 0.5 --attn_dropout_prob 0.5 \
    --learning_rate 0.001 --train_batch_size 1024 --eval_batch_size 512 --max_item_list_length 50 --temperature 10 \
    --cutoffs 5,10,20,50,100 --out_prefix "$OUT_DIR/group_metrics/group_${mode}"
  grep -E "TailSim real-tail replay|best valid result|test result" "$log_file" | tail -n 4 | tee -a "$OUT_DIR/training_summary.raw"
}

for mode in $MODES_STR; do
  case "$mode" in baseline|random|raw|strict) run_one "$mode" ;; *) echo "ERROR: unsupported mode '$mode'" >&2; exit 1 ;; esac
done

conda run --no-capture-output -n "$CONDA_ENV" python experiments/agent_retrieval/summarize_tail_sim_stage3.py \
  --input_dir "$OUT_DIR/group_metrics" --out_prefix "$OUT_DIR/Beauty_h256_len50_temp10"
echo "ALL_DONE"
