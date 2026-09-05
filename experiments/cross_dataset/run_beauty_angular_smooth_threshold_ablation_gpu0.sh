#!/usr/bin/env bash
set -euo pipefail

# Ablation for angular-neighbor confidence threshold.
# Lower threshold uses broader neighbors; higher threshold keeps only confident angular neighbors.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
INNER_SIZE="${INNER_SIZE:-$(( HIDDEN_SIZE * 4 ))}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
CUTOFFS="${CUTOFFS:-5,10,20,50,100}"

SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.1}"
SMOOTH_K="${SMOOTH_K:-10}"
SMOOTH_TEMP="${SMOOTH_TEMP:-0.1}"
SMOOTH_QUANTILE="${SMOOTH_QUANTILE:-0.67}"
POP_WEIGHT="${POP_WEIGHT:-True}"
SIM_THRESHOLDS_STR="${SIM_THRESHOLDS_STR:-0.0 0.1 0.2 0.3 0.4}"

BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT/ckpt/main_benchmark_grid}"
TAG="${TAG:-beauty_angular_smooth_threshold_ablation_gpu0}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
GROUP_OUT_DIR="${GROUP_OUT_DIR:-$ROOT/analysis_results/beauty_angular_smooth_threshold_ablation}"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$GROUP_OUT_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

latest_checkpoint_in_dir() {
  local ckpt_root="$1"
  local run_name="$2"
  find "$ckpt_root/$run_name" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | sort | tail -n 1
}

train_one() {
  local threshold="$1"
  local name="${DATASET}_AngularSmoothCANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}_w${SMOOTH_WEIGHT}_k${SMOOTH_K}_st${SMOOTH_TEMP}_q${SMOOTH_QUANTILE}_thr${threshold}"
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/${name}"

  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP train threshold=${threshold}" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  mkdir -p "$ckpt_path"
  echo "[$(date '+%F %T')] START train threshold=${threshold} gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --dataset "$DATASET" \
    --model AngularSmoothCANDSSASRec \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$INNER_SIZE" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --stopping_step "$STOPPING_STEP" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_item_list_length "$MAX_LEN" \
    --temperature "$TEMPERATURE" \
    --angular_smooth_weight "$SMOOTH_WEIGHT" \
    --angular_smooth_k "$SMOOTH_K" \
    --angular_smooth_temperature "$SMOOTH_TEMP" \
    --angular_smooth_pop_quantile "$SMOOTH_QUANTILE" \
    --angular_smooth_sim_threshold "$threshold" \
    --angular_smooth_pop_weight "$POP_WEIGHT" \
    --checkpoint_dir "$ckpt_path" \
    --verbose True \
    --show_progress True \
    > "$log_file" 2>&1
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  echo "[$(date '+%F %T')] DONE train threshold=${threshold}" | tee -a "$LOG_DIR/master.log"
}

group_eval_one() {
  local tag="$1"
  local model="$2"
  local ckpt="$3"
  local threshold="$4"
  local out_prefix="$GROUP_OUT_DIR/$tag"
  local log_file="$LOG_DIR/group_${tag}.log"

  if [ -z "$ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING group eval $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP group eval $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START group eval $tag gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --dataset "$DATASET" \
    --model "$model" \
    --checkpoint "$ckpt" \
    --tag "$tag" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$INNER_SIZE" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --angular_smooth_weight "$SMOOTH_WEIGHT" \
    --angular_smooth_k "$SMOOTH_K" \
    --angular_smooth_temperature "$SMOOTH_TEMP" \
    --angular_smooth_pop_quantile "$SMOOTH_QUANTILE" \
    --angular_smooth_sim_threshold "$threshold" \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE group eval $tag" | tee -a "$LOG_DIR/master.log"
}

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] thresholds=$SIM_THRESHOLDS_STR gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"

for threshold in $SIM_THRESHOLDS_STR; do
  train_one "$threshold"
done

python experiments/cross_dataset/collect_angular_smooth_results.py --log_dir "$LOG_DIR" || true

base_run="${DATASET}_CANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}"
base_ckpt="$(latest_checkpoint_in_dir "$BASE_CKPT_DIR" "$base_run")"
group_eval_one "CaNDS_h${HIDDEN_SIZE}_temp${TEMPERATURE}" CANDSSASRec "$base_ckpt" 0.0

for threshold in $SIM_THRESHOLDS_STR; do
  run_name="${DATASET}_AngularSmoothCANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}_w${SMOOTH_WEIGHT}_k${SMOOTH_K}_st${SMOOTH_TEMP}_q${SMOOTH_QUANTILE}_thr${threshold}"
  ckpt="$(latest_checkpoint_in_dir "$CKPT_DIR" "$run_name")"
  tag="AS_thr${threshold}"
  group_eval_one "$tag" AngularSmoothCANDSSASRec "$ckpt" "$threshold"
done

python - "$GROUP_OUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.csv")):
    if path.name == "summary.csv":
        continue
    with path.open(encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

order = {
    "CaNDS_h256_temp10": 0,
    "AS_thr0.0": 1,
    "AS_thr0.1": 2,
    "AS_thr0.2": 3,
    "AS_thr0.3": 4,
    "AS_thr0.4": 5,
}
group_order = {"all": 0, "head": 1, "mid": 2, "tail": 3}
rows.sort(key=lambda r: (order.get(r["tag"], 99), group_order.get(r["group"], 99)))

preferred = [
    "tag", "model", "hidden", "temperature", "angular_smooth_weight", "angular_smooth_k",
    "angular_smooth_temperature", "angular_smooth_pop_quantile", "angular_smooth_sim_threshold",
    "group", "n", "median_rank",
    "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
    "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
]
extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
headers = preferred + extra

with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
(out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} rows to {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}")
PY

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/master.log"
echo
cat "$LOG_DIR/angular_smooth_summary.md" 2>/dev/null || true
echo
cat "$GROUP_OUT_DIR/summary.md"
