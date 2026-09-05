#!/usr/bin/env bash
set -euo pipefail

# Grouped all/head/mid/tail evaluation for CaNDS and selected AngularSmooth runs.
# Default: Beauty, hidden size 256, GPU 0.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-0}"

DATASET="${DATASET:-Beauty}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
TEMPERATURE="${TEMPERATURE:-10}"
SEED="${SEED:-2025}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
INNER_SIZE="${INNER_SIZE:-$(( HIDDEN_SIZE * 4 ))}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.5}"
ATTN_DROPOUT_PROB="${ATTN_DROPOUT_PROB:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
CUTOFFS="${CUTOFFS:-5,10,20,50,100}"

BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT/ckpt/main_benchmark_grid}"
ANGULAR_CKPT_DIR="${ANGULAR_CKPT_DIR:-$ROOT/ckpt/beauty_angular_smooth_cands_gpu0}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/beauty_angular_smooth_group_eval}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/beauty_angular_smooth_group_eval_gpu0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
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

run_eval() {
  local tag="$1"
  local model="$2"
  local ckpt="$3"
  local weight="$4"
  local k="$5"
  local smooth_temp="$6"
  local quantile="$7"
  local threshold="$8"
  local out_prefix="$OUT_DIR/$tag"
  local log_file="$LOG_DIR/$tag.log"

  if [ -z "$ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START $tag gpu=$GPU_ID" | tee -a "$LOG_DIR/master.log"
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
    --angular_smooth_weight "$weight" \
    --angular_smooth_k "$k" \
    --angular_smooth_temperature "$smooth_temp" \
    --angular_smooth_pop_quantile "$quantile" \
    --angular_smooth_sim_threshold "$threshold" \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOG_DIR/master.log"
}

BASE_RUN="${DATASET}_CANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}"
BASE_CKPT="$(latest_checkpoint_in_dir "$BASE_CKPT_DIR" "$BASE_RUN")"
run_eval "CaNDS_h${HIDDEN_SIZE}_temp${TEMPERATURE}" CANDSSASRec "$BASE_CKPT" 0.0 10 0.2 0.67 0.0

# Representative AngularSmooth settings from the first Beauty h256 search:
# best NDCG@10, strong Recall@10, and a stable middle setting.
run_specs=(
  "AS_best_ndcg 0.03 5 0.1 0.67 0.2"
  "AS_best_recall 0.1 10 0.1 0.67 0.2"
  "AS_stable 0.05 10 0.1 0.67 0.0"
)

for spec in "${run_specs[@]}"; do
  read -r tag weight k smooth_temp quantile threshold <<< "$spec"
  run_name="${DATASET}_AngularSmoothCANDSSASRec_h${HIDDEN_SIZE}_len${MAX_LEN}_temp${TEMPERATURE}_w${weight}_k${k}_st${smooth_temp}_q${quantile}_thr${threshold}"
  ckpt="$(latest_checkpoint_in_dir "$ANGULAR_CKPT_DIR" "$run_name")"
  run_eval "$tag" AngularSmoothCANDSSASRec "$ckpt" "$weight" "$k" "$smooth_temp" "$quantile" "$threshold"
done

python - "$OUT_DIR" <<'PY'
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

order = {"CaNDS_h256_temp10": 0, "AS_best_ndcg": 1, "AS_best_recall": 2, "AS_stable": 3}
group_order = {"all": 0, "head": 1, "mid": 2, "tail": 3}
rows.sort(key=lambda r: (order.get(r["tag"], 99), group_order.get(r["group"], 99)))

preferred = [
    "tag", "model", "hidden", "temperature", "angular_smooth_weight", "angular_smooth_k",
    "angular_smooth_temperature", "angular_smooth_sim_threshold", "group", "n", "median_rank",
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
cat "$OUT_DIR/summary.md"
