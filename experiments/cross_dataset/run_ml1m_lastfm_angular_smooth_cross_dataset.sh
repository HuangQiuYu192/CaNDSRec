#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset validation for AngularSmoothCANDSSASRec on ML-1M and LastFM.
# It keeps the smoothing module hyperparameters fixed and uses dataset-specific
# CaNDS temperatures observed from the main temperature grid.
#
# Default:
#   ML-1M        max_len=50,  temp=20
#   LastFM-S3Rec max_len=200, temp=10
#   hidden=256, GPUs 0 and 1 only

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPUS_STR="${GPUS_STR:-0 1}"

DATASETS_STR="${DATASETS_STR:-ML-1M LastFM-S3Rec}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
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

ML1M_MAX_LEN="${ML1M_MAX_LEN:-50}"
LASTFM_MAX_LEN="${LASTFM_MAX_LEN:-200}"
ML1M_TEMP="${ML1M_TEMP:-20}"
LASTFM_TEMP="${LASTFM_TEMP:-10}"

SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.1}"
SMOOTH_K="${SMOOTH_K:-10}"
SMOOTH_TEMP="${SMOOTH_TEMP:-0.1}"
SMOOTH_QUANTILE="${SMOOTH_QUANTILE:-0.67}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.2}"

BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT/ckpt/main_benchmark_grid}"
TAG="${TAG:-ml1m_lastfm_angular_smooth_cross_dataset}"
LOG_DIR="${LOG_DIR:-$ROOT/log_runs/$TAG}"
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt/$TAG}"
GROUP_OUT_DIR="${GROUP_OUT_DIR:-$ROOT/analysis_results/$TAG}"
TASK_FILE="$LOG_DIR/tasks.tsv"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$GROUP_OUT_DIR"
cd "$ROOT"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
else
  echo "WARN: CONDA_SH not found: $CONDA_SH" | tee -a "$LOG_DIR/master.log"
fi

max_len_for_dataset() {
  case "$1" in
    ML-1M) echo "$ML1M_MAX_LEN" ;;
    LastFM-S3Rec) echo "$LASTFM_MAX_LEN" ;;
    *) echo "50" ;;
  esac
}

temperature_for_dataset() {
  case "$1" in
    ML-1M) echo "$ML1M_TEMP" ;;
    LastFM-S3Rec) echo "$LASTFM_TEMP" ;;
    *) echo "10" ;;
  esac
}

latest_checkpoint_in_dir() {
  local ckpt_root="$1"
  local run_name="$2"
  find "$ckpt_root/$run_name" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | sort | tail -n 1
}

run_name_for_dataset() {
  local dataset="$1"
  local max_len
  local temperature
  max_len="$(max_len_for_dataset "$dataset")"
  temperature="$(temperature_for_dataset "$dataset")"
  echo "${dataset}_AngularSmoothCANDSSASRec_h${HIDDEN_SIZE}_len${max_len}_temp${temperature}_w${SMOOTH_WEIGHT}_k${SMOOTH_K}_st${SMOOTH_TEMP}_q${SMOOTH_QUANTILE}_thr${SIM_THRESHOLD}"
}

build_tasks() {
  : > "$TASK_FILE"
  for dataset in $DATASETS_STR; do
    local name
    name="$(run_name_for_dataset "$dataset")"
    printf "%s\t%s\n" "$name" "$dataset" >> "$TASK_FILE"
  done
}

train_one() {
  local gpu="$1"
  local name="$2"
  local dataset="$3"
  local max_len
  local temperature
  local log_file="$LOG_DIR/${name}.log"
  local ckpt_path="$CKPT_DIR/${name}"
  max_len="$(max_len_for_dataset "$dataset")"
  temperature="$(temperature_for_dataset "$dataset")"

  if grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP train $name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  mkdir -p "$ckpt_path"
  echo "[$(date '+%F %T')] START train $name gpu=$gpu" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python main.py \
    --dataset "$dataset" \
    --model AngularSmoothCANDSSASRec \
    --gpu_id "$gpu" \
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
    --max_item_list_length "$max_len" \
    --temperature "$temperature" \
    --angular_smooth_weight "$SMOOTH_WEIGHT" \
    --angular_smooth_k "$SMOOTH_K" \
    --angular_smooth_temperature "$SMOOTH_TEMP" \
    --angular_smooth_pop_quantile "$SMOOTH_QUANTILE" \
    --angular_smooth_sim_threshold "$SIM_THRESHOLD" \
    --angular_smooth_pop_weight False \
    --checkpoint_dir "$ckpt_path" \
    --verbose True \
    --show_progress True \
    > "$log_file" 2>&1
  grep -E "best valid result|test result" "$log_file" >> "$LOG_DIR/summary.raw" || true
  if ! grep -q "test result" "$log_file" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR train did not finish: $name. Check $log_file" | tee -a "$LOG_DIR/master.log"
    return 1
  fi
  echo "[$(date '+%F %T')] DONE train $name" | tee -a "$LOG_DIR/master.log"
}

worker() {
  local gpu="$1"
  local shard="$2"
  local shards="$3"
  local index=0

  while IFS=$'\t' read -r name dataset; do
    if [ $(( index % shards )) -eq "$shard" ]; then
      train_one "$gpu" "$name" "$dataset"
    fi
    index=$(( index + 1 ))
  done < "$TASK_FILE"
}

group_eval_one() {
  local tag="$1"
  local dataset="$2"
  local model="$3"
  local ckpt="$4"
  local max_len
  local temperature
  local out_prefix="$GROUP_OUT_DIR/$tag"
  local log_file="$LOG_DIR/group_${tag}.log"
  max_len="$(max_len_for_dataset "$dataset")"
  temperature="$(temperature_for_dataset "$dataset")"

  if [ -z "$ckpt" ]; then
    echo "[$(date '+%F %T')] MISSING group eval $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  if [ -s "${out_prefix}.csv" ]; then
    echo "[$(date '+%F %T')] SKIP group eval $tag" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "[$(date '+%F %T')] START group eval $tag gpu=${GROUP_EVAL_GPU:-0}" | tee -a "$LOG_DIR/master.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_group_metrics.py \
    --dataset "$dataset" \
    --model "$model" \
    --checkpoint "$ckpt" \
    --tag "$tag" \
    --gpu_id "${GROUP_EVAL_GPU:-0}" \
    --seed "$SEED" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --inner_size "$INNER_SIZE" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB" \
    --learning_rate "$LEARNING_RATE" \
    --max_item_list_length "$max_len" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --temperature "$temperature" \
    --angular_smooth_weight "$SMOOTH_WEIGHT" \
    --angular_smooth_k "$SMOOTH_K" \
    --angular_smooth_temperature "$SMOOTH_TEMP" \
    --angular_smooth_pop_quantile "$SMOOTH_QUANTILE" \
    --angular_smooth_sim_threshold "$SIM_THRESHOLD" \
    --angular_smooth_pop_weight False \
    --cutoffs "$CUTOFFS" \
    --out_prefix "$out_prefix" \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE group eval $tag" | tee -a "$LOG_DIR/master.log"
}

build_tasks
mapfile -t GPUS < <(printf "%s\n" $GPUS_STR)
if [ "${#GPUS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -gt 2 ]; then
  echo "ERROR: GPUS_STR must contain one or two GPUs, e.g. '0 1'." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [ "$gpu" != "0" ] && [ "$gpu" != "1" ]; then
    echo "ERROR: this script is restricted to GPU 0/1. Got gpu=$gpu from GPUS_STR='$GPUS_STR'." >&2
    exit 1
  fi
done

echo "[$(date '+%F %T')] ROOT=$ROOT" | tee -a "$LOG_DIR/master.log"
echo "[$(date '+%F %T')] datasets=$DATASETS_STR gpus=${GPUS[*]}" | tee -a "$LOG_DIR/master.log"

for shard in "${!GPUS[@]}"; do
  worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  echo $! > "$LOG_DIR/worker_gpu${GPUS[$shard]}.pid"
done
wait

missing=0
while IFS=$'\t' read -r name dataset; do
  if ! grep -q "test result" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR missing test result for $name. Check $LOG_DIR/${name}.log" | tee -a "$LOG_DIR/master.log"
    missing=1
  fi
done < "$TASK_FILE"
if [ "$missing" -ne 0 ]; then
  exit 1
fi

python experiments/cross_dataset/collect_angular_smooth_results.py --log_dir "$LOG_DIR" || true

for dataset in $DATASETS_STR; do
  max_len="$(max_len_for_dataset "$dataset")"
  temperature="$(temperature_for_dataset "$dataset")"

  sasrec_run="${dataset}_SASRec_h${HIDDEN_SIZE}_len${max_len}"
  sasrec_ckpt="$(latest_checkpoint_in_dir "$BASE_CKPT_DIR" "$sasrec_run")"
  group_eval_one "${dataset}_SASRec_h${HIDDEN_SIZE}" "$dataset" SASRec "$sasrec_ckpt"

  cands_run="${dataset}_CANDSSASRec_h${HIDDEN_SIZE}_len${max_len}_temp${temperature}"
  cands_ckpt="$(latest_checkpoint_in_dir "$BASE_CKPT_DIR" "$cands_run")"
  group_eval_one "${dataset}_CaNDS_h${HIDDEN_SIZE}_temp${temperature}" "$dataset" CANDSSASRec "$cands_ckpt"

  as_run="$(run_name_for_dataset "$dataset")"
  as_ckpt="$(latest_checkpoint_in_dir "$CKPT_DIR" "$as_run")"
  group_eval_one "${dataset}_AngularSmooth_h${HIDDEN_SIZE}_temp${temperature}" "$dataset" AngularSmoothCANDSSASRec "$as_ckpt"
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

model_order = {"SASRec": 0, "CANDSSASRec": 1, "AngularSmoothCANDSSASRec": 2}
group_order = {"all": 0, "head": 1, "mid": 2, "tail": 3}
rows.sort(key=lambda r: (r["dataset"], model_order.get(r["model"], 99), group_order.get(r["group"], 99)))

preferred = [
    "dataset", "tag", "model", "hidden", "max_len", "temperature",
    "angular_smooth_weight", "angular_smooth_k", "angular_smooth_temperature",
    "angular_smooth_pop_quantile", "angular_smooth_sim_threshold", "group", "n",
    "median_rank", "recall@5", "recall@10", "recall@20", "recall@50",
    "recall@100", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
]
extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
headers = preferred + extra

with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)

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
