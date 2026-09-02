#!/usr/bin/env bash
set -euo pipefail

cd /home/ssh_user/code/25-HuangQiuyu/LongTailRec
source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID=0
RUN_ROOT="log_runs/tailcl_h256_2080ti"
CKPT_ROOT="ckpt/tailcl_h256_2080ti"
mkdir -p "$RUN_ROOT" "$CKPT_ROOT"

COMMON_ARGS=(
  --model TailCLCalibratedCANDSSASRec
  --dataset Beauty
  --gpu_id "$GPU_ID"
  --seed 2025
  --hidden_size 256
  --inner_size 1024
  --train_batch_size 1024
  --eval_batch_size 1024
  --epochs 300
  --eval_step 1
  --stopping_step 10
  --valid_metric NDCG@10
  --temperature 10.0
  --bias_init_scale 0.05
  --bias_reg_weight 0.0001
  --tail_cl_temperature 0.2
  --tail_cl_quantile 0.33
  --tail_cl_min_teacher_pop 20
  --tail_cl_max_items 256
  --show_progress True
  --verbose True
)

run_one() {
  local name="$1"
  shift
  local log_file="$RUN_ROOT/${name}.log"
  local ckpt_dir="$CKPT_ROOT/${name}"
  mkdir -p "$ckpt_dir"
  echo "===== START ${name} $(date '+%F %T') =====" | tee -a "$RUN_ROOT/master.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n recbole \
    python main.py "${COMMON_ARGS[@]}" --checkpoint_dir "$ckpt_dir" "$@" \
    > "$log_file" 2>&1
  echo "===== DONE ${name} $(date '+%F %T') =====" | tee -a "$RUN_ROOT/master.log"
  grep -E "best valid result|test result" "$log_file" | tail -2 | tee -a "$RUN_ROOT/master.log" || true
}

for weight in 0.001 0.005 0.010 0.020; do
  run_one "tailcl_w${weight}_t02_leaf" --tail_cl_weight "$weight"
done

run_one "tailcl_w005_t01_leaf" --tail_cl_weight 0.005 --tail_cl_temperature 0.1
run_one "tailcl_w005_t05_leaf" --tail_cl_weight 0.005 --tail_cl_temperature 0.5
run_one "tailcl_w005_teacher50_leaf" --tail_cl_weight 0.005 --tail_cl_min_teacher_pop 50

conda run --no-capture-output -n recbole python - <<'PY'
import ast
import re
from pathlib import Path

root = Path("log_runs/tailcl_h256_2080ti")
rows = []
for path in sorted(root.glob("*.log")):
    if path.name == "master.log":
        continue
    text = path.read_text(errors="ignore")
    matches = re.findall(r"test result: OrderedDict\((\[.*?\])\)", text)
    if not matches:
        rows.append([path.stem, "MISSING", "", "", "", "", "", ""])
        continue
    metrics = dict(ast.literal_eval(matches[-1]))
    rows.append([
        path.stem, "OK",
        metrics.get("recall@5", ""), metrics.get("recall@10", ""), metrics.get("recall@20", ""),
        metrics.get("ndcg@5", ""), metrics.get("ndcg@10", ""), metrics.get("ndcg@20", ""),
    ])

header = ["run", "status", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
out = root / "summary.tsv"
out.write_text("\t".join(header) + "\n" + "\n".join("\t".join(map(str, row)) for row in rows) + "\n")
print(out)
PY
