#!/usr/bin/env bash
set -euo pipefail

cd /home/ssh_user/code/25-HuangQiuyu/LongTailRec
source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID=0
RUN_ROOT="log_runs/category_intent_h256_2080ti"
CKPT_ROOT="ckpt/category_intent_h256_2080ti"
mkdir -p "$RUN_ROOT" "$CKPT_ROOT"

COMMON_ARGS=(
  --model CategoryIntentCANDSSASRec
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
  --use_category_bias True
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

for lambda in 0.02 0.05 0.10 0.20; do
  run_one "catintent_l${lambda}_t10" --cat_lambda "$lambda" --cat_temperature 10.0
done

for temp in 5.0 15.0; do
  run_one "catintent_l0.05_t${temp}" --cat_lambda 0.05 --cat_temperature "$temp"
done

conda run --no-capture-output -n recbole python - <<'PY'
import ast
import re
from pathlib import Path

root = Path("log_runs/category_intent_h256_2080ti")
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
