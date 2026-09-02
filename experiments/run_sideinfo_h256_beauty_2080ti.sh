#!/usr/bin/env bash
set -euo pipefail

cd /home/ssh_user/code/25-HuangQiuyu/LongTailRec
source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID=0
RUN_ROOT="log_runs/sideinfo_h256_2080ti"
CKPT_ROOT="ckpt/sideinfo_h256_2080ti"
mkdir -p "$RUN_ROOT" "$CKPT_ROOT"

COMMON_ARGS=(
  --model SideInfoCANDSSASRec
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

# A: metadata-aware item direction.
for beta in 0.05 0.10 0.20 0.50; do
  run_one "cat_b${beta}" \
    --use_category_side True --category_beta "$beta" --side_beta 1.0
done

# B: transition-graph enhanced item direction.
for beta in 0.05 0.10 0.20 0.50; do
  run_one "graph_b${beta}_k16" \
    --use_graph_side True --graph_beta "$beta" --graph_topk 16 --graph_self_loop True --side_beta 1.0
done

# Sensitivity to graph neighborhood size.
for k in 8 32; do
  run_one "graph_b0.20_k${k}" \
    --use_graph_side True --graph_beta 0.20 --graph_topk "$k" --graph_self_loop True --side_beta 1.0
done

# A+B: metadata and graph jointly refine item direction.
run_one "cat_graph_b0.10" \
  --use_category_side True --category_beta 0.10 \
  --use_graph_side True --graph_beta 0.10 --graph_topk 16 --graph_self_loop True --side_beta 1.0
run_one "cat_graph_b0.20" \
  --use_category_side True --category_beta 0.20 \
  --use_graph_side True --graph_beta 0.20 --graph_topk 16 --graph_self_loop True --side_beta 1.0

conda run --no-capture-output -n recbole python - <<'PY'
import ast
import re
from pathlib import Path

root = Path("log_runs/sideinfo_h256_2080ti")
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
