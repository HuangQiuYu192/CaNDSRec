#!/usr/bin/env bash
set -euo pipefail

cd /home/ssh_user/code/25-HuangQiuyu/LongTailRec
source /home/ssh_user/miniconda3/etc/profile.d/conda.sh

GPU_ID=0
RUN_ROOT="log_runs/dynamic_calibration_h256_2080ti"
CKPT_ROOT="ckpt/dynamic_calibration_h256_2080ti"
mkdir -p "$RUN_ROOT" "$CKPT_ROOT"

COMMON_ARGS=(
  --model DynamicCalibratedCANDSSASRec
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

# Module-3 baseline reproduced in the dynamic implementation.
run_one static_bias_b005 \
  --bias_init_scale 0.05

# 1) Context-aware bias: the sequence decides how strongly to use item prior.
run_one context_bias_b005_s1 \
  --bias_init_scale 0.05 --use_context_bias True --context_bias_scale 1.0
run_one context_bias_b005_s2 \
  --bias_init_scale 0.05 --use_context_bias True --context_bias_scale 2.0

# 2) Popularity-conditioned temperature: calibrate angular reliability by item frequency.
run_one poptemp_neg005_b005 \
  --bias_init_scale 0.05 --use_pop_temp True --pop_temp_scale -0.05
run_one poptemp_neg010_b005 \
  --bias_init_scale 0.05 --use_pop_temp True --pop_temp_scale -0.10
run_one poptemp_pos005_b005 \
  --bias_init_scale 0.05 --use_pop_temp True --pop_temp_scale 0.05

# 3) Residual prior: split popularity prior and learned item residual.
run_one residual_b005_reg1e-4 \
  --bias_init_scale 0.05 --use_residual_prior True --residual_bias_reg_weight 0.0001
run_one residual_b005_reg1e-3 \
  --bias_init_scale 0.05 --use_residual_prior True --residual_bias_reg_weight 0.001

# 4) Sequence-conditional long-tail gate.
run_one tailgate_b005_s002 \
  --bias_init_scale 0.05 --use_tail_gate True --tail_gate_scale 0.02
run_one tailgate_b005_s005 \
  --bias_init_scale 0.05 --use_tail_gate True --tail_gate_scale 0.05

# Combined variants.
run_one combo_ctx_pop_tail \
  --bias_init_scale 0.05 --use_context_bias True --context_bias_scale 2.0 \
  --use_pop_temp True --pop_temp_scale -0.05 --use_tail_gate True --tail_gate_scale 0.02
run_one combo_all \
  --bias_init_scale 0.05 --use_context_bias True --context_bias_scale 2.0 \
  --use_pop_temp True --pop_temp_scale -0.05 --use_tail_gate True --tail_gate_scale 0.02 \
  --use_residual_prior True --residual_bias_reg_weight 0.001

conda run --no-capture-output -n recbole python - <<'PY'
import ast
import re
from pathlib import Path

root = Path("log_runs/dynamic_calibration_h256_2080ti")
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
