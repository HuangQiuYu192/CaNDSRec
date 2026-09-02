#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_NAME="bias_calibration_h256_2080ti"
LOG_DIR="${ROOT}/log_runs/${EXP_NAME}"
CKPT_ROOT="${ROOT}/ckpt/${EXP_NAME}"
SUMMARY="${LOG_DIR}/summary.tsv"
GPUS=(0 1 3)

mkdir -p "${LOG_DIR}" "${CKPT_ROOT}" "${ROOT}/experiments/bias_calibration"

run_one() {
  local gpu="$1"
  local name="$2"
  local reg="$3"
  local init="$4"
  local log="${LOG_DIR}/${name}.log"
  local ckpt="${CKPT_ROOT}/${name}"
  mkdir -p "${ckpt}"
  {
    echo "[$(date '+%F %T')] START ${name} gpu=${gpu} reg=${reg} init=${init}"
    cd "${ROOT}"
    source "${ENV_SETUP}"
    conda run --no-capture-output -n recbole python main.py \
      --dataset Beauty \
      --gpu_id "${gpu}" \
      --hidden_size 256 \
      --inner_size 1024 \
      --train_batch_size 1024 \
      --eval_batch_size 1024 \
      --model CalibratedCANDSSASRec \
      --checkpoint_dir "${ckpt}" \
      --temperature 10.0 \
      --bias_reg_weight "${reg}" \
      --bias_init_scale "${init}"
    echo "[$(date '+%F %T')] END ${name}"
  } > "${log}" 2>&1
}

summarize() {
  /home/ssh_user/miniconda3/envs/recbole/bin/python - <<'PY' "${LOG_DIR}" "${SUMMARY}"
import ast
import csv
import glob
import os
import re
import sys

log_dir, summary = sys.argv[1:3]
metric_order = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    text = open(path, encoding="utf-8", errors="ignore").read()
    name = os.path.basename(path)[:-4]
    status = "error" if re.search(r"Traceback|OutOfMemory|CUDA error|ERROR conda.cli", text) else "ok"
    best_epoch = ""
    valid = {}
    test = {}
    m = re.search(r"Finished training, best eval result in epoch (\d+)", text)
    if m:
        best_epoch = m.group(1)
    m = re.search(r"best valid result: OrderedDict\((\[.*?\])\)", text)
    if m:
        valid = dict(ast.literal_eval(m.group(1)))
    m = re.search(r"test result: OrderedDict\((\[.*?\])\)", text)
    if m:
        test = dict(ast.literal_eval(m.group(1)))
    rows.append((name, status, best_epoch, valid, test, path))

with open(summary, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["name", "status", "best_epoch"] + ["valid_" + k for k in metric_order] + ["test_" + k for k in metric_order] + ["log"])
    for name, status, best_epoch, valid, test, path in rows:
        w.writerow([name, status, best_epoch] + [valid.get(k, "") for k in metric_order] + [test.get(k, "") for k in metric_order] + [path])

baseline = {"recall@10": 0.0899, "ndcg@10": 0.0460, "recall@20": 0.1300, "ndcg@20": 0.0561}
ok = [r for r in rows if r[1] == "ok" and r[4]]
ok.sort(key=lambda r: (r[4].get("recall@10", -1) > baseline["recall@10"],
                       r[4].get("ndcg@10", -1) > baseline["ndcg@10"],
                       r[4].get("recall@20", -1) > baseline["recall@20"],
                       r[4].get("ndcg@20", -1) > baseline["ndcg@20"],
                       r[4].get("recall@10", 0) + r[4].get("ndcg@10", 0)),
        reverse=True)
print("name\tbest_epoch\trecall@5\trecall@10\trecall@20\tndcg@5\tndcg@10\tndcg@20")
for name, status, best_epoch, valid, test, path in ok[:12]:
    print("\t".join(str(x) for x in [name, best_epoch, test.get("recall@5"), test.get("recall@10"), test.get("recall@20"), test.get("ndcg@5"), test.get("ndcg@10"), test.get("ndcg@20")]))
PY
}

worker() {
  local gpu="$1"
  shift
  while (($#)); do
    local spec="$1"
    shift
    IFS=: read -r name reg init <<< "${spec}"
    run_one "${gpu}" "${name}" "${reg}" "${init}"
  done
}

worker 0 \
  "bias_reg0_init0:0.0:0.0" \
  "bias_reg1e-6_init0:0.000001:0.0" \
  "bias_reg1e-5_init0:0.00001:0.0" \
  "bias_reg1e-4_init0:0.0001:0.0" &
pid0=$!

worker 1 \
  "bias_reg1e-3_init0:0.001:0.0" \
  "bias_reg1e-5_init0.02:0.00001:0.02" \
  "bias_reg1e-4_init0.02:0.0001:0.02" \
  "bias_reg1e-3_init0.02:0.001:0.02" &
pid1=$!

worker 3 \
  "bias_reg1e-5_init0.05:0.00001:0.05" \
  "bias_reg1e-4_init0.05:0.0001:0.05" \
  "bias_reg1e-3_init0.05:0.001:0.05" \
  "bias_reg1e-4_init-0.02:0.0001:-0.02" &
pid3=$!

wait "${pid0}" "${pid1}" "${pid3}"
summarize | tee "${LOG_DIR}/top_results.txt"
