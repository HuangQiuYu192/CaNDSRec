#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_NAME="temp_fine_h256_2080ti"
LOG_DIR="${ROOT}/log_runs/${EXP_NAME}"
CKPT_ROOT="${ROOT}/ckpt/${EXP_NAME}"
SUMMARY="${LOG_DIR}/summary.tsv"

mkdir -p "${LOG_DIR}" "${CKPT_ROOT}" "${ROOT}/experiments/temp_fine"

run_one() {
  local gpu="$1"
  local temp="$2"
  local name="cosine_temp${temp}"
  local log="${LOG_DIR}/${name}.log"
  local ckpt="${CKPT_ROOT}/${name}"
  mkdir -p "${ckpt}"
  {
    echo "[$(date '+%F %T')] START ${name} gpu=${gpu}"
    cd "${ROOT}"
    source "${ENV_SETUP}"
    conda run --no-capture-output -n recbole python main.py \
      --dataset Beauty \
      --gpu_id "${gpu}" \
      --hidden_size 256 \
      --inner_size 1024 \
      --train_batch_size 1024 \
      --eval_batch_size 1024 \
      --model CANDSSASRec \
      --checkpoint_dir "${ckpt}" \
      --temperature "${temp}"
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

ok = [r for r in rows if r[1] == "ok" and r[4]]
ok.sort(key=lambda r: (r[4].get("recall@10", 0) + r[4].get("ndcg@10", 0), r[4].get("recall@20", 0), r[4].get("ndcg@20", 0)), reverse=True)
print("name\tbest_epoch\trecall@5\trecall@10\trecall@20\tndcg@5\tndcg@10\tndcg@20")
for name, status, best_epoch, valid, test, path in ok:
    print("\t".join(str(x) for x in [name, best_epoch, test.get("recall@5"), test.get("recall@10"), test.get("recall@20"), test.get("ndcg@5"), test.get("ndcg@10"), test.get("ndcg@20")]))
PY
}

worker0() {
  run_one 0 8.0
  run_one 0 9.0
  run_one 0 10.5
  run_one 0 11.5
}

worker3() {
  run_one 3 8.5
  run_one 3 9.5
  run_one 3 11.0
  run_one 3 12.0
}

worker0 &
pid0=$!
worker3 &
pid3=$!
wait "${pid0}" "${pid3}"
summarize | tee "${LOG_DIR}/top_results.txt"
