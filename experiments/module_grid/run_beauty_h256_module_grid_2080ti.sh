#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_NAME="module_grid_h256_2080ti"
LOG_DIR="${ROOT}/log_runs/${EXP_NAME}"
CKPT_ROOT="${ROOT}/ckpt/${EXP_NAME}"
SUMMARY="${LOG_DIR}/summary.tsv"
MASTER_LOG="${LOG_DIR}/master_$(date +%Y%m%d_%H%M%S).log"
GPUS=(0 1 3)

mkdir -p "${LOG_DIR}" "${CKPT_ROOT}" "${ROOT}/experiments/module_grid"

run_one() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="${LOG_DIR}/${name}.log"
  local ckpt="${CKPT_ROOT}/${name}"
  mkdir -p "${ckpt}"
  {
    echo "[$(date '+%F %T')] START name=${name} gpu=${gpu}"
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
      "$@"
    echo "[$(date '+%F %T')] END name=${name} gpu=${gpu}"
  } > "${log}" 2>&1
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= ${#GPUS[@]} )); do
    sleep 20
  done
}

next_gpu() {
  local idx=$(( $(jobs -rp | wc -l) % ${#GPUS[@]} ))
  echo "${GPUS[$idx]}"
}

launch() {
  local name="$1"
  shift
  wait_for_slot
  local gpu
  gpu="$(next_gpu)"
  echo "[$(date '+%F %T')] launch ${name} on gpu ${gpu}" | tee -a "${MASTER_LOG}"
  run_one "${gpu}" "${name}" "$@" &
  sleep 8
}

# Pure cosine temperature grid.
for temp in 5.0 7.5 10.0 15.0 20.0; do
  launch "cosine_temp${temp}" \
    --temperature "${temp}" \
    --use_norm_residual False \
    --use_weighted_ce False \
    --use_cands_smoothing False
done

# Norm residual only. Keep temp fixed to the known strong value first.
for beta in 0.05 0.10 0.20 0.50 1.00; do
  launch "normres_t10_beta${beta}" \
    --temperature 10.0 \
    --use_norm_residual True \
    --norm_beta "${beta}" \
    --use_weighted_ce False \
    --use_cands_smoothing False
done

# Weighted CE only. Keep temp fixed to the known strong value first.
for gamma in 0.02 0.05 0.10 0.20 0.50; do
  launch "weightedce_t10_gamma${gamma}" \
    --temperature 10.0 \
    --use_norm_residual False \
    --use_weighted_ce True \
    --weight_gamma "${gamma}" \
    --use_cands_smoothing False
done

# Joint grid around mild values. This tests whether the modules need co-tuning.
for beta in 0.05 0.10 0.20 0.50; do
  for gamma in 0.02 0.05 0.10 0.20; do
    launch "combo_t10_beta${beta}_gamma${gamma}" \
      --temperature 10.0 \
      --use_norm_residual True \
      --norm_beta "${beta}" \
      --use_weighted_ce True \
      --weight_gamma "${gamma}" \
      --use_cands_smoothing False
  done
done

echo "[$(date '+%F %T')] waiting for all runs" | tee -a "${MASTER_LOG}"
wait

python - <<'PY' "${LOG_DIR}" "${SUMMARY}"
import ast
import glob
import os
import re
import sys

log_dir, summary = sys.argv[1:3]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    name = os.path.basename(path)[:-4]
    if name.startswith("master_"):
        continue
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    status = "ok"
    if "Traceback" in text or "ERROR conda.cli" in text or "OutOfMemory" in text:
        status = "error"
    valid = {}
    test = {}
    best_epoch = ""
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

metric_order = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
with open(summary, "w", encoding="utf-8") as f:
    f.write("name\tstatus\tbest_epoch")
    for k in metric_order:
        f.write(f"\tvalid_{k}")
    for k in metric_order:
        f.write(f"\ttest_{k}")
    f.write("\tlog\n")
    for name, status, best_epoch, valid, test, path in rows:
        f.write(f"{name}\t{status}\t{best_epoch}")
        for k in metric_order:
            f.write(f"\t{valid.get(k, '')}")
        for k in metric_order:
            f.write(f"\t{test.get(k, '')}")
        f.write(f"\t{path}\n")

ok_rows = [r for r in rows if r[1] == "ok" and r[4]]
ok_rows.sort(key=lambda r: (r[4].get("ndcg@10", -1), r[4].get("recall@10", -1)), reverse=True)
print("Top by test ndcg@10:")
for row in ok_rows[:10]:
    name, status, best_epoch, valid, test, path = row
    print(name, "best_epoch=", best_epoch, "test=", test)
PY

echo "[$(date '+%F %T')] all grid runs finished; summary=${SUMMARY}" | tee -a "${MASTER_LOG}"
