#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssh_user/code/25-HuangQiuyu/LongTailRec"
ENV_SETUP="/home/ssh_user/miniconda3/etc/profile.d/conda.sh"
EXP_NAME="module_grid_h256_2080ti_rerun"
LOG_DIR="${ROOT}/log_runs/${EXP_NAME}"
CKPT_ROOT="${ROOT}/ckpt/${EXP_NAME}"
MASTER_LOG="${LOG_DIR}/master_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="${LOG_DIR}/summary.tsv"

mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

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

worker0() {
  local gpu=0
  run_one "${gpu}" "normres_t10_beta0.05" --temperature 10.0 --use_norm_residual True --norm_beta 0.05 --use_weighted_ce False --use_cands_smoothing False
  run_one "${gpu}" "normres_t10_beta0.10" --temperature 10.0 --use_norm_residual True --norm_beta 0.10 --use_weighted_ce False --use_cands_smoothing False
  run_one "${gpu}" "normres_t10_beta0.20" --temperature 10.0 --use_norm_residual True --norm_beta 0.20 --use_weighted_ce False --use_cands_smoothing False
  run_one "${gpu}" "normres_t10_beta0.50" --temperature 10.0 --use_norm_residual True --norm_beta 0.50 --use_weighted_ce False --use_cands_smoothing False
  run_one "${gpu}" "normres_t10_beta1.00" --temperature 10.0 --use_norm_residual True --norm_beta 1.00 --use_weighted_ce False --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.05_gamma0.02" --temperature 10.0 --use_norm_residual True --norm_beta 0.05 --use_weighted_ce True --weight_gamma 0.02 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.05_gamma0.05" --temperature 10.0 --use_norm_residual True --norm_beta 0.05 --use_weighted_ce True --weight_gamma 0.05 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.05_gamma0.10" --temperature 10.0 --use_norm_residual True --norm_beta 0.05 --use_weighted_ce True --weight_gamma 0.10 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.05_gamma0.20" --temperature 10.0 --use_norm_residual True --norm_beta 0.05 --use_weighted_ce True --weight_gamma 0.20 --use_cands_smoothing False
}

worker1() {
  local gpu=1
  run_one "${gpu}" "weightedce_t10_gamma0.02" --temperature 10.0 --use_norm_residual False --use_weighted_ce True --weight_gamma 0.02 --use_cands_smoothing False
  run_one "${gpu}" "weightedce_t10_gamma0.05" --temperature 10.0 --use_norm_residual False --use_weighted_ce True --weight_gamma 0.05 --use_cands_smoothing False
  run_one "${gpu}" "weightedce_t10_gamma0.10" --temperature 10.0 --use_norm_residual False --use_weighted_ce True --weight_gamma 0.10 --use_cands_smoothing False
  run_one "${gpu}" "weightedce_t10_gamma0.20" --temperature 10.0 --use_norm_residual False --use_weighted_ce True --weight_gamma 0.20 --use_cands_smoothing False
  run_one "${gpu}" "weightedce_t10_gamma0.50" --temperature 10.0 --use_norm_residual False --use_weighted_ce True --weight_gamma 0.50 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.10_gamma0.02" --temperature 10.0 --use_norm_residual True --norm_beta 0.10 --use_weighted_ce True --weight_gamma 0.02 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.10_gamma0.05" --temperature 10.0 --use_norm_residual True --norm_beta 0.10 --use_weighted_ce True --weight_gamma 0.05 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.10_gamma0.10" --temperature 10.0 --use_norm_residual True --norm_beta 0.10 --use_weighted_ce True --weight_gamma 0.10 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.10_gamma0.20" --temperature 10.0 --use_norm_residual True --norm_beta 0.10 --use_weighted_ce True --weight_gamma 0.20 --use_cands_smoothing False
}

worker3() {
  local gpu=3
  run_one "${gpu}" "combo_t10_beta0.20_gamma0.02" --temperature 10.0 --use_norm_residual True --norm_beta 0.20 --use_weighted_ce True --weight_gamma 0.02 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.20_gamma0.05" --temperature 10.0 --use_norm_residual True --norm_beta 0.20 --use_weighted_ce True --weight_gamma 0.05 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.20_gamma0.10" --temperature 10.0 --use_norm_residual True --norm_beta 0.20 --use_weighted_ce True --weight_gamma 0.10 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.20_gamma0.20" --temperature 10.0 --use_norm_residual True --norm_beta 0.20 --use_weighted_ce True --weight_gamma 0.20 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.50_gamma0.02" --temperature 10.0 --use_norm_residual True --norm_beta 0.50 --use_weighted_ce True --weight_gamma 0.02 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.50_gamma0.05" --temperature 10.0 --use_norm_residual True --norm_beta 0.50 --use_weighted_ce True --weight_gamma 0.05 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.50_gamma0.10" --temperature 10.0 --use_norm_residual True --norm_beta 0.50 --use_weighted_ce True --weight_gamma 0.10 --use_cands_smoothing False
  run_one "${gpu}" "combo_t10_beta0.50_gamma0.20" --temperature 10.0 --use_norm_residual True --norm_beta 0.50 --use_weighted_ce True --weight_gamma 0.20 --use_cands_smoothing False
}

echo "[$(date '+%F %T')] launch module grid rerun workers" | tee -a "${MASTER_LOG}"
worker0 &
pid0=$!
worker1 &
pid1=$!
worker3 &
pid3=$!
wait "${pid0}" "${pid1}" "${pid3}"

python - <<'PY' "${LOG_DIR}" "${SUMMARY}"
import ast
import glob
import os
import re
import sys

log_dir, summary = sys.argv[1:3]
metric_order = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    if os.path.basename(path).startswith("master_"):
        continue
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    status = "ok"
    if "Traceback" in text or "OutOfMemory" in text or "CUDA error" in text or "ERROR conda.cli" in text:
        status = "error"
    best_epoch = ""
    valid = {}
    test = {}
    match = re.search(r"Finished training, best eval result in epoch (\d+)", text)
    if match:
        best_epoch = match.group(1)
    match = re.search(r"best valid result: OrderedDict\((\[.*?\])\)", text)
    if match:
        valid = dict(ast.literal_eval(match.group(1)))
    match = re.search(r"test result: OrderedDict\((\[.*?\])\)", text)
    if match:
        test = dict(ast.literal_eval(match.group(1)))
    rows.append((os.path.basename(path)[:-4], status, best_epoch, valid, test, path))

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
PY

echo "[$(date '+%F %T')] all module grid rerun workers finished; summary=${SUMMARY}" | tee -a "${MASTER_LOG}"
