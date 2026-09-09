#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
for dataset in Beauty Yelp-S3Rec; do
  max_len=50; batch=1024
  out="log_runs/www27_backbone_pilot/$dataset/seed2025"; mkdir -p "$out"
  run() { local gpu="$1" model="$2"; local log="$out/$model.log"; [ -s "$log" ] && grep -q 'MODEL TEST' "$log" && return; conda run --no-capture-output -n recbole python main.py --model "$model" --dataset "$dataset" --gpu_id "$gpu" --seed 2025 --hidden_size 64 --n_layers 2 --n_heads 2 --inner_size 256 --c 4 --alpha 0.5 --temperature 10 --epochs 300 --stopping_step 10 --learning_rate 0.001 --train_batch_size "$batch" --eval_batch_size 512 --max_item_list_length "$max_len" --checkpoint_dir "ckpt/www27_backbone_pilot/$dataset/seed2025/$model" --verbose True --show_progress True > "$log" 2>&1; }
  run 0 BSARec & p0=$!; run 1 CANDSBSARec & p1=$!; wait "$p0"; wait "$p1"
done
echo ALL_DONE
