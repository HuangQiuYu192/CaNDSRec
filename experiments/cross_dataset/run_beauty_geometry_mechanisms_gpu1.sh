#!/usr/bin/env bash
set -euo pipefail

# Full-test, read-only diagnostic for the paired Beauty geometry checkpoints.
# It uses GPU 1 only, never calls backward(), and does not create checkpoints.

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_SH="${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-recbole}"
GPU_ID="${GPU_ID:-1}"
SEEDS="${SEEDS:-2023 2024 2025}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis_results/beauty_geometry_mechanisms}"
OLD_CKPT_ROOT="${OLD_CKPT_ROOT:-$ROOT/ckpt/beauty_radial_power_grid_gpu12}"
TAU4_CKPT_ROOT="${TAU4_CKPT_ROOT:-$ROOT/ckpt/beauty_sequence_tau4_confirmation_gpu1}"
RERUN="${RERUN:-false}"

if [ ! -f "$CONDA_SH" ]; then echo "Missing CONDA_SH=$CONDA_SH" >&2; exit 127; fi
source "$CONDA_SH"
cd "$ROOT"

run_one() {
  local seed="$1" tag="$2" model="$3" tau="$4" geometry="$5" seq_power="$6" item_power="$7" root="$8"
  local checkpoint out
  checkpoint=$(find "$root/seed${seed}/${tag}_tau${tau}" -maxdepth 1 -type f -name '*.pth' | head -n 1)
  out="$OUT_ROOT/seed${seed}_${tag}/metrics.json"
  if [ -z "$checkpoint" ]; then echo "Missing checkpoint seed=$seed tag=$tag tau=$tau root=$root" >&2; exit 2; fi
  if [ "$RERUN" != true ] && [ -s "$out" ]; then echo "SKIP seed=$seed tag=$tag" | tee -a "$OUT_ROOT/driver.log"; return 0; fi
  echo "START seed=$seed tag=$tag gpu=$GPU_ID" | tee -a "$OUT_ROOT/driver.log"
  conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/analyze_beauty_geometry_mechanisms.py \
    --checkpoint "$checkpoint" --model "$model" --tag "$tag" --geometry "$geometry" --temperature "$tau" \
    --sequence_norm_power "$seq_power" --item_norm_power "$item_power" --seed "$seed" --gpu_id "$GPU_ID" \
    --out_dir "$OUT_ROOT/seed${seed}_${tag}" > "$OUT_ROOT/seed${seed}_${tag}.log" 2>&1
  echo "DONE seed=$seed tag=$tag" | tee -a "$OUT_ROOT/driver.log"
}

mkdir -p "$OUT_ROOT"
for seed in $SEEDS; do
  run_one "$seed" dot SASRec 1 dot 0 0 "$OLD_CKPT_ROOT"
  run_one "$seed" sequence GeometrySASRec 4 sequence 1 0 "$TAU4_CKPT_ROOT"
  run_one "$seed" both GeometrySASRec 10 both 1 1 "$OLD_CKPT_ROOT"
done
conda run --no-capture-output -n "$CONDA_ENV" python experiments/cross_dataset/collect_beauty_geometry_mechanisms.py \
  --input_root "$OUT_ROOT" --out_prefix "$OUT_ROOT/summary"
echo ALL_DONE | tee -a "$OUT_ROOT/driver.log"
