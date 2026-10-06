#!/usr/bin/env bash
set -euo pipefail

# Read-only frozen-checkpoint probe for the BSARec d=64/128/256 capacity grid.
# The checkpoints were trained previously; this script uses GPU 1 only and
# performs no optimization or hyperparameter selection.
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
source "${CONDA_SH:-/home/ssh_user/miniconda3/etc/profile.d/conda.sh}"
cd "$ROOT"
GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/analysis_results/www27_bsarec_dimension_geometry}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpt/www27_bsarec_dimension_validation/Beauty/seed2025}"

for dim in 64 128 256; do
  for model in BSARec CANDSBSARec; do
    tag="${model}_d${dim}_a0.5_c9_h2_lr0.001"
    ckpt=$(find "$CKPT_ROOT/$tag" -maxdepth 1 -type f -name '*.pth' | head -n 1)
    test -n "$ckpt"
    out="$OUT_DIR/${model}_d${dim}_seed2025.csv"
    if test -s "$out"; then continue; fi
    echo "START model=$model dimension=$dim"
    conda run --no-capture-output -n recbole python experiments/cross_dataset/analyze_www27_bsarec_dimension_geometry.py \
      --model "$model" --checkpoint "$ckpt" --dimension "$dim" --gpu_id "$GPU_ID" --out "$out"
  done
done
python - <<'PY'
import csv
from pathlib import Path
root = Path("analysis_results/www27_bsarec_dimension_geometry")
rows = []
for path in sorted(root.glob("*.csv")):
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
fields = list(rows[0]) if rows else []
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"wrote {root / 'summary.csv'}")
PY
echo ALL_DONE
