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
conda run --no-capture-output -n recbole python - <<'PY'
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
with (root / "summary.md").open("w", encoding="utf-8") as f:
    f.write("# BSARec dimension--geometry probe on Beauty\n\n")
    f.write("Frozen checkpoint analysis, seed 2025. No value in this table was used to select a dimension or temperature. `alpha=0` is angular scoring and `alpha=1` restores the dot-product candidate order for the same checkpoint.\n\n")
    f.write("| model | d | NDCG@10 angular | NDCG@10 radial | radial drop | item log-norm std. | Top-10 Jaccard | local inversion | angular margin |\n")
    f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in sorted(rows, key=lambda r: (r["model"], int(r["dimension"]))):
        angular = float(row["ndcg10_alpha_0"]); radial = float(row["ndcg10_alpha_1"])
        f.write(f"| {row['model']} | {row['dimension']} | {angular:.4f} | {radial:.4f} | {angular-radial:+.4f} | "
                f"{float(row['item_log_norm_std']):.4f} | {float(row['top10_jaccard_dot_vs_angular']):.4f} | "
                f"{float(row['local_inversion_rate']):.4f} | {float(row['angular_margin']):+.4f} |\n")
print(f"wrote {root / 'summary.csv'}")
print(f"wrote {root / 'summary.md'}")
PY
echo ALL_DONE
