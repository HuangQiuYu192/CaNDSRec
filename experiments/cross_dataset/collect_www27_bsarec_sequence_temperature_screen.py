"""Collect validation-only temperature selection for BSARec sequence normalization."""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dimension", type=int, default=256)
    args = parser.parse_args()
    logs = args.root / "log_runs/www27_bsarec_sequence_temperature_screen/Beauty" / f"seed{args.seed}"
    rows = []
    for path in logs.glob(f"GeometryBSARec_sequence_d{args.dimension}_tau*_a0.5_c9_h2_lr0.001.log"):
        records = re.findall(r"best valid result:\s*OrderedDict\((\[.*?\])\)", path.read_text(encoding="utf-8", errors="replace"))
        if not records:
            continue
        value = dict(ast.literal_eval(records[-1]))
        tau = float(re.search(r"_tau([^_]+)_", path.name).group(1))
        rows.append({"temperature": tau, "valid_ndcg@10": value["ndcg@10"], "valid_recall@10": value["recall@10"]})
    if not rows:
        raise SystemExit("No completed validation logs")
    rows.sort(key=lambda row: row["temperature"])
    out = args.root / "analysis_results/www27_bsarec_sequence_temperature_screen"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best = max(rows, key=lambda row: row["valid_ndcg@10"])
    lines = ["# BSARec sequence-only validation temperature screen", "", "Only seed-2026 validation selects the scale; its test metric is excluded from all claims.", "", "| temperature | Validation NDCG@10 | Validation Recall@10 |", "| ---: | ---: | ---: |"]
    lines.extend(f"| {row['temperature']:.4g} | {row['valid_ndcg@10']:.4f} | {row['valid_recall@10']:.4f} |" for row in rows)
    lines.append(f"\nSelected by validation NDCG@10: temperature={best['temperature']:.4g}.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.csv'} and summary.md")


if __name__ == "__main__":
    main()
