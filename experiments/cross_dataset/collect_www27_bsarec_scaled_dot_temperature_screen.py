"""Read validation results only for the BSARec scaled-dot temperature pilot."""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


def valid_result(log: Path) -> dict[str, float]:
    text = log.read_text(encoding="utf-8", errors="replace")
    records = re.findall(r"best valid result:\s*OrderedDict\((\[.*?\])\)", text)
    if not records:
        raise ValueError(f"No completed validation result in {log}")
    return dict(ast.literal_eval(records[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dimension", type=int, default=256)
    args = parser.parse_args()
    logs = args.root / "log_runs/www27_bsarec_scaled_dot_temperature_screen/Beauty" / f"seed{args.seed}"
    rows = []
    for log in logs.glob(f"ScaledDotBSARec_d{args.dimension}_tau*_a0.5_c9_h2_lr0.001.log"):
        match = re.search(r"_tau([^_]+)_", log.name)
        value = valid_result(log)
        rows.append({"temperature": float(match.group(1)), "valid_ndcg@10": value["ndcg@10"], "valid_recall@10": value["recall@10"]})
    rows.sort(key=lambda r: r["temperature"])
    out = args.root / "analysis_results/www27_bsarec_scaled_dot_temperature_screen"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["valid_ndcg@10"])
    with (out / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# BSARec scaled-dot validation-only temperature screen\n\n")
        f.write("Only validation metrics below are used to select the global dot-product scale. The development seed is excluded from final test reporting.\n\n")
        f.write("| temperature | Validation NDCG@10 | Validation Recall@10 |\n| ---: | ---: | ---: |\n")
        for row in rows:
            f.write(f"| {row['temperature']:.4g} | {row['valid_ndcg@10']:.4f} | {row['valid_recall@10']:.4f} |\n")
        f.write(f"\nSelected by validation NDCG@10: temperature={best['temperature']:.4g}.\n")
    print(f"wrote {out / 'summary.csv'} and summary.md")


if __name__ == "__main__":
    main()
