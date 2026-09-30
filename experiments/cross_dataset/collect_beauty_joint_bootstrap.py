#!/usr/bin/env python3
"""Summarize paired user-bootstrap intervals across Beauty seeds."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_root.glob("seed*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No bootstrap CSV files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Beauty paired user bootstrap: SASRec vs joint normalization", "", "Each interval resamples paired leave-one-out test users within a fixed seed; it is not used for model or temperature selection.", "",
             "| seed | metric | base | joint | delta | 95% paired bootstrap CI |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        metric = f"Recall@{row['k']}" if row['k'] != '10' else "Recall@10 / NDCG@10"
        lines.append(f"| {row['seed']} | Recall@{row['k']} | {float(row['base_recall']):.4f} | {float(row['cands_recall']):.4f} | {float(row['delta_recall']):+.4f} | [{float(row['delta_recall_ci_low']):+.4f}, {float(row['delta_recall_ci_high']):+.4f}] |")
        lines.append(f"| {row['seed']} | NDCG@{row['k']} | {float(row['base_ndcg']):.4f} | {float(row['cands_ndcg']):.4f} | {float(row['delta_ndcg']):+.4f} | [{float(row['delta_ndcg_ci_low']):+.4f}, {float(row['delta_ndcg_ci_high']):+.4f}] |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
