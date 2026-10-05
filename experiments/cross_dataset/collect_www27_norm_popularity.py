#!/usr/bin/env python3
"""Aggregate norm/popularity diagnostics across seeds without pooling them."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("norm_logpop_spearman", "norm_logpop_pearson", "recovered_mean_log_pop_gap", "recovered_mean_log_norm_gap", "introduced_mean_log_pop_gap", "introduced_mean_log_norm_gap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for file in sorted(args.input_dir.glob("*_seed*_*.csv")):
        with file.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
            grouped[(row["dataset"], row["trained_geometry"])].append(row)
    output = []
    for (dataset, geometry), rows in sorted(grouped.items()):
        record = {"dataset": dataset, "trained_geometry": geometry, "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.
        output.append(record)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
