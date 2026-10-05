#!/usr/bin/env python3
"""Aggregate frozen radial reintroduction curves across random seeds."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "mean_rank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for file in sorted(args.input_dir.glob("*_seed*_*.csv")):
        with file.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["dataset"], row["trained_geometry"], row["item_radial_exponent"])].append(row)
    output = []
    for (dataset, geometry, alpha), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], float(item[0][2]))):
        record = {"dataset": dataset, "trained_geometry": geometry, "item_radial_exponent": float(alpha), "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
