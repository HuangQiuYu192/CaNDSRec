#!/usr/bin/env python3
"""Aggregate per-seed candidate-competition decompositions without pooling seeds."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("n", "share", "rank_delta", "angular_margin", "log_norm_ratio")


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_seed: dict[tuple, list[dict]] = defaultdict(list)
    for file in sorted(args.input_dir.glob("*_seed*_*.csv")):
        with file.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_seed[(row["dataset"], row["trained_geometry"], row["seed"])].append(row)

    per_seed: list[dict] = []
    for (dataset, geometry, seed), rows in by_seed.items():
        total = len(rows)
        by_transition: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_transition[row["transition"]].append(row)
        for name, subset in by_transition.items():
            per_seed.append({
                "dataset": dataset, "trained_geometry": geometry, "seed": int(seed), "transition": name,
                "n": len(subset), "share": len(subset) / total,
                "rank_delta": mean(float(row["rank_delta"]) for row in subset),
                "angular_margin": mean(float(row["angular_margin_vs_dot_competitor"]) for row in subset),
                "log_norm_ratio": mean(float(row["log_target_over_competitor_norm"]) for row in subset),
            })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["dataset"], row["trained_geometry"], row["transition"])].append(row)
    output = []
    for (dataset, geometry, transition_name), rows in sorted(grouped.items()):
        record = {"dataset": dataset, "trained_geometry": geometry, "transition": transition_name}
        for metric in METRICS:
            m, s = mean_std([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = m; record[f"{metric}_std"] = s
        output.append(record)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
