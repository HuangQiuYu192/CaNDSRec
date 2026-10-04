#!/usr/bin/env python3
"""Locate where angular re-ranking recovers targets in the original dot ranking."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


FIELDS = (
    "n", "share", "angular_top10_rate", "angular_rank", "rank_delta",
    "recovered_competitors", "recovered_angular_margin", "recovered_log_norm_ratio",
    "introduced_competitors", "introduced_angular_disadvantage", "introduced_log_norm_ratio",
)


def dot_stratum(rank: int) -> str:
    if rank <= 10: return "dot_1_10"
    if rank <= 20: return "dot_11_20"
    if rank <= 50: return "dot_21_50"
    if rank <= 100: return "dot_51_100"
    return "dot_gt100"


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_seed: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for file in sorted(args.input_dir.glob("*_seed*_*.csv")):
        with file.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_seed[(row["dataset"], row["trained_geometry"], row["seed"])].append(row)

    per_seed = []
    for (dataset, geometry, seed), rows in by_seed.items():
        total = len(rows)
        strata: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            strata[dot_stratum(int(row["dot_rank"]))].append(row)
        for name, subset in strata.items():
            def avg(field: str) -> float:
                return mean(float(row[field]) for row in subset)
            per_seed.append({
                "dataset": dataset, "trained_geometry": geometry, "seed": int(seed), "dot_rank_stratum": name,
                "n": len(subset), "share": len(subset) / total,
                "angular_top10_rate": mean(int(row["angular_rank"]) <= 10 for row in subset),
                "angular_rank": avg("angular_rank"), "rank_delta": avg("rank_delta"),
                "recovered_competitors": avg("recovered_competitors"),
                "recovered_angular_margin": avg("recovered_angular_margin"),
                "recovered_log_norm_ratio": avg("recovered_log_norm_ratio"),
                "introduced_competitors": avg("introduced_competitors"),
                "introduced_angular_disadvantage": avg("introduced_angular_disadvantage"),
                "introduced_log_norm_ratio": avg("introduced_log_norm_ratio"),
            })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["dataset"], row["trained_geometry"], row["dot_rank_stratum"])].append(row)
    output = []
    order = {"dot_1_10": 0, "dot_11_20": 1, "dot_21_50": 2, "dot_51_100": 3, "dot_gt100": 4}
    for (dataset, geometry, stratum), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], order[x[0][2]])):
        record = {"dataset": dataset, "trained_geometry": geometry, "dot_rank_stratum": stratum}
        for field in FIELDS:
            m, s = mean_std([float(row[field]) for row in rows])
            record[f"{field}_mean"] = m; record[f"{field}_std"] = s
        output.append(record)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
