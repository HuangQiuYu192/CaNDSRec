#!/usr/bin/env python3
"""Held-out three-seed four-way score-geometry comparison for Sports/Toys."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


PAIR_RE = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
TEST_RE = re.compile(r"test result:\s*OrderedDict\(\[(.*?)\]\)", re.S)
TEMPERATURES = {"Sports": "0.125", "Toys": "0.0625"}


def read_test(path: Path) -> dict[str, float]:
    matches = TEST_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No test result in {path}")
    return {key: float(value) for key, value in PAIR_RE.findall(matches[-1])}


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-root", type=Path, default=Path("log_runs/www27_matched_seed_grid"))
    parser.add_argument("--sequence-root", type=Path, default=Path("log_runs/www27_sequence_geometry_probe_gpu1"))
    parser.add_argument("--scaled-root", type=Path, default=Path("log_runs/www27_scaled_dot_confirmation_gpu1"))
    parser.add_argument("--datasets", nargs="+", default=["Sports", "Toys"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--out", type=Path, default=Path("analysis_results/www27_scaled_dot_confirmation/summary.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        tau = TEMPERATURES[dataset]
        for seed in args.seeds:
            dot = read_test(args.matched_root / dataset / f"seed{seed}" / "SASRec.log")
            sequence = read_test(args.sequence_root / dataset / f"seed{seed}" / "sequence_tau10.log")
            joint = read_test(args.matched_root / dataset / f"seed{seed}" / "CANDSSASRec.log")
            scaled = read_test(args.scaled_root / dataset / f"seed{seed}" / f"scaled_dot_tau{tau}.log")
            for metric in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
                rows.append({"dataset": dataset, "seed": seed, "metric": metric, "scaled_dot_tau": float(tau),
                             "dot": dot[metric], "scaled_dot": scaled[metric], "sequence": sequence[metric], "joint": joint[metric],
                             "scaled_minus_dot": scaled[metric] - dot[metric],
                             "sequence_minus_scaled": sequence[metric] - scaled[metric],
                             "joint_minus_scaled": joint[metric] - scaled[metric]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Held-out Scaled-Dot confirmation on Sports and Toys", "",
        "Scaled-Dot temperatures were locked by seed-2026 validation: Sports=.125 and Toys=.0625. "
        "All reported rows use the independent seeds 2023/2024/2025.", "",
        "| dataset | metric | dot | scaled-dot | sequence | joint | scaled - dot | sequence - scaled | joint - scaled |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in args.datasets:
        for metric in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
            group = [row for row in rows if row["dataset"] == dataset and row["metric"] == metric]
            def formatted(key: str) -> str:
                mean, std = mean_sd([float(row[key]) for row in group])
                return f"{mean:.4f} ± {std:.4f}"
            lines.append(f"| {dataset} | {metric} | {formatted('dot')} | {formatted('scaled_dot')} | {formatted('sequence')} | {formatted('joint')} | {formatted('scaled_minus_dot')} | {formatted('sequence_minus_scaled')} | {formatted('joint_minus_scaled')} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
