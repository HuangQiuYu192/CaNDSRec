#!/usr/bin/env python3
"""Paired three-seed summary: dot, sequence-only and joint on Sports/Toys."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


PAIR_RE = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
TEST_RE = re.compile(r"test result:\s*OrderedDict\(\[(.*?)\]\)", re.S)


def read_test(path: Path) -> dict[str, float]:
    matches = TEST_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No test result in {path}")
    return {key: float(value) for key, value in PAIR_RE.findall(matches[-1])}


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-root", type=Path, default=Path("log_runs/www27_matched_seed_grid"))
    parser.add_argument("--sequence-root", type=Path, default=Path("log_runs/www27_sequence_geometry_probe_gpu1"))
    parser.add_argument("--datasets", nargs="+", default=["Sports", "Toys"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--out", type=Path, default=Path("analysis_results/www27_sequence_geometry_confirmation/summary.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            base = read_test(args.matched_root / dataset / f"seed{seed}" / "SASRec.log")
            sequence = read_test(args.sequence_root / dataset / f"seed{seed}" / "sequence_tau10.log")
            joint = read_test(args.matched_root / dataset / f"seed{seed}" / "CANDSSASRec.log")
            for metric in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
                rows.append({"dataset": dataset, "seed": seed, "metric": metric,
                             "dot": base[metric], "sequence": sequence[metric], "joint": joint[metric],
                             "sequence_minus_dot": sequence[metric] - base[metric],
                             "joint_minus_sequence": joint[metric] - sequence[metric]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    lines = [
        "# Sports and Toys sequence-only confirmation", "",
        "All values pair the same three seeds and common training configuration. "
        "The sequence-only runs were added after a fixed-tau screen; tau=10 is not selected on these test values.", "",
        "| dataset | metric | dot mean ± sd | sequence mean ± sd | joint mean ± sd | sequence - dot | joint - sequence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in args.datasets:
        for metric in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
            group = [row for row in rows if row["dataset"] == dataset and row["metric"] == metric]
            values = {key: [float(row[key]) for row in group] for key in ("dot", "sequence", "joint", "sequence_minus_dot", "joint_minus_sequence")}
            dot_m, dot_s = mean_sd(values["dot"]); seq_m, seq_s = mean_sd(values["sequence"]); joint_m, joint_s = mean_sd(values["joint"])
            delta_seq, delta_seq_s = mean_sd(values["sequence_minus_dot"]); delta_joint, delta_joint_s = mean_sd(values["joint_minus_sequence"])
            lines.append(f"| {dataset} | {metric} | {dot_m:.4f} ± {dot_s:.4f} | {seq_m:.4f} ± {seq_s:.4f} | {joint_m:.4f} ± {joint_s:.4f} | {delta_seq:+.4f} ± {delta_seq_s:.4f} | {delta_joint:+.4f} ± {delta_joint_s:.4f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
