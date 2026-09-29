#!/usr/bin/env python3
"""Paired summary of independent-seed sequence-only tau=4 vs tau=5 runs."""
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


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--tau5_root", type=Path, default=Path("log_runs/beauty_radial_power_grid_gpu12"))
    parser.add_argument("--tau4_root", type=Path, default=Path("log_runs/beauty_sequence_tau4_confirmation_gpu1"))
    parser.add_argument("--out", type=Path, default=Path("analysis_results/beauty_sequence_temperature_confirmation/tau4_vs_tau5.csv"))
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    for seed in args.seeds:
        tau5 = read_test(args.tau5_root / f"seed{seed}" / "sequence_tau5.log")
        tau4 = read_test(args.tau4_root / f"seed{seed}" / "sequence_tau4.log")
        rows.append({
            "seed": seed,
            "tau5_recall@10": tau5["recall@10"], "tau4_recall@10": tau4["recall@10"],
            "delta_recall@10": tau4["recall@10"] - tau5["recall@10"],
            "tau5_ndcg@10": tau5["ndcg@10"], "tau4_ndcg@10": tau4["ndcg@10"],
            "delta_ndcg@10": tau4["ndcg@10"] - tau5["ndcg@10"],
            "tau5_recall@20": tau5["recall@20"], "tau4_recall@20": tau4["recall@20"],
            "delta_recall@20": tau4["recall@20"] - tau5["recall@20"],
            "tau5_ndcg@20": tau5["ndcg@20"], "tau4_ndcg@20": tau4["ndcg@20"],
            "delta_ndcg@20": tau4["ndcg@20"] - tau5["ndcg@20"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    metrics = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")
    lines = [
        "# Beauty sequence-only temperature confirmation",
        "",
        "Tau=4 was chosen in a separate tuning seed. This table is a paired independent-seed confirmation against preserved, configuration-matched tau=5 runs. It is descriptive (n=3), not a significance claim.",
        "",
        "| metric | tau=5 mean ± sd | tau=4 mean ± sd | paired delta (tau=4 - tau=5) mean ± sd |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        base_m, base_s = mean_std([float(row[f"tau5_{metric}"]) for row in rows])
        cand_m, cand_s = mean_std([float(row[f"tau4_{metric}"]) for row in rows])
        delta_m, delta_s = mean_std([float(row[f"delta_{metric}"]) for row in rows])
        lines.append(f"| {metric} | {base_m:.4f} ± {base_s:.4f} | {cand_m:.4f} ± {cand_s:.4f} | {delta_m:+.4f} ± {delta_s:.4f} |")
    lines.extend(["", "## Per-seed results", "", "| seed | tau=5 NDCG@10 | tau=4 NDCG@10 | delta |", "| ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['seed']} | {row['tau5_ndcg@10']:.4f} | {row['tau4_ndcg@10']:.4f} | {row['delta_ndcg@10']:+.4f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
