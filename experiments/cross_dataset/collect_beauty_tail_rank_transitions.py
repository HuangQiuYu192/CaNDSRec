#!/usr/bin/env python3
"""Aggregate paired Beauty tail rank-transition diagnostics over seeds."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


TAG_ORDER = {"sequence": 0, "both": 1}
GROUP_ORDER = {"all": 0, "head": 1, "mid": 2, "tail": 3}
FIELDS = ["delta_recall@10", "delta_recall@50", "mean_rank_delta", "median_rank_delta", "improved_rate", "worsened_rate", "enter_top10_rate", "leave_top10_rate", "deep_to_11_50_rate", "top10_to_11_50_rate"]


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def fmt(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:+.4f} ± {sd:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.input_root.glob("seed*/summary.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No transition summary files found")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["tag"], row["group"])].append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_name(args.out.stem + "_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for (tag, group), values in groups.items():
        row = {"tag": tag, "group": group, "n_per_seed": values[0]["n"], "seeds": ",".join(value["seed"] for value in values)}
        for field in FIELDS:
            mean, sd = mean_sd([float(value[field]) for value in values])
            row[f"{field}_mean"] = mean; row[f"{field}_sd"] = sd
        aggregate.append(row)
    aggregate.sort(key=lambda row: (TAG_ORDER[row["tag"]], GROUP_ORDER[row["group"]]))
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)

    lines = [
        "# Beauty paired rank-transition analysis",
        "",
        "Each row compares a geometry checkpoint against the same-seed dot checkpoint on identical test targets. Negative rank deltas mean better ranks. Values are means ± sample standard deviations over three seeds.",
        "",
        "| target | group | Delta Recall@10 | Delta Recall@50 | mean rank delta | improved / worsened | enter Top-10 / leave Top-10 | >50 to 11-50 | Top-10 to 11-50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        def ratio(field: str) -> str:
            return f"{row[field + '_mean']:.4f} ± {row[field + '_sd']:.4f}"
        lines.append(
            f"| {row['tag']} | {row['group']} | {fmt([float(v['delta_recall@10']) for v in groups[(row['tag'], row['group'])]])} | "
            f"{fmt([float(v['delta_recall@50']) for v in groups[(row['tag'], row['group'])]])} | "
            f"{fmt([float(v['mean_rank_delta']) for v in groups[(row['tag'], row['group'])]])} | "
            f"{ratio('improved_rate')} / {ratio('worsened_rate')} | {ratio('enter_top10_rate')} / {ratio('leave_top10_rate')} | "
            f"{ratio('deep_to_11_50_rate')} | {ratio('top10_to_11_50_rate')} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
