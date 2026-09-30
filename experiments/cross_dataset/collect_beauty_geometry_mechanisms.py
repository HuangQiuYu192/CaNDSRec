#!/usr/bin/env python3
"""Aggregate per-checkpoint Beauty geometry diagnostics across seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TAG_ORDER = {"dot": 0, "sequence": 1, "both": 2}
DIAGNOSTIC_FIELDS = [
    "sequence_norm_mean", "sequence_norm_cv", "item_norm_cv", "score_std_mean",
    "predictive_entropy_mean", "top1_top2_margin_mean",
    "counterfactual_seq_rank_equal_rate", "counterfactual_seq_topk_equal_rate",
    "counterfactual_item_topk_jaccard",
]
GROUP_FIELDS = ["recall@10", "ndcg@10", "recall@20", "ndcg@20", "median_rank"]


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def fmt(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.4f} ± {sd:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--out_prefix", required=True, type=Path)
    args = parser.parse_args()

    diagnostics = []
    groups = []
    for path in args.input_root.glob("*/metrics.json"):
        diagnostics.append(json.loads(path.read_text(encoding="utf-8")))
        with (path.parent / "groups.csv").open(newline="", encoding="utf-8") as handle:
            groups.extend(csv.DictReader(handle))
    if not diagnostics:
        raise SystemExit(f"No diagnostics under {args.input_root}")

    diagnostics.sort(key=lambda row: (TAG_ORDER.get(row["tag"], 99), row["seed"]))
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_prefix.with_name(args.out_prefix.name + "_per_seed.csv")
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader(); writer.writerows(diagnostics)

    by_tag: dict[str, list[dict]] = defaultdict(list)
    for row in diagnostics:
        by_tag[row["tag"]].append(row)
    summary_rows = []
    for tag, rows in by_tag.items():
        summary = {"tag": tag, "seeds": ",".join(str(row["seed"]) for row in rows)}
        for field in DIAGNOSTIC_FIELDS:
            mean, sd = mean_sd([float(row[field]) for row in rows])
            summary[f"{field}_mean"] = mean
            summary[f"{field}_sd"] = sd
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: TAG_ORDER.get(row["tag"], 99))
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_diagnostics.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in groups:
        grouped[(row["tag"], row["group"])].append(row)
    group_rows = []
    for (tag, group), rows in grouped.items():
        output = {"tag": tag, "group": group, "seeds": ",".join(row["seed"] for row in rows), "n_per_seed": rows[0]["n"]}
        for field in GROUP_FIELDS:
            mean, sd = mean_sd([float(row[field]) for row in rows])
            output[f"{field}_mean"] = mean
            output[f"{field}_sd"] = sd
        group_rows.append(output)
    group_rows.sort(key=lambda row: (TAG_ORDER.get(row["tag"], 99), ["all", "head", "mid", "tail"].index(row["group"])))
    group_csv = args.out_prefix.with_name(args.out_prefix.name + "_groups.csv")
    with group_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)

    lines = [
        "# Beauty score-geometry mechanism analyses",
        "",
        "All values are means ± sample standard deviations over seeds 2023, 2024 and 2025. "
        "Counterfactual ranking statistics use a frozen representation; they diagnose score geometry rather than retraining effects.",
        "",
        "## Scale and frozen-score counterfactuals",
        "",
        "| tag | seq norm CV | score std | entropy | top1-top2 margin | seq rank equality | seq Top-10 equality | item Top-10 Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, rows in sorted(by_tag.items(), key=lambda pair: TAG_ORDER.get(pair[0], 99)):
        lines.append(
            f"| {tag} | {fmt([float(r['sequence_norm_cv']) for r in rows])} | "
            f"{fmt([float(r['score_std_mean']) for r in rows])} | {fmt([float(r['predictive_entropy_mean']) for r in rows])} | "
            f"{fmt([float(r['top1_top2_margin_mean']) for r in rows])} | "
            f"{fmt([float(r['counterfactual_seq_rank_equal_rate']) for r in rows])} | "
            f"{fmt([float(r['counterfactual_seq_topk_equal_rate']) for r in rows])} | "
            f"{fmt([float(r['counterfactual_item_topk_jaccard']) for r in rows])} |"
        )
    lines.extend([
        "",
        "`seq rank equality` and `seq Top-10 equality` should be 1 up to numerical ties: dividing a fixed user's sequence vector by its positive norm rescales every candidate score equally. "
        "The item Jaccard is not expected to be 1, because item normalization changes candidate-specific scales.",
        "",
        "## Global popularity-tertile evaluation",
        "",
        "Items are partitioned into head/mid/tail by training-interaction count before mapping test targets to the partition; hence group sizes are intentionally unequal.",
        "",
        "| tag | group | n / seed | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | median rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in group_rows:
        lines.append(
            f"| {row['tag']} | {row['group']} | {row['n_per_seed']} | "
            f"{row['recall@10_mean']:.4f} ± {row['recall@10_sd']:.4f} | "
            f"{row['ndcg@10_mean']:.4f} ± {row['ndcg@10_sd']:.4f} | "
            f"{row['recall@20_mean']:.4f} ± {row['recall@20_sd']:.4f} | "
            f"{row['ndcg@20_mean']:.4f} ± {row['ndcg@20_sd']:.4f} | "
            f"{row['median_rank_mean']:.1f} ± {row['median_rank_sd']:.1f} |"
        )
    args.out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {raw_csv}, {summary_csv}, {group_csv}, and {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
