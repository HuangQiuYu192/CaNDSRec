#!/usr/bin/env python3
"""Aggregate matched-seed global-popularity group metrics for the WWW draft."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(root: Path):
    rows = []
    for path in sorted(root.glob("group_metrics/*/seed*/*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def f(value):
    return float(value) if value not in (None, "") else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    source = read_rows(Path(args.input_root))
    by_key = defaultdict(list)
    for row in source:
        key = (row["dataset"], row["model"], row["group"])
        by_key[key].append(row)
    metrics = [f"{name}@{k}" for k in (5, 10, 20, 50, 100) for name in ("recall", "ndcg")]
    summary = []
    for (dataset, model, group), rows in sorted(by_key.items()):
        out = {"dataset": dataset, "model": model, "group": group, "seeds": len(rows), "n_per_seed": rows[0]["n"]}
        for metric in metrics:
            values = np.asarray([f(row.get(metric)) for row in rows])
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)

    # Pair CaNDS and SASRec by dataset/group/seed.  The variance here is across
    # independently trained matched seeds; it is explicitly not a per-user CI.
    per_seed = {}
    for row in source:
        seed = int(row["tag"].split("_seed", 1)[1].split("_", 1)[0])
        per_seed[(row["dataset"], row["group"], seed, row["model"])] = row
    deltas = []
    pairs = sorted({(d, g, s) for d, g, s, _ in per_seed})
    for dataset, group, seed in pairs:
        sas = per_seed.get((dataset, group, seed, "SASRec"))
        cands = per_seed.get((dataset, group, seed, "CANDSSASRec"))
        if not sas or not cands:
            continue
        row = {"dataset": dataset, "group": group, "seed": seed}
        for metric in metrics:
            row[f"delta_{metric}"] = f(cands.get(metric)) - f(sas.get(metric))
        deltas.append(row)
    delta_summary = []
    for dataset, group in sorted({(r["dataset"], r["group"]) for r in deltas}):
        matched = [r for r in deltas if r["dataset"] == dataset and r["group"] == group]
        row = {"dataset": dataset, "group": group, "paired_seeds": len(matched)}
        for metric in metrics:
            values = np.asarray([r[f"delta_{metric}"] for r in matched])
            row[f"delta_{metric}_mean"] = float(values.mean())
            row[f"delta_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"positive_seed_fraction_{metric}"] = float((values > 0).mean())
        delta_summary.append(row)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("_summary.csv", summary), ("_paired_deltas.csv", deltas), ("_paired_delta_summary.csv", delta_summary)):
        fields = sorted({key for row in rows for key in row})
        with Path(f"{prefix}{suffix}").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    lines = ["# Matched-seed paper summary", "", "## Overall metrics", "", "| dataset | model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | seeds |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        if row["group"] != "all":
            continue
        lines.append(f"| {row['dataset']} | {row['model']} | {row['recall@10_mean']:.4f} ± {row['recall@10_std']:.4f} | {row['ndcg@10_mean']:.4f} ± {row['ndcg@10_std']:.4f} | {row['recall@50_mean']:.4f} ± {row['recall@50_std']:.4f} | {row['ndcg@50_mean']:.4f} ± {row['ndcg@50_std']:.4f} | {row['seeds']} |")
    lines += ["", "## Paired CaNDS minus SASRec deltas", "", "| dataset | group | Δ Recall@10 | Δ NDCG@10 | positive seeds (NDCG@10) |", "| --- | --- | ---: | ---: | ---: |"]
    for row in delta_summary:
        lines.append(f"| {row['dataset']} | {row['group']} | {row['delta_recall@10_mean']:+.4f} ± {row['delta_recall@10_std']:.4f} | {row['delta_ndcg@10_mean']:+.4f} ± {row['delta_ndcg@10_std']:.4f} | {row['positive_seed_fraction_ndcg@10']:.0%} |")
    Path(f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote summaries with {len(summary)} aggregate rows and {len(deltas)} paired seed rows")


if __name__ == "__main__":
    main()
