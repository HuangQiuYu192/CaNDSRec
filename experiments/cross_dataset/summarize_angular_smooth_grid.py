#!/usr/bin/env python3
"""Summarize AngularSmooth grid results by dataset."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    with input_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    output = []
    for dataset in sorted(grouped):
        candidates = grouped[dataset]
        for metric in ["ndcg@10", "recall@10", "ndcg@20", "recall@20"]:
            ranked = sorted(candidates, key=lambda row: as_float(row, metric), reverse=True)
            for rank, row in enumerate(ranked[: args.topk], start=1):
                output.append(
                    {
                        "dataset": dataset,
                        "rank_by": metric,
                        "rank": rank,
                        "hidden": row.get("hidden", ""),
                        "max_len": row.get("max_len", ""),
                        "temp": row.get("temp", ""),
                        "weight": row.get("weight", ""),
                        "k": row.get("k", ""),
                        "smooth_temp": row.get("smooth_temp", ""),
                        "quantile": row.get("quantile", ""),
                        "threshold": row.get("threshold", ""),
                        "recall@10": as_float(row, "recall@10"),
                        "ndcg@10": as_float(row, "ndcg@10"),
                        "recall@20": as_float(row, "recall@20"),
                        "ndcg@20": as_float(row, "ndcg@20"),
                    }
                )

    headers = [
        "dataset", "rank_by", "rank", "hidden", "max_len", "temp",
        "weight", "k", "smooth_temp", "quantile", "threshold",
        "recall@10", "ndcg@10", "recall@20", "ndcg@20",
    ]
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown_table(output, headers), encoding="utf-8")

    out_csv = Path(args.out_csv) if args.out_csv else out_md.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)

    print(f"wrote {len(output)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
