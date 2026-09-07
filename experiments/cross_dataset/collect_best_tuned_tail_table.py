#!/usr/bin/env python3
"""Collect compact head/mid/tail table from best-tuned group evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def method_from_row(row: dict) -> str:
    if row.get("method"):
        return row["method"]
    model = row.get("model", "")
    if model == "SASRec":
        return "SASRec"
    if model == "CANDSSASRec":
        return "CaNDS"
    if model == "AngularSmoothCANDSSASRec":
        return "AngularSmooth"
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_dir", default="analysis_results/best_tuned_group_eval")
    parser.add_argument("--out_csv", default="analysis_results/best_tuned_group_eval/best_tuned_group_table.csv")
    parser.add_argument("--out_md", default="analysis_results/best_tuned_group_eval/best_tuned_group_table.md")
    args = parser.parse_args()

    group_dir = Path(args.group_dir)
    rows = []
    for path in sorted(group_dir.glob("*.csv")):
        if path.name in {"summary.csv", "best_tuned_tail_table.csv", "best_tuned_group_table.csv"}:
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("group") in {"head", "mid", "tail"}:
                    row["method"] = method_from_row(row)
                    rows.append(row)

    by_dataset_group_method = {(row["dataset"], row["group"], row["method"]): row for row in rows}
    output = []
    group_order = {"head": 0, "mid": 1, "tail": 2}
    for dataset in sorted({row["dataset"] for row in rows}):
        for group in ["head", "mid", "tail"]:
            sas = by_dataset_group_method.get((dataset, group, "SASRec"))
            cands = by_dataset_group_method.get((dataset, group, "CaNDS"))
            angular = by_dataset_group_method.get((dataset, group, "AngularSmooth"))
            for row in [sas, cands, angular]:
                if row is None:
                    continue
                out = {
                    "dataset": dataset,
                    "group": group,
                    "method": row["method"],
                    "hidden": row.get("hidden"),
                    "max_len": row.get("max_len"),
                    "temperature": row.get("temperature"),
                    "recall@10": parse_float(row.get("recall@10")),
                    "ndcg@10": parse_float(row.get("ndcg@10")),
                    "recall@20": parse_float(row.get("recall@20")),
                    "ndcg@20": parse_float(row.get("ndcg@20")),
                    "recall@50": parse_float(row.get("recall@50")),
                    "ndcg@50": parse_float(row.get("ndcg@50")),
                    "median_rank": parse_float(row.get("median_rank")),
                }
                sas_ndcg = parse_float(sas.get("ndcg@10")) if sas else None
                cands_ndcg = parse_float(cands.get("ndcg@10")) if cands else None
                value_ndcg = out["ndcg@10"]
                if sas_ndcg and value_ndcg is not None:
                    out["rel_ndcg@10_vs_best_sasrec"] = value_ndcg / sas_ndcg - 1.0
                if row["method"] == "AngularSmooth" and cands_ndcg and value_ndcg is not None:
                    out["rel_ndcg@10_vs_best_cands"] = value_ndcg / cands_ndcg - 1.0
                output.append(out)

    method_order = {"SASRec": 0, "CaNDS": 1, "AngularSmooth": 2}
    output.sort(key=lambda row: (row["dataset"], group_order.get(row["group"], 99), method_order.get(row["method"], 99)))
    headers = [
        "dataset", "group", "method", "hidden", "max_len", "temperature",
        "recall@10", "ndcg@10", "recall@20", "ndcg@20",
        "recall@50", "ndcg@50", "median_rank",
        "rel_ndcg@10_vs_best_sasrec", "rel_ndcg@10_vs_best_cands",
    ]

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)
    out_md.write_text(markdown_table(output, headers), encoding="utf-8")
    print(f"wrote {len(output)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
