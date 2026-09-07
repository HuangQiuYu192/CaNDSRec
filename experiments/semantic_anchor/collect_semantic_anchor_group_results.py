#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_dir", default="analysis_results/beauty_tfidf_semantic_anchor_group_eval")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()

    group_dir = Path(args.group_dir)
    rows = []
    for path in sorted(group_dir.glob("*.csv")):
        if path.name in {"summary.csv", "semantic_anchor_group_table.csv"}:
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("group") in {"head", "mid", "tail"}:
                    rows.append(row)

    baselines = {
        (row["dataset"], row["group"]): row
        for row in rows
        if row.get("model") == "CANDSSASRec"
    }

    output = []
    for row in rows:
        base = baselines.get((row["dataset"], row["group"]))
        out = {
            "dataset": row.get("dataset"),
            "group": row.get("group"),
            "tag": row.get("tag"),
            "model": row.get("model"),
            "hidden": row.get("hidden"),
            "max_len": row.get("max_len"),
            "temperature": row.get("temperature"),
            "semantic_fusion_mode": row.get("semantic_fusion_mode"),
            "semantic_gate": row.get("semantic_gate"),
            "semantic_weight": parse_float(row.get("semantic_weight")),
            "recall@5": parse_float(row.get("recall@5")),
            "recall@10": parse_float(row.get("recall@10")),
            "recall@20": parse_float(row.get("recall@20")),
            "recall@50": parse_float(row.get("recall@50")),
            "recall@100": parse_float(row.get("recall@100")),
            "ndcg@5": parse_float(row.get("ndcg@5")),
            "ndcg@10": parse_float(row.get("ndcg@10")),
            "ndcg@20": parse_float(row.get("ndcg@20")),
            "ndcg@50": parse_float(row.get("ndcg@50")),
            "ndcg@100": parse_float(row.get("ndcg@100")),
            "median_rank": parse_float(row.get("median_rank")),
        }
        if base:
            base_r10 = parse_float(base.get("recall@10"))
            base_n10 = parse_float(base.get("ndcg@10"))
            if base_r10 and out["recall@10"] is not None:
                out["rel_recall@10_vs_cands"] = out["recall@10"] / base_r10 - 1.0
            if base_n10 and out["ndcg@10"] is not None:
                out["rel_ndcg@10_vs_cands"] = out["ndcg@10"] / base_n10 - 1.0
        output.append(out)

    group_order = {"head": 0, "mid": 1, "tail": 2}
    output.sort(
        key=lambda row: (
            row["dataset"],
            group_order.get(row["group"], 99),
            0 if row["model"] == "CANDSSASRec" else 1,
            -float(row.get("ndcg@10") or 0.0),
            str(row.get("tag", "")),
        )
    )

    headers = [
        "dataset",
        "group",
        "tag",
        "model",
        "hidden",
        "max_len",
        "temperature",
        "semantic_fusion_mode",
        "semantic_gate",
        "semantic_weight",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg@50",
        "ndcg@100",
        "median_rank",
        "rel_recall@10_vs_cands",
        "rel_ndcg@10_vs_cands",
    ]

    out_csv = Path(args.out_csv) if args.out_csv else group_dir / "semantic_anchor_group_table.csv"
    out_md = Path(args.out_md) if args.out_md else group_dir / "semantic_anchor_group_table.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)
    out_md.write_text(markdown_table(output, headers), encoding="utf-8")
    print(f"wrote {len(output)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
