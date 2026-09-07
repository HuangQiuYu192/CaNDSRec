#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path


METRICS = [
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
]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"((?:recall|ndcg)@\d+)\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.I)
NAME_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<model>CANDSSASRec|SemanticCANDSSASRec)"
    r"_h(?P<hidden>\d+)_len(?P<max_len>\d+)_temp(?P<temp>[^_]+)"
    r"(?:(?:_svd(?P<svd_dim>\d+))|(?:_text(?P<text_model>[^_]+)_dim(?P<text_dim>\d+)))?"
    r"(?:_mode(?P<mode>[^_]+)_gate(?P<gate>[^_]+)_w(?P<weight>[^_]+))?$"
)


def parse_test_result(log_path: Path):
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="ignore"))
    pos = text.lower().rfind("test result")
    if pos < 0:
        return None
    block = text[pos : pos + 4000]
    dict_match = re.search(r"\{[^}]+\}", block, flags=re.S)
    if dict_match:
        return {str(k).lower(): float(v) for k, v in ast.literal_eval(dict_match.group(0)).items()}
    ordered_match = re.search(r"OrderedDict\((\[[\s\S]*?\])\)", block)
    if ordered_match:
        return {str(k).lower(): float(v) for k, v in dict(ast.literal_eval(ordered_match.group(1))).items()}
    metrics = {name.lower(): float(value) for name, value in METRIC_RE.findall(block)}
    return metrics or None


def parse_name(path: Path):
    match = NAME_RE.match(path.stem)
    if not match:
        return None
    row = match.groupdict()
    row["hidden"] = int(row["hidden"])
    row["max_len"] = int(row["max_len"])
    row["svd_dim"] = "" if row["svd_dim"] is None else int(row["svd_dim"])
    row["text_model"] = row["text_model"] or ""
    row["text_dim"] = "" if row["text_dim"] is None else int(row["text_dim"])
    row["mode"] = row["mode"] or ""
    row["gate"] = row["gate"] or ""
    row["weight"] = "" if row["weight"] is None else float(row["weight"])
    if row["model"] == "CANDSSASRec":
        row["method"] = "CaNDS"
    elif row["text_model"]:
        row["method"] = "CaNDS-Text"
    else:
        row["method"] = "CaNDS-TFIDF"
    return row


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    parser.add_argument("--rank_metric", default="ndcg@10")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        info = parse_name(log_path)
        if info is None:
            continue
        result = parse_test_result(log_path)
        if result is None:
            continue
        row = {**info}
        for metric in METRICS:
            row[metric] = result.get(metric, 0.0)
        rows.append(row)

    baseline = next((row for row in rows if row["method"] == "CaNDS"), None)
    if baseline is not None:
        base_r10 = baseline["recall@10"]
        base_n10 = baseline["ndcg@10"]
        for row in rows:
            row["rel_recall@10_vs_cands"] = row["recall@10"] / base_r10 - 1.0 if base_r10 else ""
            row["rel_ndcg@10_vs_cands"] = row["ndcg@10"] / base_n10 - 1.0 if base_n10 else ""
    else:
        for row in rows:
            row["rel_recall@10_vs_cands"] = ""
            row["rel_ndcg@10_vs_cands"] = ""

    rows.sort(
        key=lambda row: (
            0 if row["method"] == "CaNDS" else 1,
            -float(row.get(args.rank_metric, 0.0)),
            str(row.get("mode", "")),
            str(row.get("gate", "")),
            str(row.get("weight", "")),
        )
    )

    headers = [
        "dataset",
        "method",
        "model",
        "hidden",
        "max_len",
        "temp",
        "svd_dim",
        "text_model",
        "text_dim",
        "mode",
        "gate",
        "weight",
        *METRICS,
        "rel_recall@10_vs_cands",
        "rel_ndcg@10_vs_cands",
    ]
    out_csv = Path(args.out_csv) if args.out_csv else log_dir / "semantic_anchor_summary.csv"
    out_md = Path(args.out_md) if args.out_md else log_dir / "semantic_anchor_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    out_md.write_text(markdown_table(rows, headers), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
