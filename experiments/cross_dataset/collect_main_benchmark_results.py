#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path


NAME_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<model>SASRec|CANDSSASRec)_h(?P<hidden>\d+)_len(?P<max_len>\d+)(?:_temp(?P<temp>.+))?$"
)
METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]


def parse_test_result(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"test result:\s*(\{[^}]+\})", text)
    if not matches:
        return None
    return ast.literal_eval(matches[-1])


def parse_log_name(log_path):
    match = NAME_RE.match(log_path.stem)
    if not match:
        return None
    info = match.groupdict()
    info["hidden"] = int(info["hidden"])
    info["max_len"] = int(info["max_len"])
    info["temp"] = "" if info["temp"] is None else info["temp"]
    return info


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows):
    headers = ["dataset", "hidden", "max_len", "model", "temp", *METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="log_runs/main_benchmark_grid")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        info = parse_log_name(log_path)
        if info is None:
            continue
        result = parse_test_result(log_path)
        if result is None:
            continue
        row = {**info}
        for metric in METRICS:
            row[metric] = float(result.get(metric, 0.0))
        rows.append(row)

    rows.sort(key=lambda r: (r["dataset"], r["hidden"], r["max_len"], r["model"] != "SASRec", str(r["temp"])))
    headers = ["dataset", "hidden", "max_len", "model", "temp", *METRICS]

    out_csv = Path(args.out_csv) if args.out_csv else log_dir / "summary.csv"
    out_md = Path(args.out_md) if args.out_md else log_dir / "summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    out_md.write_text(markdown_table(rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
