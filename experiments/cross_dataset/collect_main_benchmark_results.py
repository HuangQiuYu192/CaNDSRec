#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path


NAME_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<model>SASRec|CANDSSASRec|WEARec|CANDSWEARec)_h(?P<hidden>\d+)_len(?P<max_len>\d+)(?:_temp(?P<temp>.+))?$"
)
METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"((?:recall|ndcg)@\d+)\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.I)


def parse_test_result(log_path):
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="ignore"))
    lower = text.lower()
    marker = "test result"
    pos = lower.rfind(marker)
    if pos < 0:
        return None
    block = text[pos : pos + 4000]

    dict_match = re.search(r"\{[^}]+\}", block, flags=re.S)
    if dict_match:
        return ast.literal_eval(dict_match.group(0))

    ordered_match = re.search(r"OrderedDict\((\[[\s\S]*?\])\)", block)
    if ordered_match:
        return dict(ast.literal_eval(ordered_match.group(1)))

    metrics = {name.lower(): float(value) for name, value in METRIC_RE.findall(block)}
    return metrics or None


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

    baseline_rank = {"SASRec": 0, "WEARec": 0, "CANDSSASRec": 1, "CANDSWEARec": 1}
    rows.sort(
        key=lambda r: (
            r["dataset"],
            r["hidden"],
            r["max_len"],
            baseline_rank.get(r["model"], 9),
            r["model"],
            str(r["temp"]),
        )
    )
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
