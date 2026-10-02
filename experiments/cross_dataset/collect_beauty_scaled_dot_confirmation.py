#!/usr/bin/env python3
"""Aggregate fixed-temperature, held-out-seed scaled-dot confirmation runs."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


PAIR = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK = re.compile(r"(best valid result|test result):\s*OrderedDict\(\[(.*?)\]\)", re.S)
SEED = re.compile(r"seed(\d+)\.log$")


def parse(path: Path) -> dict[str, dict[str, float]]:
    return {
        kind: {key: float(value) for key, value in PAIR.findall(body)}
        for kind, body in BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    }


def fmt(values: list[float]) -> str:
    return f"{mean(values):.4f} +/- {stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.log_root.glob("seed*.log")):
        seed = SEED.search(path.name)
        result = parse(path)
        if not seed or "best valid result" not in result or "test result" not in result:
            continue
        valid, test = result["best valid result"], result["test result"]
        rows.append({
            "seed": int(seed.group(1)), "temperature": args.temperature,
            "valid_recall@10": valid["recall@10"], "valid_ndcg@10": valid["ndcg@10"],
            "test_recall@10": test["recall@10"], "test_ndcg@10": test["ndcg@10"],
            "log": str(path),
        })
    if not rows:
        raise SystemExit("No complete scaled-dot confirmation logs found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["test_recall@10", "test_ndcg@10"]
    lines = [
        "# Beauty Scaled-Dot held-out-seed confirmation", "",
        f"Temperature is fixed to $\\tau={args.temperature:g}$ from the seed-2026 validation screen.",
        "No test metric selects this temperature.", "",
        "| metric | mean +/- sample std. |", "| --- | ---: |",
    ]
    for metric in metrics:
        lines.append(f"| {metric} | {fmt([float(row[metric]) for row in rows])} |")
    lines.extend(["", "## Per-seed results", "", "| seed | valid NDCG@10 | test Recall@10 | test NDCG@10 |", "| ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['valid_ndcg@10']:.4f} | {row['test_recall@10']:.4f} | {row['test_ndcg@10']:.4f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
