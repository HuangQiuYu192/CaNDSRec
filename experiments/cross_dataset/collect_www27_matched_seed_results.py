#!/usr/bin/env python3
"""Collect completed test metrics from the matched two-GPU training grid."""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

PATTERN = re.compile(r"test result: OrderedDict\((\[.*\])\)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for log in sorted(Path(args.log_root).glob("*/seed*/*.log")):
        match = PATTERN.search(log.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            continue
        rows.append({
            "dataset": log.parents[1].name,
            "seed": int(log.parent.name.removeprefix("seed")),
            "model": log.stem,
            "log": str(log),
            **dict(ast.literal_eval(match.group(1))),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} completed runs to {output}")


if __name__ == "__main__":
    main()
