#!/usr/bin/env python3
"""Collect the pre-registered forward-equal/backward-different control."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

PAIR = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK = re.compile(r"test result:\s*OrderedDict\(\[(.*?)\]\)", re.S)
PATH = re.compile(r"([^/\\]+)[/\\]seed(\d+)[/\\]([a-z_]+)\.log$")


def parse(path: Path) -> dict[str, float] | None:
    blocks = BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    return {name: float(value) for name, value in PAIR.findall(blocks[-1])} if blocks else None


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.log_root.glob("*/seed*/*.log")):
        match, metrics = PATH.search(str(path)), parse(path)
        if not match or metrics is None:
            continue
        dataset, seed, variant = match.groups()
        rows.append({"dataset": dataset, "seed": int(seed), "variant": variant, **metrics, "log": str(path)})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)
    lines = ["# Forward-equal / backward-different gradient control", "",
             "`joint` and `stopgrad_joint` have the same forward formula at fixed parameters. The latter detaches radial denominators only in backpropagation.", "",
             "| dataset | variant | seeds | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (dataset, variant), group in sorted(grouped.items()):
        values = {key: mean_sd([float(row[key]) for row in group]) for key in ("recall@10", "ndcg@10", "recall@20", "ndcg@20")}
        lines.append(f"| {dataset} | {variant} | {len(group)} | {values['recall@10'][0]:.4f} ± {values['recall@10'][1]:.4f} | {values['ndcg@10'][0]:.4f} ± {values['ndcg@10'][1]:.4f} | {values['recall@20'][0]:.4f} ± {values['recall@20'][1]:.4f} | {values['ndcg@20'][0]:.4f} ± {values['ndcg@20'][1]:.4f} |")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} completed rows")


if __name__ == "__main__":
    main()
