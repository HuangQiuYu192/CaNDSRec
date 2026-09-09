#!/usr/bin/env python3
"""Summarize the LastFM tau=5 vs tau=10 replication without test-set tuning."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PAIR = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK = re.compile(r"(best valid result|test result):\s*OrderedDict\(\[(.*?)\]\)", re.S)
NAME = re.compile(r"(?P<geometry>sequence|both)_tau(?P<tau>[0-9.]+)\.log$")


def parse(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for kind, payload in BLOCK.findall(path.read_text(encoding="utf-8", errors="replace")):
        result[kind] = {key: float(value) for key, value in PAIR.findall(payload)}
    return result


def fmt(values: list[float]) -> str:
    return f"{mean(values):.4f} ± {stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_roots", nargs="+", required=True, type=Path,
                    help="Include the original seed-2025 screen and the replication log root.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for root in args.log_roots:
        for path in root.glob("LastFM-S3Rec/seed*/*.log"):
            match = NAME.search(path.name)
            metrics = parse(path)
            if not match or "best valid result" not in metrics or "test result" not in metrics:
                continue
            valid, test = metrics["best valid result"], metrics["test result"]
            rows.append({"seed": int(path.parent.name.removeprefix("seed")), "geometry": match["geometry"],
                         "temperature": float(match["tau"]), "valid_ndcg@10": valid["ndcg@10"],
                         "test_ndcg@10": test["ndcg@10"], "test_recall@10": test["recall@10"], "log": str(path)})
    if not rows:
        raise SystemExit("No complete logs found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: (r["geometry"], r["temperature"], r["seed"])))
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        if r["temperature"] in (5, 10):
            groups[r["geometry"], r["temperature"]].append(r)
    lines = ["# LastFM temperature replication", "", "This is a paired, three-seed comparison of tau=5 and tau=10. It evaluates the stability of the prior screening discrepancy; it is not a test-set parameter-selection protocol.", "", "| geometry | tau | seeds | validation NDCG@10 | test NDCG@10 | test Recall@10 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for (geometry, tau), rs in sorted(groups.items()):
        lines.append(f"| {geometry} | {tau:g} | {','.join(str(r['seed']) for r in sorted(rs, key=lambda x: x['seed']))} | {fmt([r['valid_ndcg@10'] for r in rs])} | {fmt([r['test_ndcg@10'] for r in rs])} | {fmt([r['test_recall@10'] for r in rs])} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
