#!/usr/bin/env python3
"""Collect a validation-only temperature screen from completed log files."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK = re.compile(r"(best valid result|test result):\s*OrderedDict\(\[(.*?)\]\)", re.S)
NAME = re.compile(r"(?P<geometry>sequence|both)_tau(?P<tau>[0-9.]+)\.log$")


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics = {}
    for kind, payload in BLOCK.findall(text):
        metrics[kind] = {key: float(value) for key, value in PAIR.findall(payload)}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.log_root.glob("*/seed*/*.log")):
        match = NAME.search(path.name)
        if not match:
            continue
        parsed = parse_metrics(path.read_text(encoding="utf-8", errors="replace"))
        if "best valid result" not in parsed or "test result" not in parsed:
            continue
        valid, test = parsed["best valid result"], parsed["test result"]
        rows.append({
            "dataset": path.parts[-3], "seed": path.parts[-2].removeprefix("seed"),
            "geometry": match.group("geometry"), "temperature": float(match.group("tau")),
            "valid_ndcg@10": valid.get("ndcg@10"), "valid_recall@10": valid.get("recall@10"),
            "test_ndcg@10": test.get("ndcg@10"), "test_recall@10": test.get("recall@10"),
            "test_ndcg@20": test.get("ndcg@20"), "test_recall@20": test.get("recall@20"),
            "log": str(path),
        })
    if not rows:
        raise SystemExit("No completed temperature-screen logs found.")
    for row in rows:
        row["selected_by_validation"] = False
    for dataset in {row["dataset"] for row in rows}:
        for geometry in {row["geometry"] for row in rows if row["dataset"] == dataset}:
            candidates = [r for r in rows if r["dataset"] == dataset and r["geometry"] == geometry]
            max(candidates, key=lambda r: (r["valid_ndcg@10"], -r["temperature"]))["selected_by_validation"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Geometry temperature screen", "", "Temperature is selected solely by validation NDCG@10; test metrics are displayed only after selection.", "", "| dataset | geometry | tau | valid NDCG@10 | test NDCG@10 | test Recall@10 | selected |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in sorted(rows, key=lambda r: (r["dataset"], r["geometry"], r["temperature"])):
        lines.append(f"| {row['dataset']} | {row['geometry']} | {row['temperature']:g} | {row['valid_ndcg@10']:.4f} | {row['test_ndcg@10']:.4f} | {row['test_recall@10']:.4f} | {'yes' if row['selected_by_validation'] else ''} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
