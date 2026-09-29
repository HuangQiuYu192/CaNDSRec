#!/usr/bin/env python3
"""Summarize the wide Beauty score-geometry temperature screen.

The script selects one temperature per geometry by *validation* NDCG@10 and
only then exposes the test metrics saved by RecBole after checkpoint restore.
It intentionally does not select a common winner across geometries: that is a
separate, confirmatory comparison which must use fresh seeds.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR_RE = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK_RE = re.compile(
    r"(best valid result|test result):\s*OrderedDict\(\[(.*?)\]\)", re.S
)
NAME_RE = re.compile(r"^(sequence|item|both)_tau([0-9.]+)\.log$")
GEOMETRY_ORDER = {"sequence": 0, "item": 1, "both": 2}


def parse_metrics(path: Path) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for label, body in BLOCK_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
        parsed[label] = {name: float(value) for name, value in PAIR_RE.findall(body)}
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root", type=Path,
        default=Path("log_runs/beauty_outer_temperature_grid_gpu1/seed2026"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("analysis_results/beauty_outer_temperature_grid/seed2026_summary.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in args.log_root.glob("*.log"):
        name = NAME_RE.match(path.name)
        if not name:
            continue
        metrics = parse_metrics(path)
        if "best valid result" not in metrics or "test result" not in metrics:
            continue
        valid, test = metrics["best valid result"], metrics["test result"]
        rows.append({
            "geometry": name.group(1),
            "temperature": float(name.group(2)),
            "valid_recall@10": valid.get("recall@10"),
            "valid_ndcg@10": valid.get("ndcg@10"),
            "test_recall@10": test.get("recall@10"),
            "test_ndcg@10": test.get("ndcg@10"),
            "test_recall@20": test.get("recall@20"),
            "test_ndcg@20": test.get("ndcg@20"),
            "selected_by_validation": False,
            "log": str(path),
        })
    if not rows:
        raise SystemExit(f"No complete logs found under {args.log_root}")

    for geometry in {str(row["geometry"]) for row in rows}:
        candidates = [row for row in rows if row["geometry"] == geometry]
        max(candidates, key=lambda row: (float(row["valid_ndcg@10"]), -float(row["temperature"])))["selected_by_validation"] = True

    rows.sort(key=lambda row: (GEOMETRY_ORDER[str(row["geometry"])], float(row["temperature"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Beauty wide score-geometry temperature screen",
        "",
        "Each row reports the test metrics from the checkpoint selected by that run's validation NDCG@10. "
        "The `selected` flag chooses the temperature within each geometry using validation NDCG@10 only.",
        "",
        "| geometry | tau | valid Recall@10 | valid NDCG@10 | test Recall@10 | test NDCG@10 | test Recall@20 | test NDCG@20 | selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {geometry} | {temperature:g} | {valid_recall@10:.4f} | {valid_ndcg@10:.4f} | "
            "{test_recall@10:.4f} | {test_ndcg@10:.4f} | {test_recall@20:.4f} | {test_ndcg@20:.4f} | {selected} |".format(
                **row, selected="yes" if row["selected_by_validation"] else ""
            )
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
