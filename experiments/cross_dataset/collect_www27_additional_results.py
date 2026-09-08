#!/usr/bin/env python3
"""Collect reproducible summaries for the WWW strong-baseline and tau grids.

The parser reads only final ``test result`` lines from completed logs. It does
not select checkpoints or configurations using test metrics.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20")
TEST_RE = re.compile(r"test result:\s*OrderedDict\((\[.*\])\)")


def parse_test_result(path: Path) -> dict[str, float]:
    matches = TEST_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"missing final test result: {path}")
    row = dict(ast.literal_eval(matches[-1]))
    return {metric: float(row[metric]) for metric in METRICS}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, deviation: float) -> str:
    return f"{value:.4f} $\\pm$ {deviation:.4f}"


def collect_baselines(root: Path, out: Path) -> None:
    raw: list[dict[str, object]] = []
    for path in sorted((root / "log_runs" / "www27_strong_baselines").glob("*/seed*/*.log")):
        model = path.stem
        if model not in {"GRU4Rec", "BERT4Rec"}:
            continue
        dataset, seed_dir = path.parts[-3], path.parts[-2]
        if not seed_dir.startswith("seed"):
            continue
        row: dict[str, object] = {"dataset": dataset, "seed": int(seed_dir[4:]), "model": model, "log": str(path)}
        row.update(parse_test_result(path))
        raw.append(row)
    if len(raw) != 18:
        raise RuntimeError(f"expected 18 baseline endpoint logs, found {len(raw)}")
    fields = ["dataset", "seed", "model", *METRICS, "log"]
    write_csv(out / "strong_baseline_per_seed.csv", raw, fields)

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        groups[(str(row["dataset"]), str(row["model"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dataset, model), rows in sorted(groups.items()):
        item: dict[str, object] = {"dataset": dataset, "model": model, "n_seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    summary_fields = ["dataset", "model", "n_seeds", *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")]]
    write_csv(out / "strong_baseline_summary.csv", summary, summary_fields)

    lines = ["# WWW strong-baseline results", "", "Mean $\\pm$ standard deviation over three matched seeds.", "", "| Dataset | Model | R@10 | N@10 | R@20 | N@20 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {model} | {r10} | {n10} | {r20} | {n20} |".format(
            dataset=row["dataset"], model=row["model"],
            r10=fmt(float(row["recall@10_mean"]), float(row["recall@10_std"])),
            n10=fmt(float(row["ndcg@10_mean"]), float(row["ndcg@10_std"])),
            r20=fmt(float(row["recall@20_mean"]), float(row["recall@20_std"])),
            n20=fmt(float(row["ndcg@20_mean"]), float(row["ndcg@20_std"])),
        ))
    (out / "strong_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_temperatures(root: Path, out: Path) -> None:
    raw: list[dict[str, object]] = []
    for path in sorted((root / "log_runs" / "www27_temperature_sensitivity").glob("*/seed*/temp*.log")):
        dataset, seed_dir, filename = path.parts[-3], path.parts[-2], path.name
        temperature = float(filename[4:-4])
        row: dict[str, object] = {"dataset": dataset, "seed": int(seed_dir[4:]), "temperature": temperature, "log": str(path)}
        row.update(parse_test_result(path))
        raw.append(row)
    if len(raw) != 9:
        raise RuntimeError(f"expected 9 temperature endpoint logs, found {len(raw)}")
    fields = ["dataset", "seed", "temperature", *METRICS, "log"]
    write_csv(out / "temperature_per_seed.csv", raw, fields)
    lines = ["# WWW temperature sensitivity", "", "Single fixed seed (2025); this table is a robustness diagnostic, not a significance claim or test-set tuning rule.", "", "| Dataset | Temperature | R@10 | N@10 | R@20 | N@20 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in raw:
        lines.append("| {dataset} | {temperature:g} | {r10:.4f} | {n10:.4f} | {r20:.4f} | {n20:.4f} |".format(
            dataset=row["dataset"], temperature=float(row["temperature"]), r10=float(row["recall@10"]), n10=float(row["ndcg@10"]), r20=float(row["recall@20"]), n20=float(row["ndcg@20"])))
    (out / "temperature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("analysis_results/www27_additional"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    collect_baselines(root, out)
    collect_temperatures(root, out)
    print(f"wrote summaries to {out}")


if __name__ == "__main__":
    main()
