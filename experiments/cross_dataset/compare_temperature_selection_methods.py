#!/usr/bin/env python3
"""Compare fixed, oracle, and automatically selected CAND temperatures.

This script consumes two existing summaries:

1. main benchmark results, produced by collect_main_benchmark_results.py
2. temperature calibration results, produced by collect_temperature_calibration_results.py

It does not train models. For each dataset/hidden/max_len setting, it reports:

- SASRec dot-product baseline
- CAND with a fixed temperature, usually 10
- CAND with oracle grid-best temperature
- CAND with each automatic temperature estimate rounded to the nearest tested
  grid value

The output makes it easy to answer whether a calibration rule can replace
manual temperature tuning.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
DEFAULT_METHODS = [
    "std",
    "margin",
    "pos_neg_gap",
    "adaptau_all",
    "adaptau_all_x0.5",
    "adaptau_neg",
    "adaptau_neg_x0.5",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float:
    if value in {None, ""}:
        return math.nan
    return float(value)


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def key_of(row: dict[str, str]) -> tuple[str, int, int]:
    return (row["dataset"], int(row["hidden"]), int(row["max_len"]))


def load_benchmark(path: Path) -> dict[tuple[str, int, int], dict[str, list[dict[str, str]]]]:
    grouped: dict[tuple[str, int, int], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in read_csv(path):
        grouped[key_of(row)][row["model"]].append(row)
    return grouped


def load_calibration_summary(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    return {key_of(row): row for row in read_csv(path)}


def metric_at_temp(cand_rows: list[dict[str, str]], temp: float, metric: str) -> float:
    if not math.isfinite(temp):
        return math.nan
    for row in cand_rows:
        if abs(as_float(row.get("temp")) - temp) < 1e-8:
            return as_float(row.get(metric))
    return math.nan


def cand_row_at_temp(cand_rows: list[dict[str, str]], temp: float) -> dict[str, float]:
    return {metric: metric_at_temp(cand_rows, temp, metric) for metric in METRICS}


def best_cand_row(cand_rows: list[dict[str, str]], select_metric: str) -> tuple[float, dict[str, float]]:
    best = max(cand_rows, key=lambda row: as_float(row.get(select_metric)))
    return as_float(best.get("temp")), {metric: as_float(best.get(metric)) for metric in METRICS}


def add_method_row(
    rows: list[dict],
    dataset: str,
    hidden: int,
    max_len: int,
    method: str,
    temp: float,
    metrics: dict[str, float],
    sasrec_metrics: dict[str, float],
    best_metrics: dict[str, float],
) -> None:
    recall = metrics.get("recall@10", math.nan)
    ndcg = metrics.get("ndcg@10", math.nan)
    sasrec_recall = sasrec_metrics.get("recall@10", math.nan)
    sasrec_ndcg = sasrec_metrics.get("ndcg@10", math.nan)
    best_ndcg = best_metrics.get("ndcg@10", math.nan)
    rows.append(
        {
            "dataset": dataset,
            "hidden": hidden,
            "max_len": max_len,
            "method": method,
            "selected_temp": temp,
            **metrics,
            "delta_recall@10_vs_sasrec": recall - sasrec_recall,
            "delta_ndcg@10_vs_sasrec": ndcg - sasrec_ndcg,
            "rel_recall@10_vs_sasrec": recall / sasrec_recall - 1 if sasrec_recall > 0 else math.nan,
            "rel_ndcg@10_vs_sasrec": ndcg / sasrec_ndcg - 1 if sasrec_ndcg > 0 else math.nan,
            "ndcg@10_retention_vs_grid_best": ndcg / best_ndcg if best_ndcg > 0 else math.nan,
        }
    )


def aggregate(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] == "SASRec":
            continue
        by_method[row["method"]].append(row)

    output = []
    for method, method_rows in sorted(by_method.items()):
        valid = [r for r in method_rows if math.isfinite(r["ndcg@10"])]
        if not valid:
            continue
        output.append(
            {
                "method": method,
                "num_settings": len(valid),
                "avg_recall@10": sum(r["recall@10"] for r in valid) / len(valid),
                "avg_ndcg@10": sum(r["ndcg@10"] for r in valid) / len(valid),
                "avg_delta_recall@10_vs_sasrec": sum(r["delta_recall@10_vs_sasrec"] for r in valid)
                / len(valid),
                "avg_delta_ndcg@10_vs_sasrec": sum(r["delta_ndcg@10_vs_sasrec"] for r in valid)
                / len(valid),
                "avg_rel_recall@10_vs_sasrec": sum(r["rel_recall@10_vs_sasrec"] for r in valid)
                / len(valid),
                "avg_rel_ndcg@10_vs_sasrec": sum(r["rel_ndcg@10_vs_sasrec"] for r in valid)
                / len(valid),
                "avg_ndcg@10_retention_vs_grid_best": sum(
                    r["ndcg@10_retention_vs_grid_best"] for r in valid
                )
                / len(valid),
                "wins_vs_sasrec_ndcg@10": sum(
                    1 for r in valid if r["delta_ndcg@10_vs_sasrec"] > 0
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["avg_ndcg@10_retention_vs_grid_best"], row["avg_delta_ndcg@10_vs_sasrec"]),
        reverse=True,
    )


def aggregate_by_hidden(rows: list[dict]) -> list[dict]:
    output = []
    hidden_values = sorted({int(row["hidden"]) for row in rows})
    for hidden in hidden_values:
        hidden_rows = [row for row in rows if int(row["hidden"]) == hidden]
        for row in aggregate(hidden_rows):
            row = dict(row)
            row["hidden"] = hidden
            output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row["hidden"],
            -row["avg_ndcg@10_retention_vs_grid_best"],
            -row["avg_delta_ndcg@10_vs_sasrec"],
        ),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_csv", default="log_runs/main_benchmark_grid/summary.csv")
    parser.add_argument(
        "--calibration_summary",
        default="analysis_results/temperature_calibration_adaptau/summary.csv",
    )
    parser.add_argument("--fixed_temp", default=10.0, type=float)
    parser.add_argument("--select_metric", default="ndcg@10")
    parser.add_argument("--methods", default=" ".join(DEFAULT_METHODS))
    parser.add_argument("--out_dir", default="analysis_results/temperature_selection")
    args = parser.parse_args()

    benchmark = load_benchmark(Path(args.benchmark_csv))
    calibration = load_calibration_summary(Path(args.calibration_summary))
    methods = [m for m in args.methods.replace(",", " ").split() if m]

    output_rows = []
    missing = []
    for key, model_rows in sorted(benchmark.items()):
        dataset, hidden, max_len = key
        sasrec_rows = model_rows.get("SASRec", [])
        cand_rows = model_rows.get("CANDSSASRec", [])
        calib = calibration.get(key)
        if not sasrec_rows or not cand_rows or calib is None:
            missing.append(key)
            continue

        sasrec_metrics = {metric: as_float(sasrec_rows[0].get(metric)) for metric in METRICS}
        best_temp, best_metrics = best_cand_row(cand_rows, args.select_metric)
        fixed_metrics = cand_row_at_temp(cand_rows, args.fixed_temp)

        add_method_row(
            output_rows,
            dataset,
            hidden,
            max_len,
            "SASRec",
            math.nan,
            sasrec_metrics,
            sasrec_metrics,
            best_metrics,
        )
        add_method_row(
            output_rows,
            dataset,
            hidden,
            max_len,
            f"fixed_temp_{args.fixed_temp:g}",
            args.fixed_temp,
            fixed_metrics,
            sasrec_metrics,
            best_metrics,
        )
        add_method_row(
            output_rows,
            dataset,
            hidden,
            max_len,
            "grid_best",
            best_temp,
            best_metrics,
            sasrec_metrics,
            best_metrics,
        )

        for method in methods:
            selected_temp = as_float(calib.get(f"auto_grid_{method}"))
            selected_metrics = cand_row_at_temp(cand_rows, selected_temp)
            add_method_row(
                output_rows,
                dataset,
                hidden,
                max_len,
                method,
                selected_temp,
                selected_metrics,
                sasrec_metrics,
                best_metrics,
            )

    out_dir = Path(args.out_dir)
    per_setting_csv = out_dir / "per_setting.csv"
    per_setting_md = out_dir / "per_setting.md"
    aggregate_csv = out_dir / "aggregate.csv"
    aggregate_md = out_dir / "aggregate.md"
    aggregate_by_hidden_csv = out_dir / "aggregate_by_hidden.csv"
    aggregate_by_hidden_md = out_dir / "aggregate_by_hidden.md"
    missing_path = out_dir / "missing_settings.txt"

    write_csv(output_rows, per_setting_csv)
    write_markdown(
        output_rows,
        per_setting_md,
        [
            "dataset",
            "hidden",
            "max_len",
            "method",
            "selected_temp",
            "recall@10",
            "ndcg@10",
            "rel_recall@10_vs_sasrec",
            "rel_ndcg@10_vs_sasrec",
            "ndcg@10_retention_vs_grid_best",
        ],
    )

    aggregate_rows = aggregate(output_rows)
    aggregate_hidden_rows = aggregate_by_hidden(output_rows)
    write_csv(aggregate_rows, aggregate_csv)
    write_markdown(
        aggregate_rows,
        aggregate_md,
        [
            "method",
            "num_settings",
            "avg_recall@10",
            "avg_ndcg@10",
            "avg_rel_recall@10_vs_sasrec",
            "avg_rel_ndcg@10_vs_sasrec",
            "avg_ndcg@10_retention_vs_grid_best",
            "wins_vs_sasrec_ndcg@10",
        ],
    )
    write_csv(aggregate_hidden_rows, aggregate_by_hidden_csv)
    write_markdown(
        aggregate_hidden_rows,
        aggregate_by_hidden_md,
        [
            "hidden",
            "method",
            "num_settings",
            "avg_recall@10",
            "avg_ndcg@10",
            "avg_rel_recall@10_vs_sasrec",
            "avg_rel_ndcg@10_vs_sasrec",
            "avg_ndcg@10_retention_vs_grid_best",
            "wins_vs_sasrec_ndcg@10",
        ],
    )
    missing_path.write_text(
        "\n".join(f"{dataset},h{hidden},len{max_len}" for dataset, hidden, max_len in missing) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(output_rows)} method rows to {per_setting_csv}")
    print(f"wrote {len(aggregate_rows)} aggregate rows to {aggregate_csv}")
    print(f"wrote {len(aggregate_hidden_rows)} hidden aggregate rows to {aggregate_by_hidden_csv}")
    if missing:
        print(f"missing {len(missing)} settings; see {missing_path}")


if __name__ == "__main__":
    main()
