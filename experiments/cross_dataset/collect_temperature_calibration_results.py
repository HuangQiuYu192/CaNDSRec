#!/usr/bin/env python3
"""Join temperature calibration estimates with benchmark-grid results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
GRID_TEMPS = [2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return float(value) if value not in {"", None} else math.nan


def nearest_grid_temp(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return min(GRID_TEMPS, key=lambda x: abs(x - value))


def load_calibration_rows(calibration_dir: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = {}
    for path in sorted(calibration_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        for row in read_csv(path):
            key = (row["dataset"], int(row["hidden"]), int(row["max_len"]))
            rows[key] = row
    return rows


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "dataset",
        "hidden",
        "max_len",
        "sasrec_ndcg@10",
        "best_temp",
        "best_ndcg@10",
        "best_recall@10",
        "auto_temp_std",
        "auto_grid_std",
        "auto_temp_margin",
        "auto_grid_margin",
        "dot_std",
        "cosine_std",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_csv", default="log_runs/main_benchmark_grid/summary.csv")
    parser.add_argument("--calibration_dir", default="analysis_results/temperature_calibration")
    parser.add_argument("--select_metric", default="ndcg@10")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()

    benchmark = read_csv(Path(args.benchmark_csv))
    calibration = load_calibration_rows(Path(args.calibration_dir))

    groups: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in benchmark:
        key = (row["dataset"], int(row["hidden"]), int(row["max_len"]))
        groups.setdefault(key, []).append(row)

    output_rows = []
    for key, rows in sorted(groups.items()):
        sasrec_rows = [r for r in rows if r["model"] == "SASRec"]
        cand_rows = [r for r in rows if r["model"] == "CANDSSASRec"]
        if not sasrec_rows or not cand_rows:
            continue
        sasrec = sasrec_rows[0]
        best = max(cand_rows, key=lambda r: as_float(r[args.select_metric]))
        calib = calibration.get(key, {})
        auto_std = as_float(calib.get("temp_by_logit_std", ""))
        auto_margin = as_float(calib.get("temp_by_margin_std", ""))

        out = {
            "dataset": key[0],
            "hidden": key[1],
            "max_len": key[2],
            "sasrec_recall@10": as_float(sasrec["recall@10"]),
            "sasrec_ndcg@10": as_float(sasrec["ndcg@10"]),
            "best_temp": as_float(best["temp"]),
            "best_recall@10": as_float(best["recall@10"]),
            "best_ndcg@10": as_float(best["ndcg@10"]),
            "auto_temp_std": auto_std,
            "auto_grid_std": nearest_grid_temp(auto_std),
            "auto_temp_margin": auto_margin,
            "auto_grid_margin": nearest_grid_temp(auto_margin),
            "dot_std": as_float(calib.get("dot_std", "")),
            "cosine_std": as_float(calib.get("cosine_std", "")),
            "margin_dot_std": as_float(calib.get("margin_dot_std", "")),
            "margin_cosine_std": as_float(calib.get("margin_cosine_std", "")),
        }
        output_rows.append(out)

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.calibration_dir) / "summary.csv"
    out_md = Path(args.out_md) if args.out_md else Path(args.calibration_dir) / "summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = list(output_rows[0].keys()) if output_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(output_rows)
    write_markdown(output_rows, out_md)
    print(f"wrote {len(output_rows)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
