#!/usr/bin/env python3
"""Join temperature calibration estimates with benchmark-grid results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
GRID_TEMPS = [2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0]
DEFAULT_DAMPING_VALUES = [0.4, 0.5, 0.6, 0.7]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return float(value) if value not in {"", None} else math.nan


def nearest_grid_temp(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return min(GRID_TEMPS, key=lambda x: abs(x - value))


def parse_damping_values(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(x) for x in value.replace(",", " ").split()]


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


def metric_at_temp(rows: list[dict[str, str]], temp: float, metric: str) -> float:
    if not math.isfinite(temp):
        return math.nan
    for row in rows:
        if abs(as_float(row.get("temp", "")) - temp) < 1e-8:
            return as_float(row.get(metric, ""))
    return math.nan


def add_auto_method(
    out: dict,
    cand_rows: list[dict[str, str]],
    method: str,
    raw_temp: float,
    select_metric: str,
    damping_values: list[float],
) -> None:
    grid_temp = nearest_grid_temp(raw_temp)
    out[f"auto_temp_{method}"] = raw_temp
    out[f"auto_grid_{method}"] = grid_temp
    out[f"auto_{method}_recall@10"] = metric_at_temp(cand_rows, grid_temp, "recall@10")
    out[f"auto_{method}_ndcg@10"] = metric_at_temp(cand_rows, grid_temp, "ndcg@10")
    out[f"auto_{method}_{select_metric}"] = metric_at_temp(cand_rows, grid_temp, select_metric)

    for damping in damping_values:
        suffix = f"{method}_x{damping:g}"
        damped_temp = raw_temp * damping if math.isfinite(raw_temp) else math.nan
        damped_grid = nearest_grid_temp(damped_temp)
        out[f"auto_temp_{suffix}"] = damped_temp
        out[f"auto_grid_{suffix}"] = damped_grid
        out[f"auto_{suffix}_recall@10"] = metric_at_temp(cand_rows, damped_grid, "recall@10")
        out[f"auto_{suffix}_ndcg@10"] = metric_at_temp(cand_rows, damped_grid, "ndcg@10")


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
        "auto_grid_adaptau_all",
        "auto_grid_adaptau_all_x0.5",
        "auto_grid_adaptau_neg",
        "auto_grid_adaptau_neg_x0.5",
        "auto_grid_adaptau_hard",
        "dot_std",
        "cosine_std",
        "gap_pos_all_cosine",
        "gap_pos_neg_cosine",
        "gap_pos_hard_cosine",
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
    parser.add_argument(
        "--damping_values",
        default=" ".join(str(x) for x in DEFAULT_DAMPING_VALUES),
        help="Multipliers tested for over-estimated analytic temperatures.",
    )
    args = parser.parse_args()
    damping_values = parse_damping_values(args.damping_values)

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

        out = {
            "dataset": key[0],
            "hidden": key[1],
            "max_len": key[2],
            "sasrec_recall@10": as_float(sasrec["recall@10"]),
            "sasrec_ndcg@10": as_float(sasrec["ndcg@10"]),
            "best_temp": as_float(best["temp"]),
            "best_recall@10": as_float(best["recall@10"]),
            "best_ndcg@10": as_float(best["ndcg@10"]),
            "dot_std": as_float(calib.get("dot_std", "")),
            "cosine_std": as_float(calib.get("cosine_std", "")),
            "margin_dot_std": as_float(calib.get("margin_dot_std", "")),
            "margin_cosine_std": as_float(calib.get("margin_cosine_std", "")),
            "pos_cosine_mean": as_float(calib.get("pos_cosine_mean", "")),
            "neg_cosine_mean": as_float(calib.get("neg_cosine_mean", "")),
            "hard_cosine_mean": as_float(calib.get("hard_cosine_mean", "")),
            "adap_log_density": as_float(calib.get("adap_log_density", "")),
            "gap_pos_all_cosine": as_float(calib.get("gap_pos_all_cosine", "")),
            "gap_pos_neg_cosine": as_float(calib.get("gap_pos_neg_cosine", "")),
            "gap_pos_hard_cosine": as_float(calib.get("gap_pos_hard_cosine", "")),
        }
        add_auto_method(out, cand_rows, "std", as_float(calib.get("temp_by_logit_std", "")), args.select_metric, [])
        add_auto_method(out, cand_rows, "margin", as_float(calib.get("temp_by_margin_std", "")), args.select_metric, [])
        add_auto_method(out, cand_rows, "pos_neg_gap", as_float(calib.get("temp_by_pos_neg_gap", "")), args.select_metric, [])
        add_auto_method(
            out,
            cand_rows,
            "adaptau_all",
            as_float(calib.get("temp_by_adaptau_all_gap", "")),
            args.select_metric,
            damping_values,
        )
        add_auto_method(
            out,
            cand_rows,
            "adaptau_neg",
            as_float(calib.get("temp_by_adaptau_neg_gap", "")),
            args.select_metric,
            damping_values,
        )
        add_auto_method(
            out,
            cand_rows,
            "adaptau_hard",
            as_float(calib.get("temp_by_adaptau_hard_gap", "")),
            args.select_metric,
            damping_values,
        )
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
