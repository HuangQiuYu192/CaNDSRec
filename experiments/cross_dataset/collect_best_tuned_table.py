#!/usr/bin/env python3
"""Collect true best-tuned results from existing experiment logs.

Selection rule:
  * SASRec: best NDCG@10 over available hidden sizes.
  * CaNDS: best NDCG@10 over available hidden sizes and temperatures.
  * AngularSmooth: best NDCG@10 over available smoothing grids/fixed runs.

This script only reads logs; it does not train or evaluate checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path


MAIN_NAME_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<model>SASRec|CANDSSASRec)_h(?P<hidden>\d+)_len(?P<max_len>\d+)"
    r"(?:_temp(?P<temp>[^_]+))?(?:_.*)?$"
)
AS_NAME_RE = re.compile(
    r"^(?P<dataset>.+)_AngularSmoothCANDSSASRec_h(?P<hidden>\d+)_len(?P<max_len>\d+)"
    r"_temp(?P<temp>[^_]+)_w(?P<weight>[^_]+)_k(?P<k>\d+)_st(?P<smooth_temp>[^_]+)"
    r"_q(?P<quantile>[^_]+)_thr(?P<threshold>[^_]+)(?:_.*)?$"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"((?:recall|ndcg)@\d+)\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.I)
METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
DEFAULT_DATASETS = ["Beauty", "Sports", "Toys", "Yelp-S3Rec", "LastFM-S3Rec"]
DEFAULT_MAIN_LOG_DIRS = [
    "log_runs/main_benchmark_grid",
    "log_runs/beauty_sasrec_fixed_inner256_grid_gpu0",
    "log_runs/beauty_sasrec_h128_grid_gpu0",
]
DEFAULT_AS_LOG_DIRS = [
    "log_runs/beauty_angular_smooth_cands_gpu0",
    "log_runs/amazon_angular_smooth_cross_dataset",
    "log_runs/yelp_angular_smooth_cross_dataset",
    "log_runs/yelp_angular_smooth_grid_gpu2_bs1024",
    "log_runs/lastfm_len200_angular_smooth_bs512",
]


def parse_test_result(log_path: Path) -> dict[str, float] | None:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="ignore"))
    pos = text.lower().rfind("test result")
    if pos < 0:
        return None
    block = text[pos : pos + 4000]

    dict_match = re.search(r"\{[^}]+\}", block, flags=re.S)
    if dict_match:
        parsed = ast.literal_eval(dict_match.group(0))
        return {str(k).lower(): float(v) for k, v in parsed.items()}

    ordered_match = re.search(r"OrderedDict\((\[[\s\S]*?\])\)", block)
    if ordered_match:
        parsed = dict(ast.literal_eval(ordered_match.group(1)))
        return {str(k).lower(): float(v) for k, v in parsed.items()}

    metrics = {name.lower(): float(value) for name, value in METRIC_RE.findall(block)}
    return metrics or None


def parse_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def read_main_logs(log_dirs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for log_path in sorted(log_dir.glob("*.log")):
            match = MAIN_NAME_RE.match(log_path.stem)
            if not match:
                continue
            result = parse_test_result(log_path)
            if result is None:
                continue
            info = match.groupdict()
            model = info["model"]
            row = {
                "dataset": info["dataset"],
                "method": "SASRec" if model == "SASRec" else "CaNDS",
                "model": model,
                "hidden": int(info["hidden"]),
                "max_len": int(info["max_len"]),
                "temp": "" if info["temp"] is None else info["temp"],
                "source": str(log_dir),
            }
            for metric in METRICS:
                row[metric] = float(result.get(metric, 0.0))
            key = (row["dataset"], row["method"], row["hidden"], row["max_len"], row["temp"], row["source"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def read_angular_smooth_logs(log_dirs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for log_path in sorted(log_dir.glob("*.log")):
            match = AS_NAME_RE.match(log_path.stem)
            if not match:
                continue
            result = parse_test_result(log_path)
            if result is None:
                continue
            info = match.groupdict()
            row = {
                "dataset": info["dataset"],
                "method": "AngularSmooth",
                "model": "AngularSmoothCANDSSASRec",
                "hidden": int(info["hidden"]),
                "max_len": int(info["max_len"]),
                "temp": info["temp"],
                "weight": info["weight"],
                "k": info["k"],
                "smooth_temp": info["smooth_temp"],
                "quantile": info["quantile"],
                "threshold": info["threshold"],
                "source": str(log_dir),
            }
            for metric in METRICS:
                row[metric] = float(result.get(metric, 0.0))
            key = (
                row["dataset"],
                row["hidden"],
                row["max_len"],
                row["temp"],
                row["weight"],
                row["k"],
                row["smooth_temp"],
                row["quantile"],
                row["threshold"],
                row["source"],
            )
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def select_best(rows: list[dict], datasets: set[str]) -> list[dict]:
    best_rows = []
    for dataset in sorted(datasets):
        for method in ["SASRec", "CaNDS", "AngularSmooth"]:
            candidates = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: (row["ndcg@10"], row["recall@10"], row["ndcg@20"], row["recall@20"]))
            best_rows.append({**best, "selection": "best_ndcg@10"})
    method_order = {"SASRec": 0, "CaNDS": 1, "AngularSmooth": 2}
    best_rows.sort(key=lambda row: (row["dataset"], method_order[row["method"]]))
    return best_rows


def add_relative_columns(rows: list[dict]) -> list[dict]:
    by_dataset_method = {(row["dataset"], row["method"]): row for row in rows}
    output = []
    for row in rows:
        sas = by_dataset_method.get((row["dataset"], "SASRec"))
        prev = None
        if row["method"] == "CaNDS":
            prev = sas
        elif row["method"] == "AngularSmooth":
            prev = by_dataset_method.get((row["dataset"], "CaNDS"))

        new_row = dict(row)
        for metric in ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]:
            value = parse_float(row.get(metric))
            sas_value = parse_float(sas.get(metric)) if sas else None
            prev_value = parse_float(prev.get(metric)) if prev else None
            if value is not None and sas_value and sas_value != 0:
                new_row[f"rel_{metric}_vs_best_sasrec"] = value / sas_value - 1.0
            if value is not None and prev_value and prev_value != 0:
                new_row[f"rel_{metric}_vs_prev"] = value / prev_value - 1.0
        output.append(new_row)
    return output


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict], headers: list[str]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_log_dirs", nargs="*", default=DEFAULT_MAIN_LOG_DIRS)
    parser.add_argument("--angular_log_dirs", nargs="*", default=DEFAULT_AS_LOG_DIRS)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--out_dir", default="analysis_results/best_tuned_table")
    args = parser.parse_args()

    datasets = set(args.datasets)
    rows = read_main_logs([Path(path) for path in args.main_log_dirs])
    rows.extend(read_angular_smooth_logs([Path(path) for path in args.angular_log_dirs]))
    rows = [row for row in rows if row["dataset"] in datasets]
    best_rows = add_relative_columns(select_best(rows, datasets))

    headers = [
        "dataset", "method", "model", "selection", "hidden", "max_len", "temp",
        "weight", "k", "smooth_temp", "quantile", "threshold",
        "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20",
        "rel_recall@10_vs_best_sasrec", "rel_ndcg@10_vs_best_sasrec",
        "rel_recall@10_vs_prev", "rel_ndcg@10_vs_prev", "source",
    ]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "best_tuned_table.csv", best_rows, headers)
    write_md(out_dir / "best_tuned_table.md", best_rows, headers)
    write_csv(out_dir / "all_candidates.csv", rows, headers)
    print(f"wrote {len(best_rows)} best rows to {out_dir / 'best_tuned_table.csv'} and .md")
    print(f"wrote {len(rows)} candidates to {out_dir / 'all_candidates.csv'}")


if __name__ == "__main__":
    main()
