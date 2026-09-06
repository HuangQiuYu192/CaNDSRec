#!/usr/bin/env python3
"""Collect unified paper tables for SASRec, CaNDS, and AngularSmooth-CaNDS.

The script intentionally reads only experiment artifacts and does not train
models. It is designed for the Jupyter server after experiments finish.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


MAIN_NAME_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<model>SASRec|CANDSSASRec)_h(?P<hidden>\d+)_len(?P<max_len>\d+)"
    r"(?:_temp(?P<temp>[^_]+))?$"
)
AS_NAME_RE = re.compile(
    r"^(?P<dataset>.+)_AngularSmoothCANDSSASRec_h(?P<hidden>\d+)_len(?P<max_len>\d+)"
    r"_temp(?P<temp>[^_]+)_w(?P<weight>[^_]+)_k(?P<k>\d+)_st(?P<smooth_temp>[^_]+)"
    r"_q(?P<quantile>[^_]+)_thr(?P<threshold>[^_]+)(?:_.*)?$"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"((?:recall|ndcg)@\d+)\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.I)
METRICS = ["recall@5", "recall@10", "recall@20", "recall@50", "recall@100", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100"]

DEFAULT_DATASETS = ["Beauty", "Sports", "Toys", "Yelp-S3Rec", "LastFM-S3Rec"]
DEFAULT_GROUP_DIRS = [
    "analysis_results/amazon_angular_smooth_cross_dataset",
    "analysis_results/yelp_angular_smooth_cross_dataset",
    "analysis_results/lastfm_len200_angular_smooth_bs512",
    "analysis_results/ml1m_len50_angular_smooth",
    "analysis_results/ml1m_lastfm_len200_angular_smooth_cross_dataset",
    "analysis_results/sasrec_group_eval",
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


def parse_float(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | int | float | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None or math.isnan(parsed):
        return None
    return int(parsed)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def read_main_logs(log_dir: Path) -> list[dict]:
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        match = MAIN_NAME_RE.match(log_path.stem)
        if not match:
            continue
        result = parse_test_result(log_path)
        if result is None:
            continue
        info = match.groupdict()
        row = {
            "dataset": info["dataset"],
            "model": info["model"],
            "method": "SASRec" if info["model"] == "SASRec" else "CaNDS",
            "hidden": int(info["hidden"]),
            "max_len": int(info["max_len"]),
            "temp": "" if info["temp"] is None else info["temp"],
        }
        for metric in METRICS:
            if metric in result:
                row[metric] = result[metric]
        rows.append(row)
    return rows


def read_angular_smooth_logs(log_dirs: list[Path]) -> list[dict]:
    rows = []
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
                "model": "AngularSmoothCANDSSASRec",
                "method": "AngularSmooth",
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
                if metric in result:
                    row[metric] = result[metric]
            rows.append(row)
    return rows


def read_group_rows(group_dirs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for group_dir in group_dirs:
        if not group_dir.exists():
            continue
        for csv_path in sorted(group_dir.glob("*.csv")):
            if csv_path.name == "summary.csv":
                continue
            with csv_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (row.get("dataset"), row.get("tag"), row.get("model"), row.get("group"))
                    if key in seen:
                        continue
                    seen.add(key)
                    row["source"] = str(group_dir)
                    rows.append(row)
    return rows


def best_or_selected_cands(rows: list[dict], selected_temps: dict[tuple[str, int, int], str]) -> list[dict]:
    output = []
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] == "CaNDS":
            grouped[(row["dataset"], row["hidden"], row["max_len"])].append(row)

    for key, candidates in grouped.items():
        selected_temp = selected_temps.get(key)
        selected = None
        if selected_temp is not None:
            for row in candidates:
                if str(row.get("temp")) == selected_temp:
                    selected = row
                    break
        if selected is None:
            selected = max(candidates, key=lambda r: (float(r.get("ndcg@10", -1.0)), float(r.get("recall@10", -1.0))))
        output.append({**selected, "selection": "selected_temp" if selected_temp is not None else "best_ndcg@10"})
    return output


def default_selected_temps() -> dict[tuple[str, int, int], str]:
    selected = {}
    for dataset in ["Beauty", "Sports", "Toys", "Yelp-S3Rec"]:
        selected[(dataset, 256, 50)] = "10"
    selected[("LastFM-S3Rec", 256, 200)] = "10"
    selected[("ML-1M", 256, 50)] = "20"
    selected[("ML-1M", 256, 200)] = "20"
    return selected


def select_overall_rows(main_rows: list[dict], as_rows: list[dict], datasets: set[str], hidden: int | None) -> list[dict]:
    selected_temps = default_selected_temps()
    sasrec = [r for r in main_rows if r["method"] == "SASRec"]
    cands = best_or_selected_cands(main_rows, selected_temps)

    rows = []
    for row in [*sasrec, *cands, *as_rows]:
        if row["dataset"] not in datasets:
            continue
        if hidden is not None and row["hidden"] != hidden:
            continue
        if row["method"] == "AngularSmooth":
            row = {**row, "selection": "fixed_smoothing"}
        rows.append(row)

    method_order = {"SASRec": 0, "CaNDS": 1, "AngularSmooth": 2}
    rows.sort(key=lambda r: (r["dataset"], r["hidden"], r["max_len"], method_order.get(r["method"], 99), str(r.get("temp", ""))))
    return rows


def add_relative_columns(rows: list[dict]) -> list[dict]:
    by_setting_method = {(r["dataset"], r["hidden"], r["max_len"], r["method"]): r for r in rows}
    output = []
    for row in rows:
        base_key = (row["dataset"], row["hidden"], row["max_len"])
        sas = by_setting_method.get((*base_key, "SASRec"))
        cands = by_setting_method.get((*base_key, "CaNDS"))
        new_row = dict(row)
        for metric in ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]:
            value = parse_float(row.get(metric))
            sas_value = parse_float(sas.get(metric)) if sas else None
            cands_value = parse_float(cands.get(metric)) if cands else None
            if value is not None and sas_value and sas_value != 0:
                new_row[f"rel_{metric}_vs_sasrec"] = value / sas_value - 1.0
            if value is not None and cands_value and cands_value != 0:
                new_row[f"rel_{metric}_vs_cands"] = value / cands_value - 1.0
        output.append(new_row)
    return output


def filter_group_rows(rows: list[dict], datasets: set[str], hidden: int | None) -> list[dict]:
    output = []
    for row in rows:
        dataset = row.get("dataset", "")
        row_hidden = parse_int(row.get("hidden"))
        if dataset not in datasets:
            continue
        if hidden is not None and row_hidden != hidden:
            continue
        model = row.get("model", "")
        if model == "SASRec":
            row["method"] = "SASRec"
        elif model == "CANDSSASRec":
            row["method"] = "CaNDS"
        elif model == "AngularSmoothCANDSSASRec":
            row["method"] = "AngularSmooth"
        else:
            continue
        output.append(row)

    method_order = {"SASRec": 0, "CaNDS": 1, "AngularSmooth": 2}
    group_order = {"all": 0, "head": 1, "mid": 2, "tail": 3}
    output.sort(
        key=lambda r: (
            r.get("dataset", ""),
            parse_int(r.get("hidden")) or 0,
            parse_int(r.get("max_len")) or 0,
            method_order.get(r.get("method"), 99),
            group_order.get(r.get("group"), 99),
        )
    )
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
    parser.add_argument("--main_log_dir", default="log_runs/main_benchmark_grid")
    parser.add_argument("--angular_log_dirs", nargs="*", default=[
        "log_runs/amazon_angular_smooth_cross_dataset",
        "log_runs/yelp_angular_smooth_cross_dataset",
        "log_runs/lastfm_len200_angular_smooth_bs512",
        "log_runs/ml1m_len50_angular_smooth",
        "log_runs/ml1m_lastfm_len200_angular_smooth_cross_dataset",
    ])
    parser.add_argument("--group_dirs", nargs="*", default=DEFAULT_GROUP_DIRS)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dir", default="analysis_results/unified_main_table")
    args = parser.parse_args()

    datasets = set(args.datasets)
    out_dir = Path(args.out_dir)

    main_rows = read_main_logs(Path(args.main_log_dir))
    as_rows = read_angular_smooth_logs([Path(p) for p in args.angular_log_dirs])
    overall_rows = add_relative_columns(select_overall_rows(main_rows, as_rows, datasets, args.hidden))

    overall_headers = [
        "dataset", "method", "model", "hidden", "max_len", "temp", "selection",
        "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
        "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
        "rel_recall@10_vs_sasrec", "rel_ndcg@10_vs_sasrec",
        "rel_recall@10_vs_cands", "rel_ndcg@10_vs_cands",
        "weight", "k", "smooth_temp", "quantile", "threshold", "source",
    ]
    write_csv(out_dir / "overall_table.csv", overall_rows, overall_headers)
    write_md(out_dir / "overall_table.md", overall_rows, overall_headers)

    group_rows = filter_group_rows(read_group_rows([Path(p) for p in args.group_dirs]), datasets, args.hidden)
    group_headers = [
        "dataset", "method", "model", "hidden", "max_len", "temperature",
        "group", "n", "median_rank", "mean_rank",
        "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
        "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100",
        "angular_smooth_weight", "angular_smooth_k", "angular_smooth_temperature",
        "angular_smooth_pop_quantile", "angular_smooth_sim_threshold", "tag", "source",
    ]
    write_csv(out_dir / "group_table.csv", group_rows, group_headers)
    write_md(out_dir / "group_table.md", group_rows, group_headers)

    print(f"wrote {len(overall_rows)} overall rows to {out_dir / 'overall_table.csv'} and .md")
    print(f"wrote {len(group_rows)} group rows to {out_dir / 'group_table.csv'} and .md")


if __name__ == "__main__":
    main()
