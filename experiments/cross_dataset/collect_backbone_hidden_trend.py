#!/usr/bin/env python3
"""Collect hidden-size trends across SASRec, WEARec, and FMLPRec backbones."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
DEFAULT_BACKBONES = {
    "SASRec": ("SASRec", "CANDSSASRec", "log_runs/main_benchmark_grid/summary.csv"),
    "WEARec": ("WEARec", "CANDSWEARec", "log_runs/beauty_wearec_cands_grid/summary.csv"),
    "FMLPRec": ("FMLPRec", "CANDSFMLPRec", "log_runs/beauty_fmlprec_cands_grid_gpu0/summary.csv"),
}


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


def parse_backbone_specs(specs: list[str]) -> dict[str, tuple[str, str, str]]:
    if not specs:
        return DEFAULT_BACKBONES
    output = {}
    for spec in specs:
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                "--backbone entries must be backbone:base_model:cands_model:summary_csv"
            )
        output[parts[0]] = (parts[1], parts[2], parts[3])
    return output


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, str], list[dict[str, str]]]:
    grouped = {}
    for row in rows:
        key = (row["dataset"], int(row["hidden"]), int(row["max_len"]), row["model"])
        grouped.setdefault(key, []).append(row)
    return grouped


def best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: as_float(row.get(metric)))


def row_at_temp(rows: list[dict[str, str]], temp: float) -> dict[str, str] | None:
    for row in rows:
        if abs(as_float(row.get("temp")) - temp) < 1e-8:
            return row
    return None


def add_metric_columns(out: dict, prefix: str, row: dict[str, str] | None) -> None:
    for metric in METRICS:
        out[f"{prefix}_{metric}"] = as_float(row.get(metric)) if row is not None else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--select_metric", default="ndcg@10")
    parser.add_argument("--fixed_temp", default=10.0, type=float)
    parser.add_argument(
        "--backbone",
        action="append",
        default=[],
        help="Optional backbone:base_model:cands_model:summary_csv. Can be repeated.",
    )
    parser.add_argument("--out_dir", default="analysis_results/backbone_hidden_trend")
    args = parser.parse_args()

    specs = parse_backbone_specs(args.backbone)
    output_rows = []
    missing = []

    for backbone, (base_model, cands_model, summary_path) in specs.items():
        path = Path(summary_path)
        if not path.exists():
            missing.append(f"{backbone}: missing summary {summary_path}")
            continue
        grouped = group_rows(read_csv(path))
        settings = sorted(
            {
                (dataset, hidden, max_len)
                for dataset, hidden, max_len, model in grouped
                if dataset == args.dataset and model in {base_model, cands_model}
            }
        )
        for dataset, hidden, max_len in settings:
            base_rows = grouped.get((dataset, hidden, max_len, base_model), [])
            cands_rows = grouped.get((dataset, hidden, max_len, cands_model), [])
            if not base_rows or not cands_rows:
                missing.append(f"{backbone}: missing rows for {dataset} h{hidden} len{max_len}")
                continue
            base = base_rows[0]
            fixed = row_at_temp(cands_rows, args.fixed_temp)
            best = best_row(cands_rows, args.select_metric)

            row = {
                "dataset": dataset,
                "backbone": backbone,
                "hidden": hidden,
                "max_len": max_len,
                "base_model": base_model,
                "cands_model": cands_model,
                "fixed_temp": args.fixed_temp,
                "best_temp": as_float(best.get("temp")),
            }
            add_metric_columns(row, "base", base)
            add_metric_columns(row, "fixed", fixed)
            add_metric_columns(row, "best", best)
            for metric in ["recall@10", "ndcg@10"]:
                base_value = row[f"base_{metric}"]
                fixed_value = row[f"fixed_{metric}"]
                best_value = row[f"best_{metric}"]
                row[f"fixed_delta_{metric}"] = fixed_value - base_value
                row[f"best_delta_{metric}"] = best_value - base_value
                row[f"fixed_rel_{metric}"] = fixed_value / base_value - 1 if base_value > 0 else math.nan
                row[f"best_rel_{metric}"] = best_value / base_value - 1 if base_value > 0 else math.nan
            output_rows.append(row)

    output_rows.sort(key=lambda row: (row["backbone"], row["hidden"], row["max_len"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"
    out_md = out_dir / "summary.md"
    missing_path = out_dir / "missing.txt"

    headers = list(output_rows[0].keys()) if output_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(output_rows)

    md_headers = [
        "dataset",
        "backbone",
        "hidden",
        "base_recall@10",
        "base_ndcg@10",
        "fixed_temp",
        "fixed_recall@10",
        "fixed_ndcg@10",
        "fixed_rel_recall@10",
        "fixed_rel_ndcg@10",
        "best_temp",
        "best_recall@10",
        "best_ndcg@10",
        "best_rel_recall@10",
        "best_rel_ndcg@10",
    ]
    lines = [
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join(["---"] * len(md_headers)) + " |",
    ]
    for row in output_rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in md_headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"wrote {len(output_rows)} rows to {out_csv} and {out_md}")
    if missing:
        print(f"missing {len(missing)} entries; see {missing_path}")


if __name__ == "__main__":
    main()
