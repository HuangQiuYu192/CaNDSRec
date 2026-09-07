#!/usr/bin/env python3
"""Prepare checkpoint evaluation tasks for best-tuned group metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def infer_run_name(row: dict) -> str:
    if row.get("run_name"):
        return row["run_name"]
    dataset = row["dataset"]
    model = row["model"]
    hidden = row["hidden"]
    max_len = row["max_len"]
    if model == "SASRec":
        return f"{dataset}_{model}_h{hidden}_len{max_len}"
    if model == "CANDSSASRec":
        return f"{dataset}_{model}_h{hidden}_len{max_len}_temp{row['temp']}"
    return (
        f"{dataset}_{model}_h{hidden}_len{max_len}_temp{row['temp']}"
        f"_w{row['weight']}_k{row['k']}_st{row['smooth_temp']}"
        f"_q{row['quantile']}_thr{row['threshold']}"
    )


def ckpt_root_from_log_source(source: str) -> Path:
    path = Path(source)
    parts = path.parts
    if parts and parts[0] == "log_runs":
        return Path("ckpt").joinpath(*parts[1:])
    if "log_runs" in parts:
        idx = parts.index("log_runs")
        return Path(*parts[:idx], "ckpt", *parts[idx + 1 :])
    return Path(source.replace("log_runs", "ckpt", 1))


def latest_checkpoint(ckpt_root: Path, run_name: str) -> str:
    run_dir = ckpt_root / run_name
    checkpoints = sorted(run_dir.glob("*.pth"))
    return str(checkpoints[-1]) if checkpoints else ""


def infer_inner_size(row: dict) -> int:
    source = row.get("source", "")
    if "fixed_inner256" in source:
        return 256
    return int(float(row["hidden"])) * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_csv", default="analysis_results/best_tuned_table/best_tuned_table.csv")
    parser.add_argument("--out_tsv", default="analysis_results/best_tuned_group_eval/tasks.tsv")
    args = parser.parse_args()

    best_csv = Path(args.best_csv)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with best_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_name = infer_run_name(row)
            ckpt_root = ckpt_root_from_log_source(row["source"])
            checkpoint = latest_checkpoint(ckpt_root, run_name)
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "model": row["model"],
                    "hidden": str(int(float(row["hidden"]))),
                    "max_len": str(int(float(row["max_len"]))),
                    "inner_size": str(infer_inner_size(row)),
                    "temp": row.get("temp") or "10",
                    "weight": row.get("weight") or "0.0",
                    "k": row.get("k") or "10",
                    "smooth_temp": row.get("smooth_temp") or "0.2",
                    "quantile": row.get("quantile") or "0.67",
                    "threshold": row.get("threshold") or "0.0",
                    "run_name": run_name,
                    "checkpoint": checkpoint,
                }
            )

    headers = [
        "dataset", "method", "model", "hidden", "max_len", "inner_size", "temp",
        "weight", "k", "smooth_temp", "quantile", "threshold", "run_name", "checkpoint",
    ]
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    missing = [row for row in rows if not row["checkpoint"]]
    print(f"wrote {len(rows)} tasks to {out_tsv}")
    if missing:
        print("missing checkpoints:")
        for row in missing:
            print(f"  {row['dataset']} {row['method']} {row['run_name']}")


if __name__ == "__main__":
    main()
