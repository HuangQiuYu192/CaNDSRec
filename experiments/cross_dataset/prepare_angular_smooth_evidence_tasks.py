#!/usr/bin/env python3
"""Pair each best AngularSmooth run with a matched CaNDS checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.prepare_best_tuned_group_eval_tasks import ckpt_root_from_log_source


def latest_checkpoint(root: Path, run_name: str) -> str:
    if not root.exists():
        return ""
    direct = list((root / run_name).glob("*.pth"))
    if direct:
        return str(max(direct, key=lambda path: path.stat().st_mtime))
    matches = list(root.rglob(f"{run_name}/*.pth"))
    return str(max(matches, key=lambda path: path.stat().st_mtime)) if matches else ""


def checkpoint(row: dict, search_root: Path) -> str:
    return latest_checkpoint(ckpt_root_from_log_source(row["source"]), row["run_name"]) or latest_checkpoint(search_root, row["run_name"])


def infer_inner_size(row: dict) -> int:
    """Recover the only non-default architecture variant used in these runs."""
    if "fixed_inner256" in row.get("source", ""):
        return 256
    return int(float(row["hidden"])) * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", default="analysis_results/best_tuned_table/all_candidates.csv")
    parser.add_argument("--datasets", nargs="*", default=["Beauty", "Sports", "Toys", "LastFM-S3Rec", "Yelp-S3Rec"])
    parser.add_argument("--ckpt_root", default="ckpt")
    parser.add_argument("--out_tsv", default="analysis_results/angular_smooth_evidence/tasks.tsv")
    args = parser.parse_args()

    with Path(args.candidates_csv).open(encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    rows = []
    for dataset in args.datasets:
        smooth_runs = [row for row in candidates if row["dataset"] == dataset and row["method"] == "AngularSmooth"]
        if not smooth_runs:
            print(f"SKIP {dataset}: no AngularSmooth result")
            continue
        smooth = max(smooth_runs, key=lambda row: (float(row["ndcg@10"]), float(row["recall@10"])))
        smooth_inner_size = infer_inner_size(smooth)
        matched_base = [
            row for row in candidates
            if row["dataset"] == dataset and row["method"] == "CaNDS"
            and row["hidden"] == smooth["hidden"] and row["max_len"] == smooth["max_len"] and row["temp"] == smooth["temp"]
            and infer_inner_size(row) == smooth_inner_size
        ]
        if not matched_base:
            print(f"SKIP {dataset}: no CaNDS run matching h={smooth['hidden']}, len={smooth['max_len']}, temp={smooth['temp']}, inner={smooth_inner_size}")
            continue
        base = max(matched_base, key=lambda row: (float(row["ndcg@10"]), float(row["recall@10"])))
        rows.append({
            "dataset": dataset, "hidden": smooth["hidden"], "max_len": smooth["max_len"], "temp": smooth["temp"],
            "cands_inner_size": infer_inner_size(base), "smooth_inner_size": smooth_inner_size,
            "weight": smooth["weight"], "k": smooth["k"],
            "smooth_temp": smooth["smooth_temp"], "quantile": smooth["quantile"], "threshold": smooth["threshold"],
            "base_run": base["run_name"], "smooth_run": smooth["run_name"],
            "base_checkpoint": checkpoint(base, Path(args.ckpt_root)),
            "smooth_checkpoint": checkpoint(smooth, Path(args.ckpt_root)),
        })
    headers = ["dataset", "hidden", "max_len", "temp", "cands_inner_size", "smooth_inner_size", "weight", "k", "smooth_temp", "quantile", "threshold", "base_run", "smooth_run", "base_checkpoint", "smooth_checkpoint"]
    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tasks to {out}")
    for row in rows:
        if not row["base_checkpoint"] or not row["smooth_checkpoint"]:
            print(f"MISSING {row['dataset']}: base={bool(row['base_checkpoint'])} smooth={bool(row['smooth_checkpoint'])}")


if __name__ == "__main__":
    main()
