#!/usr/bin/env python3
"""Validation-only temperature selection report for Sports/Toys Scaled-Dot."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR_RE = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
RESULT_RE = re.compile(r"(best valid result:|test result:)\s*OrderedDict\(\[(.*?)\]\)", re.S)


def read(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    records = {label: {key: float(value) for key, value in PAIR_RE.findall(payload)}
               for label, payload in RESULT_RE.findall(path.read_text(encoding="utf-8", errors="replace"))}
    if "best valid result:" not in records or "test result:" not in records:
        raise ValueError(f"Incomplete log: {path}")
    return records["best valid result:"], records["test result:"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("log_runs/www27_scaled_dot_temperature_screen_gpu1"))
    parser.add_argument("--datasets", nargs="+", default=["Sports", "Toys"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0625, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--out", type=Path, default=Path("analysis_results/www27_scaled_dot_temperature_screen/summary.csv"))
    args = parser.parse_args()
    rows = []
    selected = {}
    for dataset in args.datasets:
        candidates = []
        for tau in args.temperatures:
            tag = str(int(tau)) if tau.is_integer() else str(tau)
            valid, test = read(args.log_root / dataset / f"seed{args.seed}" / f"scaled_dot_tau{tag}.log")
            row = {"dataset": dataset, "seed": args.seed, "temperature": tau,
                   "valid_recall@10": valid["recall@10"], "valid_ndcg@10": valid["ndcg@10"],
                   "test_recall@10": test["recall@10"], "test_ndcg@10": test["ndcg@10"]}
            candidates.append(row)
        chosen = max(candidates, key=lambda row: row["valid_ndcg@10"])
        selected[dataset] = chosen["temperature"]
        for row in candidates:
            row["selected_by_valid_ndcg@10"] = row["temperature"] == chosen["temperature"]
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Scaled-Dot temperature development screen", "",
        "The selected temperature maximises validation NDCG@10 within each dataset. Test columns are descriptive and play no role in selection.", "",
        "| dataset | tau | validation NDCG@10 | validation Recall@10 | test NDCG@10 | test Recall@10 | selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['temperature']:.4g} | {row['valid_ndcg@10']:.4f} | {row['valid_recall@10']:.4f} | {row['test_ndcg@10']:.4f} | {row['test_recall@10']:.4f} | {'yes' if row['selected_by_valid_ndcg@10'] else ''} |")
    lines.extend(["", "## Locked temperatures for confirmation", ""])
    lines.extend(f"- {dataset}: `tau={tau:g}`" for dataset, tau in selected.items())
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("selected " + " ".join(f"{dataset}=tau{tau:g}" for dataset, tau in selected.items()))


if __name__ == "__main__":
    main()
