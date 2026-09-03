#!/usr/bin/env python3
"""Plot CANDSSASRec temperature sensitivity curves from benchmark summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", default="log_runs/main_benchmark_grid/summary.csv")
    parser.add_argument("--out_dir", default="analysis_results/temperature_sensitivity")
    parser.add_argument("--metrics", nargs="+", default=["recall@10", "ndcg@10"])
    args = parser.parse_args()

    rows = read_rows(Path(args.summary_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in rows:
        if row["model"] != "CANDSSASRec":
            continue
        key = (row["dataset"], int(row["hidden"]), int(row["max_len"]))
        groups.setdefault(key, []).append(row)

    for metric in args.metrics:
        plt.figure(figsize=(8, 5))
        for key, group in sorted(groups.items()):
            group = sorted(group, key=lambda r: as_float(r["temp"]))
            temps = [as_float(r["temp"]) for r in group]
            values = [as_float(r[metric]) for r in group]
            label = f"{key[0]} h{key[1]} L{key[2]}"
            plt.plot(temps, values, marker="o", linewidth=1.5, markersize=4, label=label)
        plt.xlabel("Temperature")
        plt.ylabel(metric)
        plt.title(f"CANDSSASRec Temperature Sensitivity ({metric})")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        safe_metric = metric.replace("@", "")
        png_path = out_dir / f"temperature_sensitivity_{safe_metric}.png"
        pdf_path = out_dir / f"temperature_sensitivity_{safe_metric}.pdf"
        plt.savefig(png_path, dpi=240)
        plt.savefig(pdf_path)
        plt.close()
        print(f"wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
