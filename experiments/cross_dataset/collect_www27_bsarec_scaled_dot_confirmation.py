"""Aggregate the held-out-seed BSARec scaled-dot confirmation."""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path


SEEDS = (2023, 2024, 2025)


def read_result(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    found = re.findall(r"test result:\s*OrderedDict\((\[.*?\])\)", text)
    if not found:
        raise ValueError(f"No completed test result in {path}")
    return dict(ast.literal_eval(found[-1]))


def mean_sd(values: list[float]) -> tuple[float, float]:
    mu = sum(values) / len(values)
    return mu, math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--dimension", type=int, default=256)
    args = parser.parse_args()
    root = args.root.resolve()
    tag = f"ScaledDotBSARec_d{args.dimension}_tau{args.temperature:g}_a0.5_c9_h2_lr0.001"
    rows = []
    for seed in SEEDS:
        path = root / "log_runs/www27_bsarec_scaled_dot_confirmation/Beauty" / f"seed{seed}" / f"{tag}.log"
        result = read_result(path)
        rows.append({"seed": seed, "temperature": args.temperature, "recall@10": result["recall@10"], "ndcg@10": result["ndcg@10"], "log": str(path)})

    out = root / "analysis_results/www27_bsarec_scaled_dot_confirmation"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    r_mu, r_sd = mean_sd([row["recall@10"] for row in rows])
    n_mu, n_sd = mean_sd([row["ndcg@10"] for row in rows])
    with (out / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# BSARec scaled-dot held-out-seed confirmation\n\n")
        handle.write(f"The global dot-product scale $\\tau={args.temperature:g}$ was fixed by seed-2026 validation before these three test runs. No test metric is used for selection.\n\n")
        handle.write("| metric | mean ± sample std. |\n| --- | ---: |\n")
        handle.write(f"| Recall@10 | {r_mu:.4f} ± {r_sd:.4f} |\n| NDCG@10 | {n_mu:.4f} ± {n_sd:.4f} |\n\n")
        handle.write("| seed | Recall@10 | NDCG@10 |\n| ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(f"| {row['seed']} | {row['recall@10']:.4f} | {row['ndcg@10']:.4f} |\n")
    print(f"wrote {out / 'per_seed.csv'} and summary.md")


if __name__ == "__main__":
    main()
