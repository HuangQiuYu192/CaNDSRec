"""Aggregate held-out BSARec positive sequence-scale-jitter confirmation."""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


SEEDS = (2023, 2024, 2025)


def read_test(path: Path) -> dict[str, float]:
    found = re.findall(r"test result:\s*OrderedDict\((\[.*?\])\)", path.read_text(encoding="utf-8", errors="replace"))
    if not found:
        raise ValueError(f"No completed test result in {path}")
    return dict(ast.literal_eval(found[-1]))


def mean_sd(values: list[float]) -> tuple[float, float]:
    mu = sum(values) / len(values)
    return mu, math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def dot_path(root: Path, seed: int) -> Path:
    tag = "BSARec_d256_a0.5_c9_h2_lr0.001.log"
    base = root / ("log_runs/www27_bsarec_dimension_validation/Beauty/seed2025" if seed == 2025 else f"log_runs/www27_bsarec_dimension_replication/Beauty/seed{seed}")
    return base / tag


def sequence_path(root: Path, seed: int) -> Path:
    tag = "GeometryBSARec_sequence_d256_tau4_a0.5_c9_h2_lr0.001.log"
    return root / f"log_runs/www27_bsarec_sequence_mechanism_confirmation/Beauty/seed{seed}/{tag}"


def jitter_path(root: Path, seed: int, variant: str, sigma: float) -> Path:
    tag = f"{variant}_sigma{sigma:g}_d256_a0.5_c9_h2_lr0.001.log"
    return root / f"log_runs/www27_bsarec_scale_jitter_confirmation/Beauty/seed{seed}/{tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sigma", type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = {
        "dot": lambda seed: dot_path(root, seed),
        "dot_jitter": lambda seed: jitter_path(root, seed, "dot_jitter", args.sigma),
        "sequence": lambda seed: sequence_path(root, seed),
        "sequence_jitter": lambda seed: jitter_path(root, seed, "sequence_jitter", args.sigma),
    }
    rows = []
    for seed in SEEDS:
        for variant, get_path in sources.items():
            result = read_test(get_path(seed))
            rows.append({"seed": seed, "variant": variant, "recall@10": result["recall@10"], "ndcg@10": result["ndcg@10"], "log": str(get_path(seed))})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = []
    for variant in sources:
        group = grouped[variant]
        r_mu, r_sd = mean_sd([row["recall@10"] for row in group])
        n_mu, n_sd = mean_sd([row["ndcg@10"] for row in group])
        reference = "dot" if variant.startswith("dot") else "sequence"
        paired = [row["ndcg@10"] - next(base["ndcg@10"] for base in grouped[reference] if base["seed"] == row["seed"]) for row in group]
        d_mu, d_sd = mean_sd(paired)
        summary.append({"variant": variant, "recall@10_mean": r_mu, "recall@10_sd": r_sd, "ndcg@10_mean": n_mu, "ndcg@10_sd": n_sd, "paired_delta_ndcg@10_mean": d_mu, "paired_delta_ndcg@10_sd": d_sd})
    out = root / "analysis_results/www27_bsarec_scale_jitter_confirmation"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# BSARec positive sequence-scale jitter confirmation", "", f"sigma={args.sigma:g} is the largest pre-specified pilot perturbation. The unjittered same-seed baselines are reused; no test value selects sigma.", "", "| variant | Recall@10 | NDCG@10 | paired delta NDCG@10 vs. corresponding unjittered head |", "| --- | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['recall@10_mean']:.4f} ± {row['recall@10_sd']:.4f} | {row['ndcg@10_mean']:.4f} ± {row['ndcg@10_sd']:.4f} | {row['paired_delta_ndcg@10_mean']:+.4f} ± {row['paired_delta_ndcg@10_sd']:.4f} |")
    lines.append("\nPositive scale jitter is cancelled by the sequence-normalized forward score but not by raw dot scoring.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'per_seed.csv'}, summary.csv and summary.md")


if __name__ == "__main__":
    main()
