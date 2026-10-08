"""Aggregate the BSARec scale, sequence-norm, and gradient controls."""
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
    matches = re.findall(r"test result:\s*OrderedDict\((\[.*?\])\)", path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed test result in {path}")
    return dict(ast.literal_eval(matches[-1]))


def base_tag(model: str) -> str:
    return f"{model}_d256_a0.5_c9_h2_lr0.001"


def original_path(root: Path, seed: int, model: str) -> Path:
    if seed == 2025:
        folder = root / "log_runs/www27_bsarec_dimension_validation/Beauty/seed2025"
    else:
        folder = root / f"log_runs/www27_bsarec_dimension_replication/Beauty/seed{seed}"
    return folder / f"{base_tag(model)}.log"


def control_path(root: Path, seed: int, model: str, temperature: float) -> Path:
    tag = f"{model}_sequence_d256_tau{temperature:g}_a0.5_c9_h2_lr0.001"
    return root / f"log_runs/www27_bsarec_sequence_mechanism_confirmation/Beauty/seed{seed}/{tag}.log"


def scaled_path(root: Path, seed: int, temperature: float) -> Path:
    tag = f"ScaledDotBSARec_d256_tau{temperature:g}_a0.5_c9_h2_lr0.001"
    return root / f"log_runs/www27_bsarec_scaled_dot_confirmation/Beauty/seed{seed}/{tag}.log"


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    readers = {
        "dot": lambda seed: original_path(root, seed, "BSARec"),
        "scaled_dot": lambda seed: scaled_path(root, seed, 0.04),
        "sequence": lambda seed: control_path(root, seed, "GeometryBSARec", args.temperature),
        "stopgrad_sequence": lambda seed: control_path(root, seed, "StopGradSequenceGeometryBSARec", args.temperature),
        "joint": lambda seed: original_path(root, seed, "CANDSBSARec"),
    }
    rows = []
    for seed in SEEDS:
        for variant, get_path in readers.items():
            result = read_test(get_path(seed))
            rows.append({"seed": seed, "variant": variant, "recall@10": result["recall@10"], "ndcg@10": result["ndcg@10"], "log": str(get_path(seed))})
    out = root / "analysis_results/www27_bsarec_sequence_mechanism_confirmation"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = []
    for variant in readers:
        group = grouped[variant]
        r_mean, r_sd = mean_sd([row["recall@10"] for row in group])
        n_mean, n_sd = mean_sd([row["ndcg@10"] for row in group])
        baseline = grouped["dot"]
        paired = [row["ndcg@10"] - next(base["ndcg@10"] for base in baseline if base["seed"] == row["seed"]) for row in group]
        d_mean, d_sd = mean_sd(paired)
        summary.append({"variant": variant, "recall@10_mean": r_mean, "recall@10_sd": r_sd, "ndcg@10_mean": n_mean, "ndcg@10_sd": n_sd, "delta_ndcg@10_vs_dot_mean": d_mean, "delta_ndcg@10_vs_dot_sd": d_sd})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# BSARec sequence-normalization mechanism confirmation", "", f"`tau={args.temperature:g}` for sequence and stop-gradient sequence was selected by independent seed-2026 validation. Scaled-dot uses its separately selected `tau=0.04`.", "", "| variant | Recall@10 | NDCG@10 | paired delta NDCG@10 vs. dot |", "| --- | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['recall@10_mean']:.4f} ± {row['recall@10_sd']:.4f} | {row['ndcg@10_mean']:.4f} ± {row['ndcg@10_sd']:.4f} | {row['delta_ndcg@10_vs_dot_mean']:+.4f} ± {row['delta_ndcg@10_vs_dot_sd']:.4f} |")
    lines.extend(["", "`sequence` and `stopgrad_sequence` have exactly the same forward scoring function at a fixed parameter state; only the sequence-norm denominator is detached in the latter's backward pass."])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'per_seed.csv'}, summary.csv and summary.md")


if __name__ == "__main__":
    main()
