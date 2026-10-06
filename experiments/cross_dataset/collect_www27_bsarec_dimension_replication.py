"""Collect the pre-specified BSARec capacity--geometry replication."""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


MODELS = ("BSARec", "CANDSBSARec")
DIMS = (64, 128, 256)
SEEDS = (2023, 2024, 2025)


def tag(model: str, dim: int) -> str:
    return f"{model}_d{dim}_a0.5_c9_h2_lr0.001"


def log_path(root: Path, seed: int, model: str, dim: int) -> Path:
    if seed == 2025:
        return root / "log_runs/www27_bsarec_dimension_validation/Beauty/seed2025" / f"{tag(model, dim)}.log"
    if dim == 64:
        return root / f"log_runs/www27_bsarec_geometry_replication/Beauty/seed{seed}" / f"{tag(model, dim)}.log"
    return root / f"log_runs/www27_bsarec_dimension_replication/Beauty/seed{seed}" / f"{tag(model, dim)}.log"


def read_test(log: Path) -> dict[str, float]:
    text = log.read_text(encoding="utf-8", errors="replace")
    records = re.findall(r"test result:\s*OrderedDict\((\[.*?\])\)", text)
    if not records:
        raise ValueError(f"No completed test result in {log}")
    return dict(ast.literal_eval(records[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir or root / "analysis_results/www27_bsarec_dimension_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dim in DIMS:
        for seed in SEEDS:
            result = {model: read_test(log_path(root, seed, model, dim)) for model in MODELS}
            base, joint = result["BSARec"], result["CANDSBSARec"]
            rows.append({
                "dimension": dim, "seed": seed,
                "base_ndcg@10": base["ndcg@10"], "joint_ndcg@10": joint["ndcg@10"],
                "delta_ndcg@10": joint["ndcg@10"] - base["ndcg@10"],
                "base_recall@10": base["recall@10"], "joint_recall@10": joint["recall@10"],
                "delta_recall@10": joint["recall@10"] - base["recall@10"],
            })
    with (out_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["dimension"]].append(row)
    summary = []
    for dim in DIMS:
        group = grouped[dim]
        def mean(key): return sum(float(r[key]) for r in group) / len(group)
        def sd(key):
            mu = mean(key)
            return math.sqrt(sum((float(r[key]) - mu) ** 2 for r in group) / len(group))
        summary.append({
            "dimension": dim, "base_ndcg@10_mean": mean("base_ndcg@10"), "base_ndcg@10_sd": sd("base_ndcg@10"),
            "joint_ndcg@10_mean": mean("joint_ndcg@10"), "joint_ndcg@10_sd": sd("joint_ndcg@10"),
            "delta_ndcg@10_mean": mean("delta_ndcg@10"), "delta_ndcg@10_sd": sd("delta_ndcg@10"),
            "base_recall@10_mean": mean("base_recall@10"), "joint_recall@10_mean": mean("joint_recall@10"),
            "delta_recall@10_mean": mean("delta_recall@10"),
        })
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# BSARec capacity--geometry replication on Beauty\n\n")
        f.write("All BSARec backbone settings are fixed before testing. `d=64` reuses the earlier completed replication; `d=128/256` are the pre-specified additional seeds.\n\n")
        f.write("| d | BSARec NDCG@10 | Joint NDCG@10 | delta | BSARec Recall@10 | Joint Recall@10 | delta |\n")
        f.write("| --: | --: | --: | --: | --: | --: | --: |\n")
        for row in summary:
            f.write(f"| {row['dimension']} | {row['base_ndcg@10_mean']:.4f} ± {row['base_ndcg@10_sd']:.4f} | "
                    f"{row['joint_ndcg@10_mean']:.4f} ± {row['joint_ndcg@10_sd']:.4f} | {row['delta_ndcg@10_mean']:+.4f} | "
                    f"{row['base_recall@10_mean']:.4f} | {row['joint_recall@10_mean']:.4f} | {row['delta_recall@10_mean']:+.4f} |\n")
        f.write("\nThe grid is a fixed-configuration capacity interaction test, not a per-dimension hyperparameter search.\n")
    print(f"wrote {out_dir / 'per_seed.csv'}, summary.csv and summary.md")


if __name__ == "__main__":
    main()
