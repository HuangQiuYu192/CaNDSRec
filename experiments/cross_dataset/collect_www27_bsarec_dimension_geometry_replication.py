"""Aggregate frozen BSARec dimension--geometry diagnostics across seeds."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = (
    "ndcg10_alpha_0", "ndcg10_alpha_1", "item_log_norm_std",
    "top10_jaccard_dot_vs_angular", "local_inversion_rate", "angular_margin",
)


def source_path(model: str, dim: int, seed: int) -> Path:
    name = f"{model}_d{dim}_seed{seed}.csv"
    if seed == 2025:
        return ROOT / "analysis_results/www27_bsarec_dimension_geometry" / name
    return ROOT / "analysis_results/www27_bsarec_dimension_geometry_replication" / name


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def main() -> None:
    rows = []
    for seed in (2023, 2024, 2025):
        for dim in (64, 128, 256):
            for model in ("BSARec", "CANDSBSARec"):
                with source_path(model, dim, seed).open(newline="", encoding="utf-8") as f:
                    rows.extend(csv.DictReader(f))
    out = ROOT / "analysis_results/www27_bsarec_dimension_geometry_replication"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], int(row["dimension"]))].append(row)
    summary = []
    for (model, dim), group in sorted(groups.items()):
        row = {"model": model, "dimension": dim}
        for key in CORE:
            values = [float(item[key]) for item in group]
            row[f"{key}_mean"] = mean(values); row[f"{key}_sd"] = sd(values)
        row["radial_drop_mean"] = row["ndcg10_alpha_0_mean"] - row["ndcg10_alpha_1_mean"]
        summary.append(row)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with (out / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# BSARec dimension--geometry replication on Beauty\n\n")
        f.write("Frozen checkpoints from three matched seeds. No diagnostic is used for model selection.\n\n")
        f.write("| model | d | angular NDCG@10 | radial drop | item log-norm std. | Top-10 Jaccard | local inversion | angular margin |\n")
        f.write("| --- | --: | --: | --: | --: | --: | --: | --: |\n")
        for row in summary:
            f.write(f"| {row['model']} | {row['dimension']} | {row['ndcg10_alpha_0_mean']:.4f} ± {row['ndcg10_alpha_0_sd']:.4f} | "
                    f"{row['radial_drop_mean']:+.4f} | {row['item_log_norm_std_mean']:.4f} | "
                    f"{row['top10_jaccard_dot_vs_angular_mean']:.4f} | {row['local_inversion_rate_mean']:.4f} | "
                    f"{row['angular_margin_mean']:+.4f} |\n")
    print(f"wrote {out / 'per_seed.csv'}, summary.csv and summary.md")


if __name__ == "__main__":
    main()
