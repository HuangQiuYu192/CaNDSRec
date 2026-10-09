"""Aggregate three-seed sequence-scale quintile diagnostics."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("dot", "scaled_dot", "sequence", "stopgrad_sequence", "joint")
STRATA = ("Q1", "Q2", "Q3", "Q4", "Q5")
METRICS = ("raw_sequence_norm_mean", "logit_std_mean", "entropy_mean", "target_cross_entropy_mean", "recall@10", "ndcg@10")


def main() -> None:
    out = ROOT / "analysis_results/www27_bsarec_sequence_scale_strata"
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in out.glob("*_seed*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["variant"], row["stratum"])].append(row)
    rows = []
    for variant in VARIANTS:
        for stratum in STRATA:
            values = grouped[(variant, stratum)]
            if len(values) != 3:
                raise SystemExit(f"Expected three seeds for {variant}/{stratum}, found {len(values)}")
            row = {"variant": variant, "stratum": stratum, "n_seeds": len(values)}
            for metric in METRICS:
                row[metric] = sum(float(value[metric]) for value in values) / len(values)
            rows.append(row)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# BSARec sequence-scale quintile diagnostics", "", "Each checkpoint's test examples are partitioned into quintiles by its encoder's raw sequence norm. Values are means over the three fixed seeds; this is a read-only checkpoint analysis.", ""]
    for variant in VARIANTS:
        lines.extend([f"## {variant}", "", "| raw-norm stratum | raw sequence norm | logit std. | entropy | target CE | Recall@10 | NDCG@10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in (item for item in rows if item["variant"] == variant):
            lines.append(f"| {row['stratum']} | {row['raw_sequence_norm_mean']:.3f} | {row['logit_std_mean']:.3f} | {row['entropy_mean']:.3f} | {row['target_cross_entropy_mean']:.3f} | {row['recall@10']:.4f} | {row['ndcg@10']:.4f} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'summary.csv'} and summary.md")


if __name__ == "__main__":
    main()
