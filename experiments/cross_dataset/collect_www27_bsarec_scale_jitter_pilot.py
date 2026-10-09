"""Collect the development-only BSARec positive-scale-jitter intervention."""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse(path: Path, marker: str) -> dict[str, float]:
    found = re.findall(rf"{marker}:\s*OrderedDict\((\[.*?\])\)", path.read_text(encoding="utf-8", errors="replace"))
    if not found:
        raise ValueError(f"Missing {marker} in {path}")
    return dict(ast.literal_eval(found[-1]))


def main() -> None:
    logs = ROOT / "log_runs/www27_bsarec_scale_jitter_pilot/Beauty/seed2026"
    rows = []
    for path in logs.glob("*_sigma*_d256_a0.5_c9_h2_lr0.001.log"):
        match = re.match(r"(dot|sequence)_sigma([^_]+)_", path.name)
        valid, test = parse(path, "best valid result"), parse(path, "test result")
        rows.append({"variant": match.group(1), "sigma": float(match.group(2)), "valid_ndcg@10": valid["ndcg@10"], "valid_recall@10": valid["recall@10"], "development_test_ndcg@10": test["ndcg@10"], "development_test_recall@10": test["recall@10"]})
    rows.sort(key=lambda row: (row["variant"], row["sigma"]))
    out = ROOT / "analysis_results/www27_bsarec_scale_jitter_pilot"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# BSARec positive sequence-scale jitter pilot", "", "Development seed 2026 only. Test values are diagnostic and must not be used for model or jitter-strength selection.", "", "| variant | sigma | Validation NDCG@10 | Validation Recall@10 | Development-test NDCG@10 | Development-test Recall@10 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {row['variant']} | {row['sigma']:.2f} | {row['valid_ndcg@10']:.4f} | {row['valid_recall@10']:.4f} | {row['development_test_ndcg@10']:.4f} | {row['development_test_recall@10']:.4f} |" for row in rows)
    lines.append("\nFor `sequence`, positive jitter is algebraically cancelled by sequence normalization; any large difference is a diagnostic failure.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.csv'} and summary.md")


if __name__ == "__main__":
    main()
