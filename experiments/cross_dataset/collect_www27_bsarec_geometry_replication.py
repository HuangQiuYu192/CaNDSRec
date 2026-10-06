"""Collect the preregistered BSARec score-head replication.

The 2025 row is the original d=64 capacity-validation run.  The 2023/2024
rows are produced by ``run_www27_bsarec_geometry_replication_gpu1.sh`` with
the same validation-selected backbone configuration.  The collector never
uses test values to choose a configuration.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


TAG = "{model}_d64_a0.5_c9_h2_lr0.001"


def parse_result(log: Path) -> dict[str, float]:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"test result:\s*OrderedDict\((\[.*?\])\)", text)
    if not matches:
        raise ValueError(f"No completed test result in {log}")
    return dict(ast.literal_eval(matches[-1]))


def locate(root: Path, seed: int, model: str) -> Path:
    if seed == 2025:
        return root / "log_runs" / "www27_bsarec_dimension_validation" / "Beauty" / "seed2025" / f"{TAG.format(model=model)}.log"
    return root / "log_runs" / "www27_bsarec_geometry_replication" / "Beauty" / f"seed{seed}" / f"{TAG.format(model=model)}.log"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir or root / "analysis_results" / "www27_bsarec_geometry_replication"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in (2023, 2024, 2025):
        base = parse_result(locate(root, seed, "BSARec"))
        joint = parse_result(locate(root, seed, "CANDSBSARec"))
        rows.append({
            "dataset": "Beauty", "seed": seed,
            "base_ndcg@10": base["ndcg@10"], "joint_ndcg@10": joint["ndcg@10"],
            "delta_ndcg@10": joint["ndcg@10"] - base["ndcg@10"],
            "base_recall@10": base["recall@10"], "joint_recall@10": joint["recall@10"],
            "delta_recall@10": joint["recall@10"] - base["recall@10"],
        })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean_delta_ndcg = sum(row["delta_ndcg@10"] for row in rows) / len(rows)
    mean_delta_recall = sum(row["delta_recall@10"] for row in rows) / len(rows)
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# BSARec score-geometry replication on Beauty\n\n")
        f.write("The d=64 BSARec backbone was selected using validation only. "
                "All rows use the same architecture and training settings; `CANDSBSARec` changes only the final head.\n\n")
        f.write("| seed | BSARec NDCG@10 | Joint NDCG@10 | delta | BSARec Recall@10 | Joint Recall@10 | delta |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write("| {seed} | {base_ndcg@10:.4f} | {joint_ndcg@10:.4f} | {delta_ndcg@10:+.4f} | "
                    "{base_recall@10:.4f} | {joint_recall@10:.4f} | {delta_recall@10:+.4f} |\n".format(**row))
        f.write(f"\nMean delta: NDCG@10={mean_delta_ndcg:+.4f}; Recall@10={mean_delta_recall:+.4f}.\n")
        f.write("This is an external encoder-family check, not a temperature or capacity search.\n")
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
