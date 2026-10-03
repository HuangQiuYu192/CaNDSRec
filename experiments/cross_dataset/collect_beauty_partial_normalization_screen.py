#!/usr/bin/env python3
"""Summarise the fixed-temperature partial-normalisation screen on Beauty.

The source runs use one development seed and a common temperature (tau=10 for
normalised variants).  This deliberately is a mechanism screen, not a
multi-seed significance claim: it asks whether retaining either radial degree
of freedom is sufficient before spending compute on replication.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR_RE = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
RESULT_RE = re.compile(r"(best valid result:|test result:)\s*OrderedDict\(\[(.*?)\]\)", re.S)

VARIANTS = [
    ("dot", "dot", 0.0, 0.0, 1.0),
    ("sequence", "sequence", 1.0, 0.0, 10.0),
    ("item", "item", 0.0, 1.0, 10.0),
    ("both", "both", 1.0, 1.0, 10.0),
    ("partial_05_05", "partial", 0.5, 0.5, 10.0),
    ("partial_10_05", "partial", 1.0, 0.5, 10.0),
    ("partial_05_10", "partial", 0.5, 1.0, 10.0),
]


def parse_results(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    matches = RESULT_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    found = {kind: {key: float(value) for key, value in PAIR_RE.findall(payload)} for kind, payload in matches}
    if "best valid result:" not in found or "test result:" not in found:
        raise ValueError(f"Missing result block in {path}")
    return found["best valid result:"], found["test result:"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("log_runs/beauty_radial_power_grid_gpu12"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("analysis_results/beauty_partial_normalization_screen/summary.csv"))
    args = parser.parse_args()

    rows = []
    for tag, geometry, sequence_power, item_power, temperature in VARIANTS:
        filename = f"{tag}_tau{int(temperature) if temperature.is_integer() else temperature}.log"
        valid, test = parse_results(args.log_root / f"seed{args.seed}" / filename)
        rows.append({
            "tag": tag, "geometry": geometry, "sequence_norm_power": sequence_power,
            "item_norm_power": item_power, "temperature": temperature,
            "valid_recall@10": valid["recall@10"], "valid_ndcg@10": valid["ndcg@10"],
            "test_recall@10": test["recall@10"], "test_ndcg@10": test["ndcg@10"],
            "test_recall@20": test["recall@20"], "test_ndcg@20": test["ndcg@20"],
        })

    baseline = next(row for row in rows if row["tag"] == "dot")
    for row in rows:
        row["delta_test_recall@10_vs_dot"] = row["test_recall@10"] - baseline["test_recall@10"]
        row["delta_test_ndcg@10_vs_dot"] = row["test_ndcg@10"] - baseline["test_ndcg@10"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Beauty partial-normalisation mechanism screen", "",
        "All normalised variants use the same development seed and tau=10; dot uses its unscaled score. "
        "It is a fixed-configuration mechanism screen, not a multi-seed comparison.", "",
        "| score geometry | powers `(p_h, p_e)` | tau | Test Recall@10 | delta vs dot | Test NDCG@10 | delta vs dot |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['tag']} | ({row['sequence_norm_power']:.1f}, {row['item_norm_power']:.1f}) | "
            f"{row['temperature']:.0f} | {row['test_recall@10']:.4f} | "
            f"{row['delta_test_recall@10_vs_dot']:+.4f} | {row['test_ndcg@10']:.4f} | "
            f"{row['delta_test_ndcg@10_vs_dot']:+.4f} |"
        )
    lines.extend([
        "", "## Reading rule", "",
        "`p_h` and `p_e` are the sequence- and item-side radial-removal powers. "
        "The screen tests whether either one-sided removal, or a partial removal, reproduces the joint endpoint. "
        "Only the selected variants should proceed to independent-seed confirmation.",
    ])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
