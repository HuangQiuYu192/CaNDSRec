#!/usr/bin/env python3
"""Collect Stage-3 grouped test metrics and deltas versus the no-augmentation run."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).glob("group_*.csv"))
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No group_*.csv files found; training/evaluation may not have completed.")
    baseline = {row["group"]: row for row in rows if row["tag"] == "baseline"}
    if len(baseline) != 4:
        raise ValueError("Expected four baseline group rows before computing deltas.")
    for row in rows:
        reference = baseline[row["group"]]
        for metric in ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "recall@100", "ndcg@100"]:
            row[f"delta_{metric}_vs_baseline"] = float(row[metric]) - float(reference[metric])
    output = Path(args.out_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tag", "model", "group", "n", "recall@10", "ndcg@10", "recall@50", "ndcg@50", "recall@100", "ndcg@100", "delta_recall@10_vs_baseline", "delta_ndcg@10_vs_baseline", "delta_recall@50_vs_baseline", "delta_ndcg@50_vs_baseline", "delta_recall@100_vs_baseline", "delta_ndcg@100_vs_baseline"]
    with Path(f"{output}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Stage-3 TailSim auxiliary-training comparison", "", "All deltas use the fresh same-seed `baseline` run, not an older checkpoint.", "", "| variant | group | n | Recall@10 | Δ Recall@10 | NDCG@10 | Δ NDCG@10 | Recall@50 | Δ Recall@50 | NDCG@50 | Δ NDCG@50 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| {tag} | {group} | {n} | {r10:.4f} | {dr10:+.4f} | {n10:.4f} | {dn10:+.4f} | {r50:.4f} | {dr50:+.4f} | {n50:.4f} | {dn50:+.4f} |".format(tag=row["tag"], group=row["group"], n=row["n"], r10=float(row["recall@10"]), dr10=float(row["delta_recall@10_vs_baseline"]), n10=float(row["ndcg@10"]), dn10=float(row["delta_ndcg@10_vs_baseline"]), r50=float(row["recall@50"]), dr50=float(row["delta_recall@50_vs_baseline"]), n50=float(row["ndcg@50"]), dn50=float(row["delta_ndcg@50_vs_baseline"])))
    lines += ["", "## Reading rule", "", "The strict variant is the proposed method. It replays only real, leave-one-out Critic-recovered tail targets; it does not label an unverified synthetic candidate as positive. `random` and `raw` receive the same low auxiliary-loss weight and matched replay volume, isolating whether the gain comes from high-confidence selection rather than merely upweighting tail interactions.", ""]
    Path(f"{output}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}.csv and {output}.md")


if __name__ == "__main__":
    main()
