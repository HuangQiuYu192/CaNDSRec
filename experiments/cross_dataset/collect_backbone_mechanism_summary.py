#!/usr/bin/env python3
"""Summarize backbone mechanism CSVs into compact paper-friendly tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_SPECS = {
    "SASRec": ("SASRec", "CANDSSASRec", "analysis_results/beauty_sasrec_mechanism"),
    "WEARec": ("WEARec", "CANDSWEARec", "analysis_results/beauty_wearec_mechanism"),
    "FMLPRec": ("FMLPRec", "CANDSFMLPRec", "analysis_results/beauty_fmlprec_mechanism"),
}
FILENAME_RE = re.compile(r"(?P<dataset>.+)_(?P<base>SASRec|WEARec|FMLPRec)_vs_CANDS_h(?P<hidden>\d+)_len(?P<max_len>\d+)_temp(?P<temp>.+)\.csv$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float:
    if value in {None, ""}:
        return math.nan
    return float(value)


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def parse_specs(specs: list[str]) -> dict[str, tuple[str, str, str]]:
    if not specs:
        return DEFAULT_SPECS
    output = {}
    for spec in specs:
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise ValueError("--mechanism_dir entries must be backbone:base_model:cands_model:dir")
        output[parts[0]] = (parts[1], parts[2], parts[3])
    return output


def find_row(rows: list[dict[str, str]], model: str, group: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("model") == model and row.get("group") == group:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism_dir",
        action="append",
        default=[],
        help="Optional backbone:base_model:cands_model:dir. Can be repeated.",
    )
    parser.add_argument("--out_dir", default="analysis_results/backbone_mechanism_summary")
    args = parser.parse_args()

    specs = parse_specs(args.mechanism_dir)
    rows_out = []
    missing = []
    for backbone, (base_model, cands_model, directory) in specs.items():
        root = Path(directory)
        if not root.exists():
            missing.append(f"{backbone}: missing dir {directory}")
            continue
        for path in sorted(root.glob("*.csv")):
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            info = match.groupdict()
            hidden = int(info["hidden"])
            max_len = int(info["max_len"])
            temp = as_float(info["temp"])
            rows = read_csv(path)
            delta_model = f"{cands_model}_minus_{base_model}"
            for group in ["all", "head", "mid", "tail"]:
                base = find_row(rows, base_model, group)
                cands = find_row(rows, cands_model, group)
                delta = find_row(rows, delta_model, group)
                if base is None or cands is None or delta is None:
                    missing.append(f"{backbone}: missing {group} rows in {path}")
                    continue
                row = {
                    "dataset": info["dataset"],
                    "backbone": backbone,
                    "hidden": hidden,
                    "max_len": max_len,
                    "temperature": temp,
                    "group": group,
                    "n": int(float(base["n"])),
                    "base_recall@10": as_float(base.get("rank_recall@10")),
                    "cands_recall@10": as_float(cands.get("rank_recall@10")),
                    "base_ndcg@10": as_float(base.get("rank_ndcg@10")),
                    "cands_ndcg@10": as_float(cands.get("rank_ndcg@10")),
                    "base_pos_cos_mean": as_float(base.get("pos_cos_mean")),
                    "cands_pos_cos_mean": as_float(cands.get("pos_cos_mean")),
                    "base_hard_neg_cos_mean": as_float(base.get("hard_neg_cos_mean")),
                    "cands_hard_neg_cos_mean": as_float(cands.get("hard_neg_cos_mean")),
                    "base_cos_margin_mean": as_float(base.get("cos_margin_mean")),
                    "cands_cos_margin_mean": as_float(cands.get("cos_margin_mean")),
                    "base_seq_norm_mean": as_float(base.get("seq_norm_mean")),
                    "cands_seq_norm_mean": as_float(cands.get("seq_norm_mean")),
                    "base_target_item_norm_mean": as_float(base.get("target_item_norm_mean")),
                    "cands_target_item_norm_mean": as_float(cands.get("target_item_norm_mean")),
                    "base_item_norm_spearman_pop": as_float(base.get("item_norm_spearman_pop")),
                    "cands_item_norm_spearman_pop": as_float(cands.get("item_norm_spearman_pop")),
                    "mean_delta_rank": as_float(delta.get("mean_delta_rank")),
                    "median_delta_rank": as_float(delta.get("median_delta_rank")),
                    "improved_pct": as_float(delta.get("improved_pct")),
                    "worse_pct": as_float(delta.get("worse_pct")),
                    "hit10_gain": int(float(delta.get("hit10_gain", 0) or 0)),
                    "hit10_loss": int(float(delta.get("hit10_loss", 0) or 0)),
                    "hit20_gain": int(float(delta.get("hit20_gain", 0) or 0)),
                    "hit20_loss": int(float(delta.get("hit20_loss", 0) or 0)),
                }
                row["delta_recall@10"] = row["cands_recall@10"] - row["base_recall@10"]
                row["delta_ndcg@10"] = row["cands_ndcg@10"] - row["base_ndcg@10"]
                row["rel_recall@10"] = (
                    row["cands_recall@10"] / row["base_recall@10"] - 1
                    if row["base_recall@10"] > 0
                    else math.nan
                )
                row["rel_ndcg@10"] = (
                    row["cands_ndcg@10"] / row["base_ndcg@10"] - 1
                    if row["base_ndcg@10"] > 0
                    else math.nan
                )
                rows_out.append(row)

    rows_out.sort(key=lambda row: (row["backbone"], row["hidden"], row["group"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"
    out_md = out_dir / "summary.md"
    missing_path = out_dir / "missing.txt"

    headers = list(rows_out[0].keys()) if rows_out else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_out)

    md_headers = [
        "backbone",
        "hidden",
        "group",
        "base_recall@10",
        "cands_recall@10",
        "rel_recall@10",
        "base_ndcg@10",
        "cands_ndcg@10",
        "rel_ndcg@10",
        "improved_pct",
        "worse_pct",
        "hit10_gain",
        "hit10_loss",
        "mean_delta_rank",
        "base_item_norm_spearman_pop",
        "cands_item_norm_spearman_pop",
    ]
    lines = [
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join(["---"] * len(md_headers)) + " |",
    ]
    for row in rows_out:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in md_headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"wrote {len(rows_out)} rows to {out_csv} and {out_md}")
    if missing:
        print(f"missing {len(missing)} entries; see {missing_path}")


if __name__ == "__main__":
    main()
