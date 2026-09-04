#!/usr/bin/env python3
"""Collect grouped Recall/NDCG at wider cutoffs from rank-transition samples."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(
    r"(?P<dataset>.+)_(?P<backbone>SASRec|WEARec|FMLPRec)_h(?P<hidden>\d+)_len(?P<max_len>\d+)_temp(?P<temp>.+)\.samples\.csv$"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    ranks = ranks.astype(np.float64)
    if len(ranks) == 0:
        return math.nan, math.nan
    hit = ranks <= k
    recall = float(hit.mean())
    ndcg = float((hit / np.log2(ranks + 1.0)).mean())
    return recall, ndcg


def transition_at(base_rank: np.ndarray, cands_rank: np.ndarray, k: int) -> dict[str, float | int]:
    base_hit = base_rank <= k
    cands_hit = cands_rank <= k
    gain = (~base_hit) & cands_hit
    loss = base_hit & (~cands_hit)
    stay = base_hit & cands_hit
    miss_stay = (~base_hit) & (~cands_hit)
    delta = base_rank - cands_rank
    miss_improved = miss_stay & (delta > 0)
    miss_worse = miss_stay & (delta < 0)
    return {
        f"hit{k}_stay": int(stay.sum()),
        f"hit{k}_gain": int(gain.sum()),
        f"hit{k}_loss": int(loss.sum()),
        f"net_hit{k}": int(gain.sum() - loss.sum()),
        f"miss{k}_stay": int(miss_stay.sum()),
        f"miss{k}_improved": int(miss_improved.sum()),
        f"miss{k}_worse": int(miss_worse.sum()),
        f"miss{k}_improved_pct": float(miss_improved.sum() / miss_stay.sum()) if miss_stay.sum() else math.nan,
    }


def summarize_one(path: Path, cutoffs: list[int]) -> list[dict]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return []
    info = match.groupdict()
    rows = read_csv(path)
    output = []
    for group in ["all", "head", "mid", "tail"]:
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        base_rank = np.asarray([int(row["base_rank"]) for row in group_rows], dtype=np.int64)
        cands_rank = np.asarray([int(row["cands_rank"]) for row in group_rows], dtype=np.int64)
        delta_rank = base_rank - cands_rank
        out = {
            "dataset": info["dataset"],
            "backbone": info["backbone"],
            "hidden": int(info["hidden"]),
            "max_len": int(info["max_len"]),
            "temperature": float(info["temp"]),
            "group": group,
            "n": int(len(group_rows)),
            "delta_rank_mean": float(delta_rank.mean()),
            "delta_rank_median": float(np.median(delta_rank)),
            "improved_pct": float((delta_rank > 0).mean()),
            "worse_pct": float((delta_rank < 0).mean()),
        }
        for k in cutoffs:
            base_recall, base_ndcg = metric_at(base_rank, k)
            cands_recall, cands_ndcg = metric_at(cands_rank, k)
            out[f"base_recall@{k}"] = base_recall
            out[f"cands_recall@{k}"] = cands_recall
            out[f"delta_recall@{k}"] = cands_recall - base_recall
            out[f"rel_recall@{k}"] = cands_recall / base_recall - 1 if base_recall > 0 else math.nan
            out[f"base_ndcg@{k}"] = base_ndcg
            out[f"cands_ndcg@{k}"] = cands_ndcg
            out[f"delta_ndcg@{k}"] = cands_ndcg - base_ndcg
            out[f"rel_ndcg@{k}"] = cands_ndcg / base_ndcg - 1 if base_ndcg > 0 else math.nan
            out.update(transition_at(base_rank, cands_rank, k))
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], cutoffs: list[int]) -> None:
    headers = ["backbone", "hidden", "group", "n", "improved_pct", "delta_rank_median"]
    for k in cutoffs:
        headers.extend(
            [
                f"base_recall@{k}",
                f"cands_recall@{k}",
                f"rel_recall@{k}",
                f"base_ndcg@{k}",
                f"cands_ndcg@{k}",
                f"rel_ndcg@{k}",
                f"net_hit{k}",
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_transition_dir", default="analysis_results/beauty_rank_transition")
    parser.add_argument("--out_dir", default="analysis_results/rank_cutoff_metrics")
    parser.add_argument("--cutoffs", default="5,10,20,50,100")
    args = parser.parse_args()

    cutoffs = [int(value) for value in args.cutoffs.split(",") if value.strip()]
    rows = []
    for path in sorted(Path(args.rank_transition_dir).glob("*.samples.csv")):
        rows.extend(summarize_one(path, cutoffs))
    rows.sort(key=lambda row: (row["backbone"], row["hidden"], row["group"]))

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "summary.csv", rows)
    write_markdown(out_dir / "summary.md", rows, cutoffs)
    print(f"wrote {len(rows)} rows to {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
