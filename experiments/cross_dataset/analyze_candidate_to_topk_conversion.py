#!/usr/bin/env python3
"""Diagnose whether broad CaNDS candidate gains can convert to top-k hits.

This script operates on rank-transition ``*.samples.csv`` files. Since these
files contain target ranks rather than full candidate lists and scores, the
reported variants are diagnostic proxies:

1. candidate_oracle: if an eligible target is already within top-M under CaNDS,
   promote it to the top-k boundary. This estimates the upper-bound room for a
   second-stage reranker that can identify those candidates.
2. rank_shift: subtract a fixed rank budget from eligible target ranks.
3. rank_scale: multiply eligible target ranks by a factor smaller than 1.

The goal is not to claim a deployable reranking model, but to quantify whether
CaNDS creates enough tail candidates near the metric boundary to justify a
second-stage method.
"""

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


def parse_number_list(text: str, cast=float) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    ranks = ranks.astype(np.float64)
    if len(ranks) == 0:
        return math.nan, math.nan
    hit = ranks <= k
    return float(hit.mean()), float((hit / np.log2(ranks + 1.0)).mean())


def group_metrics(ranks_by_group: dict[str, np.ndarray], cutoffs: list[int]) -> dict[str, dict[str, float]]:
    metrics = {}
    for group in ["all", "head", "mid", "tail"]:
        ranks = ranks_by_group[group]
        row = {"n": int(len(ranks))}
        for k in cutoffs:
            recall, ndcg = metric_at(ranks, k)
            row[f"recall@{k}"] = recall
            row[f"ndcg@{k}"] = ndcg
        metrics[group] = row
    return metrics


def load_grouped_ranks(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    groups = {}
    for group in ["head", "mid", "tail"]:
        group_rows = [row for row in rows if row["group"] == group]
        groups[group] = np.asarray([int(row["cands_rank"]) for row in group_rows], dtype=np.int64)
    groups["all"] = np.concatenate([groups["head"], groups["mid"], groups["tail"]])
    return groups


def apply_candidate_oracle(
    ranks_by_group: dict[str, np.ndarray],
    eligible_groups: set[str],
    candidate_cutoff: int,
    target_cutoff: int,
    promoted_rank: int,
) -> dict[str, np.ndarray]:
    output = {group: ranks.copy() for group, ranks in ranks_by_group.items() if group != "all"}
    for group in eligible_groups:
        ranks = output[group]
        mask = (ranks > target_cutoff) & (ranks <= candidate_cutoff)
        ranks[mask] = promoted_rank
    output["all"] = np.concatenate([output["head"], output["mid"], output["tail"]])
    return output


def apply_rank_shift(
    ranks_by_group: dict[str, np.ndarray],
    eligible_groups: set[str],
    shift: int,
) -> dict[str, np.ndarray]:
    output = {group: ranks.copy() for group, ranks in ranks_by_group.items() if group != "all"}
    for group in eligible_groups:
        output[group] = np.maximum(1, output[group] - shift)
    output["all"] = np.concatenate([output["head"], output["mid"], output["tail"]])
    return output


def apply_rank_scale(
    ranks_by_group: dict[str, np.ndarray],
    eligible_groups: set[str],
    scale: float,
) -> dict[str, np.ndarray]:
    output = {group: ranks.copy() for group, ranks in ranks_by_group.items() if group != "all"}
    for group in eligible_groups:
        output[group] = np.maximum(1, np.ceil(output[group].astype(np.float64) * scale).astype(np.int64))
    output["all"] = np.concatenate([output["head"], output["mid"], output["tail"]])
    return output


def summarize_variant(
    meta: dict,
    variant: str,
    params: str,
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    cutoffs: list[int],
) -> list[dict]:
    rows = []
    for group in ["all", "head", "mid", "tail"]:
        row = {**meta, "variant": variant, "params": params, "group": group, "n": candidate[group]["n"]}
        for k in cutoffs:
            base_r = baseline[group][f"recall@{k}"]
            cand_r = candidate[group][f"recall@{k}"]
            base_n = baseline[group][f"ndcg@{k}"]
            cand_n = candidate[group][f"ndcg@{k}"]
            row[f"base_recall@{k}"] = base_r
            row[f"rerank_recall@{k}"] = cand_r
            row[f"delta_recall@{k}"] = cand_r - base_r
            row[f"rel_recall@{k}"] = cand_r / base_r - 1 if base_r > 0 else math.nan
            row[f"base_ndcg@{k}"] = base_n
            row[f"rerank_ndcg@{k}"] = cand_n
            row[f"delta_ndcg@{k}"] = cand_n - base_n
            row[f"rel_ndcg@{k}"] = cand_n / base_n - 1 if base_n > 0 else math.nan
            row[f"net_hit{k}"] = int(round((cand_r - base_r) * candidate[group]["n"]))
        rows.append(row)
    return rows


def analyze_one(
    path: Path,
    cutoffs: list[int],
    candidate_cutoffs: list[int],
    target_cutoff: int,
    promoted_rank: int,
    shifts: list[int],
    scales: list[float],
    eligible_group_sets: list[tuple[str, set[str]]],
) -> list[dict]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return []
    meta = match.groupdict()
    meta.update(
        {
            "hidden": int(meta["hidden"]),
            "max_len": int(meta["max_len"]),
            "temperature": float(meta.pop("temp")),
        }
    )
    ranks = load_grouped_ranks(path)
    baseline = group_metrics(ranks, cutoffs)
    rows = []

    for label, groups in eligible_group_sets:
        for candidate_cutoff in candidate_cutoffs:
            candidate = apply_candidate_oracle(
                ranks,
                groups,
                candidate_cutoff=candidate_cutoff,
                target_cutoff=target_cutoff,
                promoted_rank=promoted_rank,
            )
            metrics = group_metrics(candidate, cutoffs)
            params = f"eligible={label};candidate_cutoff={candidate_cutoff};target_cutoff={target_cutoff};promoted_rank={promoted_rank}"
            rows.extend(summarize_variant(meta, "candidate_oracle", params, baseline, metrics, cutoffs))

        for shift in shifts:
            candidate = apply_rank_shift(ranks, groups, shift=shift)
            metrics = group_metrics(candidate, cutoffs)
            params = f"eligible={label};shift={shift}"
            rows.extend(summarize_variant(meta, "rank_shift", params, baseline, metrics, cutoffs))

        for scale in scales:
            candidate = apply_rank_scale(ranks, groups, scale=scale)
            metrics = group_metrics(candidate, cutoffs)
            params = f"eligible={label};scale={scale}"
            rows.extend(summarize_variant(meta, "rank_scale", params, baseline, metrics, cutoffs))

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], primary_cutoff: int) -> None:
    headers = [
        "dataset",
        "backbone",
        "hidden",
        "variant",
        "params",
        "group",
        f"base_recall@{primary_cutoff}",
        f"rerank_recall@{primary_cutoff}",
        f"rel_recall@{primary_cutoff}",
        f"base_ndcg@{primary_cutoff}",
        f"rerank_ndcg@{primary_cutoff}",
        f"rel_ndcg@{primary_cutoff}",
        f"net_hit{primary_cutoff}",
    ]
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
    parser.add_argument("--out_dir", default="analysis_results/candidate_to_topk_conversion")
    parser.add_argument("--cutoffs", default="10,20,50,100")
    parser.add_argument("--candidate_cutoffs", default="20,50,100")
    parser.add_argument("--target_cutoff", default=10, type=int)
    parser.add_argument("--promoted_rank", default=10, type=int)
    parser.add_argument("--shifts", default="5,10,20,50,100")
    parser.add_argument("--scales", default="0.25,0.5,0.75")
    parser.add_argument(
        "--eligible_sets",
        default="tail:tail,mid_tail:mid+tail",
        help="Comma-separated label:group+group definitions, e.g. tail:tail,mid_tail:mid+tail.",
    )
    args = parser.parse_args()

    cutoffs = parse_number_list(args.cutoffs, int)
    candidate_cutoffs = parse_number_list(args.candidate_cutoffs, int)
    shifts = parse_number_list(args.shifts, int)
    scales = parse_number_list(args.scales, float)
    eligible_group_sets = []
    for spec in args.eligible_sets.split(","):
        if not spec.strip():
            continue
        label, groups = spec.split(":", 1)
        eligible_group_sets.append((label, set(groups.split("+"))))

    rows = []
    for path in sorted(Path(args.rank_transition_dir).glob("*.samples.csv")):
        rows.extend(
            analyze_one(
                path,
                cutoffs=cutoffs,
                candidate_cutoffs=candidate_cutoffs,
                target_cutoff=args.target_cutoff,
                promoted_rank=args.promoted_rank,
                shifts=shifts,
                scales=scales,
                eligible_group_sets=eligible_group_sets,
            )
        )
    rows.sort(key=lambda row: (row["backbone"], row["hidden"], row["variant"], row["params"], row["group"]))

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "summary.csv", rows)
    write_markdown(out_dir / "summary.md", rows, primary_cutoff=args.target_cutoff)
    print(f"wrote {len(rows)} rows to {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
