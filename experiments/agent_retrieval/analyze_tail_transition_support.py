#!/usr/bin/env python3
"""Stage-0 audit for tail-constrained user simulation.

The audit is read-only: it uses only the RecBole *training* split to measure
whether each tail item has enough genuine predecessor/context support to be a
safe target for future simulated continuations.  It does not generate any
pseudo-interaction or train a new model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import align_architecture_to_checkpoint
from experiments.cross_dataset.analyze_group_metrics import build_model


def item_popularity_groups(popularity: np.ndarray) -> dict[str, np.ndarray]:
    active = np.flatnonzero(popularity > 0)
    order = active[np.argsort(-popularity[active], kind="stable")]
    return dict(zip(["head", "mid", "tail"], np.array_split(order, 3)))


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(list(counter.values()), dtype=np.float64) / total
    return float(-(probabilities * np.log(probabilities)).sum())


def recent_predecessors(item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
    """Get the final real history item from RecBole's right-padded sequences."""
    lengths = item_seq_len.long().clamp_min(1)
    return item_seq[torch.arange(item_seq.size(0), device=item_seq.device), lengths - 1]


def collect_support(train_data, model, config, min_contexts: int, min_predecessors: int):
    device = next(model.parameters()).device
    inter = train_data.dataset.inter_feat
    item_field, user_field = config["ITEM_ID_FIELD"], config["USER_ID_FIELD"]
    targets = inter[item_field].cpu().numpy().astype(np.int64)
    users = inter[user_field].cpu().numpy().astype(np.int64)
    sequences = inter[model.ITEM_SEQ].to(device)
    lengths = inter[model.ITEM_SEQ_LEN].to(device)
    predecessors = recent_predecessors(sequences, lengths).cpu().numpy().astype(np.int64)
    item_num = model.n_items
    popularity = np.bincount(targets, minlength=item_num)

    predecessor_counts: dict[int, Counter] = defaultdict(Counter)
    user_sets: dict[int, set[int]] = defaultdict(set)
    target_contexts = Counter()
    for target, predecessor, user in zip(targets, predecessors, users):
        if target <= 0 or predecessor <= 0:
            continue
        target_contexts[int(target)] += 1
        predecessor_counts[int(target)][int(predecessor)] += 1
        user_sets[int(target)].add(int(user))

    with torch.no_grad():
        item_dir = model._normalized_item_embedding().detach().float().cpu().numpy()
    rows = []
    for item_id in np.flatnonzero(popularity > 0):
        contexts = int(target_contexts[item_id])
        predecessors_for_item = predecessor_counts[item_id]
        unique_predecessors = len(predecessors_for_item)
        mean_cosine = (
            float(np.average([np.dot(item_dir[item_id], item_dir[pred]) for pred in predecessors_for_item], weights=list(predecessors_for_item.values())))
            if predecessors_for_item else math.nan
        )
        eligible = contexts >= min_contexts and unique_predecessors >= min_predecessors
        rows.append({
            "item_id": int(item_id),
            "train_target_popularity": int(popularity[item_id]),
            "real_transition_contexts": contexts,
            "unique_predecessors": unique_predecessors,
            "unique_users": len(user_sets[item_id]),
            "predecessor_entropy": entropy(predecessors_for_item),
            "mean_predecessor_cosine": mean_cosine,
            "eligible_for_simulation": int(eligible),
        })
    groups = item_popularity_groups(popularity)
    group_of = {int(item): name for name, items in groups.items() for item in items}
    for row in rows:
        row["popularity_group"] = group_of[row["item_id"]]
    return popularity, rows


def mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else math.nan


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for group in ["head", "mid", "tail"]:
        subset = [row for row in rows if row["popularity_group"] == group]
        eligible = [row for row in subset if row["eligible_for_simulation"]]
        with_context = [row for row in subset if row["real_transition_contexts"] > 0]
        summary.append({
            "group": group,
            "items": len(subset),
            "items_with_real_context": len(with_context),
            "context_coverage": len(with_context) / max(len(subset), 1),
            "eligible_items": len(eligible),
            "eligible_fraction": len(eligible) / max(len(subset), 1),
            "mean_train_target_popularity": mean([row["train_target_popularity"] for row in subset]),
            "mean_real_transition_contexts": mean([row["real_transition_contexts"] for row in subset]),
            "mean_unique_predecessors": mean([row["unique_predecessors"] for row in subset]),
            "mean_unique_users": mean([row["unique_users"] for row in subset]),
            "mean_predecessor_entropy": mean([row["predecessor_entropy"] for row in subset]),
            "mean_predecessor_cosine": mean([row["mean_predecessor_cosine"] for row in subset]),
        })
    return summary


def write_outputs(out_prefix: Path, summary: list[dict], item_rows: list[dict], metadata: dict) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary[0])
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    item_fields = list(item_rows[0])
    with Path(f"{out_prefix}_tail_items.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=item_fields)
        writer.writeheader()
        writer.writerows(row for row in item_rows if row["popularity_group"] == "tail")
    lines = ["# Tail transition-support audit", "", "All statistics use only RecBole training-split sequence targets and their real immediate predecessors.", "", "| group | items | context coverage | eligible items | eligible fraction | mean contexts | mean unique predecessors | mean predecessor cosine |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['group']} | {row['items']} | {row['context_coverage']:.4f} | {row['eligible_items']} | {row['eligible_fraction']:.4f} | {row['mean_real_transition_contexts']:.2f} | {row['mean_unique_predecessors']:.2f} | {row['mean_predecessor_cosine']:.4f} |")
    lines += ["", "## Decision rule", "", f"A tail item is eligible for later simulation only if it has at least `{metadata['min_contexts']}` genuine training transition contexts and `{metadata['min_predecessors']}` distinct real predecessors. Items outside this whitelist must never receive invented pseudo-positives.", "", "The detailed tail whitelist/audit is in the accompanying `_tail_items.csv` file.", ""]
    Path(f"{out_prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--min_contexts", default=3, type=int)
    parser.add_argument("--min_predecessors", default=2, type=int)
    parser.add_argument("--out_prefix", default="analysis_results/agent_simulator/beauty_tail_transition_support/Beauty_h256_len50_temp10")
    args = parser.parse_args()
    align_architecture_to_checkpoint(args)
    config, _, train_data, _, _, model = build_model("CANDSSASRec", args.checkpoint, args)
    _, item_rows = collect_support(train_data, model, config, args.min_contexts, args.min_predecessors)
    summary = summarize(item_rows)
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "min_contexts": args.min_contexts, "min_predecessors": args.min_predecessors, "tail_items": int(sum(row["popularity_group"] == "tail" for row in item_rows)), "eligible_tail_items": int(sum(row["popularity_group"] == "tail" and row["eligible_for_simulation"] for row in item_rows))}
    write_outputs(Path(args.out_prefix), summary, item_rows, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix}_summary.csv, .md, _tail_items.csv and _meta.json")


if __name__ == "__main__":
    main()
