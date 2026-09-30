#!/usr/bin/env python3
"""Compare dot-to-geometry rank transitions on Beauty without retraining."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analyze_beauty_geometry_mechanisms import build_model, masked, popularity_groups


BINS = ("1-5", "6-10", "11-20", "21-50", "51-100", ">100")
GROUP_ORDER = ("all", "head", "mid", "tail")
TAG_ORDER = ("sequence", "both")


def rank_bin(ranks: np.ndarray) -> np.ndarray:
    return np.select(
        [ranks <= 5, ranks <= 10, ranks <= 20, ranks <= 50, ranks <= 100], BINS[:-1], default=BINS[-1]
    )


def load_ranked(checkpoint: str, model_name: str, geometry: str, temperature: float,
                sequence_norm_power: float, item_norm_power: float, seed: int, gpu_id: int,
                eval_batch_size: int):
    args = SimpleNamespace(
        checkpoint=checkpoint, model=model_name, geometry=geometry, temperature=temperature,
        sequence_norm_power=sequence_norm_power, item_norm_power=item_norm_power, seed=seed,
        gpu_id=gpu_id, eval_batch_size=eval_batch_size,
    )
    config, dataset, train_data, test_data, model = build_model(args)
    device = config["device"]
    targets, ranks = [], []
    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            scores = model.full_sort_predict(interaction)
            if scores.dim() == 1:
                scores = scores.view(positive_i.size(0), -1)
            scores = masked(scores, history_index)
            positive_score = scores[positive_u, positive_i]
            rank = (scores[positive_u] > positive_score.unsqueeze(1)).sum(dim=1) + 1
            targets.extend(positive_i.cpu().tolist())
            ranks.extend(rank.cpu().tolist())
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    return np.asarray(targets, dtype=np.int64), np.asarray(ranks, dtype=np.int64), popularity


def rate(condition: np.ndarray) -> float:
    return float(condition.mean()) if len(condition) else math.nan


def transition_rows(seed: int, tag: str, targets: np.ndarray, dot: np.ndarray, candidate: np.ndarray,
                    popularity: np.ndarray) -> tuple[list[dict], list[dict]]:
    summary, flows = [], []
    source_bins, target_bins = rank_bin(dot), rank_bin(candidate)
    for group, index in popularity_groups(targets, popularity).items():
        source, destination = dot[index], candidate[index]
        difference = destination.astype(np.int64) - source.astype(np.int64)
        row = {
            "seed": seed, "tag": tag, "group": group, "n": int(len(index)),
            "dot_recall@10": rate(source <= 10), "candidate_recall@10": rate(destination <= 10),
            "delta_recall@10": rate(destination <= 10) - rate(source <= 10),
            "dot_recall@50": rate(source <= 50), "candidate_recall@50": rate(destination <= 50),
            "delta_recall@50": rate(destination <= 50) - rate(source <= 50),
            "mean_rank_delta": float(difference.mean()), "median_rank_delta": float(np.median(difference)),
            "improved_rate": rate(difference < 0), "worsened_rate": rate(difference > 0), "unchanged_rate": rate(difference == 0),
            "enter_top10_rate": rate((source > 10) & (destination <= 10)),
            "leave_top10_rate": rate((source <= 10) & (destination > 10)),
            "deep_to_11_50_rate": rate((source > 50) & (destination >= 11) & (destination <= 50)),
            "top10_to_11_50_rate": rate((source <= 10) & (destination >= 11) & (destination <= 50)),
        }
        summary.append(row)
        for source_bin in BINS:
            for target_bin in BINS:
                count = int(((source_bins[index] == source_bin) & (target_bins[index] == target_bin)).sum())
                if count:
                    flows.append({"seed": seed, "tag": tag, "group": group, "from_bin": source_bin,
                                  "to_bin": target_bin, "n": count, "rate_within_group": count / len(index)})
    return summary, flows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu_id", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--dot_checkpoint", required=True)
    parser.add_argument("--sequence_checkpoint", required=True)
    parser.add_argument("--both_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    targets, dot, popularity = load_ranked(args.dot_checkpoint, "SASRec", "dot", 1, 0, 0, args.seed, args.gpu_id, args.eval_batch_size)
    all_summary, all_flows = [], []
    for tag, checkpoint, geometry, temperature, seq_power, item_power in (
        ("sequence", args.sequence_checkpoint, "sequence", 4, 1, 0),
        ("both", args.both_checkpoint, "both", 10, 1, 1),
    ):
        candidate_targets, candidate_ranks, _ = load_ranked(checkpoint, "GeometrySASRec", geometry, temperature,
                                                              seq_power, item_power, args.seed, args.gpu_id, args.eval_batch_size)
        if not np.array_equal(targets, candidate_targets):
            raise RuntimeError(f"Test target order differs for seed={args.seed}, tag={tag}")
        summary, flows = transition_rows(args.seed, tag, targets, dot, candidate_ranks, popularity)
        all_summary.extend(summary); all_flows.extend(flows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "flows.csv", all_flows)
    print(f"wrote {args.out_dir / 'summary.csv'} and {args.out_dir / 'flows.csv'}")


if __name__ == "__main__":
    main()
