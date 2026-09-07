#!/usr/bin/env python3
"""Validate state-routed selection among separately trained CaNDS temperatures.

Changing the positive score multiplier T at inference cannot change a ranking.
This experiment therefore treats checkpoints trained at different temperatures
as different experts.  It learns a validation-only route for short histories
and low/mid/high multi-item sequence coherence, then evaluates it on test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import align_architecture_to_checkpoint
from experiments.cross_dataset.analyze_group_metrics import build_model


def parse_checkpoints(text: str) -> dict[float, str]:
    pairs = {}
    for part in text.split(","):
        if not part.strip() or "=" not in part:
            raise ValueError("--checkpoints must be temperature=/path/model.pth pairs separated by commas")
        temperature, path = part.split("=", 1)
        pairs[float(temperature)] = path
    if len(pairs) < 2:
        raise ValueError("At least two temperature-trained experts are required.")
    return dict(sorted(pairs.items()))


def metric(ranks: np.ndarray, k: int) -> tuple[float, float]:
    hit = ranks <= k
    return float(hit.mean()), float((hit / np.log2(ranks + 1.0)).mean())


def tail_groups(items: np.ndarray, popularity: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-popularity[items], kind="stable")
    return dict(zip(["head", "mid", "tail"], np.array_split(order, 3)))


def coherence_values(
    item_seq: torch.Tensor, item_seq_len: torch.Tensor, item_dir: torch.Tensor, recent_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recent pairwise angular consistency and available history length.

    A length-one history has no pairwise consistency.  It must be a separate
    state, not assigned an artificial coherence of one and mixed with stable
    multi-item histories.
    """
    emb = item_dir[item_seq]
    positions = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
    lengths = item_seq_len.long().clamp(min=0, max=item_seq.size(1)).unsqueeze(1)
    start = (lengths - recent_window).clamp_min(0) if recent_window > 0 else torch.zeros_like(lengths)
    # RecBole sequential tensors are right-padded: valid history occupies
    # [:sequence_length], so the recent window is [length-window:length].
    mask = (item_seq > 0) & (positions >= start) & (positions < lengths)
    count = mask.sum(dim=1)
    pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)) & ~torch.eye(
        item_seq.size(1), device=item_seq.device, dtype=torch.bool
    ).unsqueeze(0)
    pairwise = torch.matmul(emb, emb.transpose(1, 2))
    coherence = (pairwise * pair_mask).sum(dim=(1, 2)) / (count * (count - 1)).clamp_min(1)
    coherence = torch.where(count >= 2, coherence, torch.full_like(coherence, float("nan")))
    return coherence, count


def collect(models: dict[float, object], data, reference_temp: float, recent_window: int, max_batches: int | None):
    reference = models[reference_temp]
    device = next(reference.parameters()).device
    item_dir = reference._normalized_item_embedding().detach()
    output = {"items": [], "coherence": [], "recent_history_length": []}
    for temperature in models:
        output[f"rank_t{temperature:g}"] = []
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            output["items"].extend(positive_i.cpu().tolist())
            coherence, history_length = coherence_values(
                interaction[reference.ITEM_SEQ], interaction[reference.ITEM_SEQ_LEN], item_dir, recent_window
            )
            output["coherence"].extend(coherence.cpu().tolist())
            output["recent_history_length"].extend(history_length.cpu().tolist())
            for temperature, model in models.items():
                scores = model.full_sort_predict(interaction)
                if scores.dim() == 1:
                    scores = scores.view(positive_i.size(0), -1)
                scores[:, 0] = -float("inf")
                if history_index is not None:
                    scores[history_index] = -float("inf")
                pos_score = scores[positive_u, positive_i]
                ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1
                output[f"rank_t{temperature:g}"].extend(ranks.cpu().tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def assign_coherence_groups(values: np.ndarray, history_length: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    low, high = thresholds
    groups = np.full(len(values), "short", dtype="U5")
    multi = history_length >= 2
    groups[multi & (values <= low)] = "low"
    groups[multi & (values > low) & (values <= high)] = "mid"
    groups[multi & (values > high)] = "high"
    return groups


def best_expert_by_group(valid: dict, temperatures: list[float], groups: np.ndarray, select_k: int) -> tuple[dict[str, float], float]:
    mapping = {}
    for group in ["short", "low", "mid", "high"]:
        idx = np.flatnonzero(groups == group)
        if not len(idx):
            raise ValueError(f"State group {group!r} is empty; coherence feature cannot support a four-state router.")
        candidates = []
        for temperature in temperatures:
            _, ndcg = metric(valid[f"rank_t{temperature:g}"][idx], select_k)
            candidates.append((ndcg, -temperature, temperature))
        mapping[group] = max(candidates)[2]
    all_candidates = []
    for temperature in temperatures:
        _, ndcg = metric(valid[f"rank_t{temperature:g}"], select_k)
        all_candidates.append((ndcg, -temperature, temperature))
    return mapping, max(all_candidates)[2]


def routed_ranks(stats: dict, route: dict[str, float], coherence_group: np.ndarray) -> np.ndarray:
    return np.asarray([stats[f"rank_t{route[group]:g}"][i] for i, group in enumerate(coherence_group)], dtype=np.int64)


def rows_for_split(stats: dict, split: str, temperatures: list[float], coherence_group: np.ndarray, route: dict[str, float], always_temp: float, reference_temp: float, popularity: np.ndarray, cutoffs: list[int]) -> list[dict]:
    partitions = {"all": np.arange(len(stats["items"]))}
    partitions.update({f"state_{group}": np.flatnonzero(coherence_group == group) for group in ["short", "low", "mid", "high"]})
    for group, idx in tail_groups(stats["items"], popularity).items():
        partitions[group] = idx
    rank_sets = {f"expert_t{temperature:g}": stats[f"rank_t{temperature:g}"] for temperature in temperatures}
    rank_sets["validation_selected_route"] = routed_ranks(stats, route, coherence_group)
    rank_sets[f"always_best_valid_t{always_temp:g}"] = stats[f"rank_t{always_temp:g}"]
    rank_sets[f"reference_t{reference_temp:g}"] = stats[f"rank_t{reference_temp:g}"]
    rank_sets["per_instance_oracle_test_only"] = np.minimum.reduce([stats[f"rank_t{temperature:g}"] for temperature in temperatures])
    rows = []
    for label, indices in partitions.items():
        for variant, ranks in rank_sets.items():
            row = {"split": split, "group": label, "n": int(len(indices)), "variant": variant}
            for k in cutoffs:
                recall, ndcg = metric(ranks[indices], k)
                row[f"recall@{k}"], row[f"ndcg@{k}"] = recall, ndcg
            row["median_rank"] = float(np.median(ranks[indices]))
            rows.append(row)
    return rows


def write_outputs(out_prefix: Path, rows: list[dict], metadata: dict, cutoffs: list[int]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "group", "n", "variant", "median_rank"] + [field for k in cutoffs for field in (f"recall@{k}", f"ndcg@{k}")]
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Temperature-expert routing validation", "", "The route is chosen on validation coherence groups and frozen for test. `per_instance_oracle_test_only` is an unattainable upper bound, not a method.", "", "| split | group | n | variant | Recall@10 | NDCG@10 | Recall@100 | median rank |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['group']} | {row['n']} | {row['variant']} | {row.get('recall@10', math.nan):.4f} | {row.get('ndcg@10', math.nan):.4f} | {row.get('recall@100', math.nan):.4f} | {row['median_rank']:.1f} |")
    Path(f"{out_prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Comma-separated T=checkpoint.pth pairs.")
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
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--reference_temperature", default=10.0, type=float)
    parser.add_argument("--recent_window", default=5, type=int)
    parser.add_argument("--select_k", default=10, type=int)
    parser.add_argument("--cutoffs", default="10,100")
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    temperatures = list(checkpoints)
    if args.reference_temperature not in checkpoints:
        raise ValueError(f"reference_temperature={args.reference_temperature} has no checkpoint")
    models, train_data = {}, None
    valid_data = test_data = config = dataset = None
    for temperature, checkpoint in checkpoints.items():
        expert_args = copy.deepcopy(args)
        expert_args.checkpoint, expert_args.temperature = checkpoint, temperature
        align_architecture_to_checkpoint(expert_args)
        expert_config, expert_dataset, expert_train, expert_valid, expert_test, expert_model = build_model("CANDSSASRec", checkpoint, expert_args)
        models[temperature] = expert_model
        if train_data is None:
            config, dataset, train_data, valid_data, test_data = expert_config, expert_dataset, expert_train, expert_valid, expert_test
    valid = collect(models, valid_data, args.reference_temperature, args.recent_window, args.max_batches)
    valid_multi = valid["coherence"][valid["recent_history_length"] >= 2]
    if len(valid_multi) < 30 or len(np.unique(valid_multi)) < 3:
        raise ValueError("Recent pairwise coherence has insufficient variation for a routing experiment.")
    thresholds = tuple(np.quantile(valid_multi, [1 / 3, 2 / 3]).tolist())
    valid_groups = assign_coherence_groups(valid["coherence"], valid["recent_history_length"], thresholds)
    route, always_temp = best_expert_by_group(valid, temperatures, valid_groups, args.select_k)
    test = collect(models, test_data, args.reference_temperature, args.recent_window, args.max_batches)
    test_groups = assign_coherence_groups(test["coherence"], test["recent_history_length"], thresholds)
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    cutoffs = [int(value) for value in args.cutoffs.split(",") if value.strip()]
    rows = rows_for_split(valid, "valid", temperatures, valid_groups, route, always_temp, args.reference_temperature, popularity, cutoffs)
    rows += rows_for_split(test, "test", temperatures, test_groups, route, always_temp, args.reference_temperature, popularity, cutoffs)
    metadata = {"dataset": args.dataset, "checkpoints": checkpoints, "reference_temperature": args.reference_temperature, "recent_window": args.recent_window, "coherence_thresholds_from_valid": thresholds, "validation_state_counts": {state: int((valid_groups == state).sum()) for state in ["short", "low", "mid", "high"]}, "test_state_counts": {state: int((test_groups == state).sum()) for state in ["short", "low", "mid", "high"]}, "validation_selected_route": route, "always_best_validation_temperature": always_temp, "select_metric": f"NDCG@{args.select_k}", "max_batches": args.max_batches}
    write_outputs(Path(args.out_prefix), rows, metadata, cutoffs)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix}_summary.csv, .md and _meta.json")


if __name__ == "__main__":
    main()
