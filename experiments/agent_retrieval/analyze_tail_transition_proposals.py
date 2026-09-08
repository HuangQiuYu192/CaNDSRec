#!/usr/bin/env python3
"""Stage-1 validation of a real-transition tail proposal simulator.

No pseudo-interaction is generated here. A simulator is built exclusively from
train-split direct predecessor -> eligible-tail-target transitions, then asked
to retrieve the held-out validation/test tail target from the current real
history. Random-tail, tail-popularity, and CaNDS full-sort candidates are
reported as controls.
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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import align_architecture_to_checkpoint
from experiments.agent_retrieval.analyze_tail_transition_support import collect_support, recent_predecessors
from experiments.cross_dataset.analyze_group_metrics import build_model


def parse_ints(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--cutoffs must contain positive integers")
    return values


def train_transition_matrix(train_data, model, config, eligible_tail: set[int]) -> sparse.csr_matrix:
    """Build P(eligible tail target | direct predecessor) using train only."""
    inter = train_data.dataset.inter_feat
    targets = inter[config["ITEM_ID_FIELD"]].cpu().numpy().astype(np.int64)
    predecessors = recent_predecessors(inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN]).cpu().numpy().astype(np.int64)
    counts: dict[int, Counter] = defaultdict(Counter)
    for predecessor, target in zip(predecessors, targets):
        if predecessor > 0 and target in eligible_tail:
            counts[int(predecessor)][int(target)] += 1
    row, col, values = [], [], []
    for predecessor, targets_for_pred in counts.items():
        denom = sum(targets_for_pred.values())
        for target, count in targets_for_pred.items():
            row.append(predecessor)
            col.append(target)
            values.append(count / denom)
    return sparse.csr_matrix((np.asarray(values, dtype=np.float32), (row, col)), shape=(model.n_items, model.n_items))


def sequence_histories(item_seq: torch.Tensor, item_seq_len: torch.Tensor, window: int) -> list[np.ndarray]:
    """Return final real history items from RecBole right-padded tensors."""
    sequences, lengths = item_seq.cpu().numpy(), item_seq_len.cpu().numpy()
    histories = []
    for sequence, length in zip(sequences, lengths):
        length = min(int(length), len(sequence))
        start = max(0, length - window) if window > 0 else 0
        histories.append(sequence[start:length][sequence[start:length] > 0].astype(np.int64, copy=False))
    return histories


def history_transition_scores(histories: list[np.ndarray], transition: sparse.csr_matrix, decay: float) -> sparse.csr_matrix:
    rows, cols, values = [], [], []
    for row, history in enumerate(histories):
        for position, item in enumerate(history):
            rows.append(row)
            cols.append(int(item))
            values.append(float(decay ** (len(history) - 1 - position)))
    history_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(histories), transition.shape[0]), dtype=np.float32)
    return (history_matrix @ transition).tocsr()


def ordered_sparse_candidates(scores: sparse.csr_matrix, row: int, history: np.ndarray, cutoff: int) -> np.ndarray:
    begin, end = scores.indptr[row], scores.indptr[row + 1]
    items, values = scores.indices[begin:end], scores.data[begin:end]
    if len(history):
        keep = ~np.isin(items, history)
        items, values = items[keep], values[keep]
    if not len(items):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((items, -values))[:cutoff]
    return items[order].astype(np.int64, copy=False)


def top_cands(model, interaction, history_index, positive_u, cutoff: int) -> list[np.ndarray]:
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)
        if scores.dim() == 1:
            scores = scores.view(positive_u.size(0), -1)
        scores[:, 0] = -float("inf")
        if history_index is not None:
            scores[history_index] = -float("inf")
        values, indices = torch.topk(scores[positive_u], k=cutoff, dim=1)
    return [row.cpu().numpy().astype(np.int64, copy=False) for row in indices]


def candidate_metrics(targets: np.ndarray, candidates: list[np.ndarray], indices: np.ndarray, cutoffs: list[int]) -> dict:
    row = {"n": int(len(indices)), "mean_candidate_count": float(np.mean([len(candidates[i]) for i in indices])) if len(indices) else math.nan}
    for cutoff in cutoffs:
        row[f"candidate_recall@{cutoff}"] = float(np.mean([targets[i] in candidates[i][:cutoff] for i in indices])) if len(indices) else math.nan
    return row


def evaluate_split(model, data, transition, eligible_tail: np.ndarray, tail_items: set[int], pop_order: np.ndarray, cutoffs: list[int], window: int, decay: float, seed: int):
    device = next(model.parameters()).device
    max_cutoff = max(cutoffs)
    rng = np.random.default_rng(seed)
    outputs = {key: [] for key in ["targets", "target_eligible", "transition", "random_tail", "popularity_tail", "cands"]}
    for batched_data in data:
        interaction = batched_data[0].to(device)
        history_index = batched_data[1]
        positive_u = torch.as_tensor(batched_data[2], device=device).long()
        positive_i = torch.as_tensor(batched_data[3], device=device).long()
        histories = sequence_histories(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN], window)
        transition_scores = history_transition_scores(histories, transition, decay)
        cands = top_cands(model, interaction, history_index, positive_u, max_cutoff)
        targets = positive_i.cpu().numpy().astype(np.int64)
        for row, (target, history) in enumerate(zip(targets, histories)):
            available = eligible_tail[~np.isin(eligible_tail, history)]
            pop = pop_order[~np.isin(pop_order, history)][:max_cutoff]
            random = rng.choice(available, size=min(max_cutoff, len(available)), replace=False) if len(available) else np.empty(0, dtype=np.int64)
            outputs["targets"].append(int(target))
            outputs["target_eligible"].append(int(target in set(eligible_tail.tolist())))
            outputs["transition"].append(ordered_sparse_candidates(transition_scores, row, history, max_cutoff))
            outputs["random_tail"].append(random)
            outputs["popularity_tail"].append(pop)
            outputs["cands"].append(cands[row])
    outputs["targets"] = np.asarray(outputs["targets"], dtype=np.int64)
    outputs["target_eligible"] = np.asarray(outputs["target_eligible"], dtype=bool)
    tail_idx = np.asarray([index for index, item in enumerate(outputs["targets"]) if int(item) in tail_items], dtype=np.int64)
    return outputs, tail_idx


def summarize_split(name: str, outputs: dict, tail_idx: np.ndarray, cutoffs: list[int]) -> list[dict]:
    rows = []
    groups = {"tail_all": tail_idx, "tail_eligible": tail_idx[outputs["target_eligible"][tail_idx]]}
    for group, indices in groups.items():
        for variant in ["transition", "random_tail", "popularity_tail", "cands"]:
            row = {"split": name, "group": group, "variant": variant, **candidate_metrics(outputs["targets"], outputs[variant], indices, cutoffs)}
            rows.append(row)
    return rows


def write_outputs(out_prefix: Path, rows: list[dict], metadata: dict, cutoffs: list[int]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "group", "variant", "n", "mean_candidate_count"] + [f"candidate_recall@{cutoff}" for cutoff in cutoffs]
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Tail transition proposal validation", "", "The transition simulator uses only training-split direct edges and proposes only Stage-0 eligible tail items. No pseudo-positive is trained in this stage.", "", "| split | group | variant | n | mean candidates | " + " | ".join(f"Tail Candidate Recall@{cutoff}" for cutoff in cutoffs) + " |", "| --- | --- | --- | ---: | ---: | " + " | ".join(["---:"] * len(cutoffs)) + " |"]
    for row in rows:
        lines.append("| {split} | {group} | {variant} | {n} | {count:.2f} | {values} |".format(split=row["split"], group=row["group"], variant=row["variant"], n=row["n"], count=row["mean_candidate_count"], values=" | ".join(f"{row[f'candidate_recall@{cutoff}']:.4f}" for cutoff in cutoffs)))
    lines += ["", "## Go/no-go rule", "", "Use validation `tail_eligible` results. The transition simulator must beat both `random_tail` and `popularity_tail` at practical cutoffs before it may create pseudo-positive proposals. Test rows are frozen external checks, not tuning inputs.", ""]
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
    parser.add_argument("--recent_window", default=5, type=int)
    parser.add_argument("--transition_decay", default=0.7, type=float)
    parser.add_argument("--cutoffs", default="5,10,20,50")
    parser.add_argument("--out_prefix", default="analysis_results/agent_simulator/beauty_tail_transition_proposals/Beauty_h256_len50_temp10")
    args = parser.parse_args()
    cutoffs = parse_ints(args.cutoffs)
    align_architecture_to_checkpoint(args)
    config, dataset, train_data, valid_data, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    popularity, item_rows = collect_support(train_data, model, config, args.min_contexts, args.min_predecessors)
    tail_items = {row["item_id"] for row in item_rows if row["popularity_group"] == "tail"}
    eligible_tail = np.asarray([row["item_id"] for row in item_rows if row["popularity_group"] == "tail" and row["eligible_for_simulation"]], dtype=np.int64)
    if not len(eligible_tail):
        raise ValueError("Stage-0 whitelist is empty; no eligible tail items can be proposed.")
    transition = train_transition_matrix(train_data, model, config, set(eligible_tail.tolist()))
    pop_order = eligible_tail[np.argsort(-popularity[eligible_tail], kind="stable")]
    valid_outputs, valid_tail = evaluate_split(model, valid_data, transition, eligible_tail, tail_items, pop_order, cutoffs, args.recent_window, args.transition_decay, args.seed)
    test_outputs, test_tail = evaluate_split(model, test_data, transition, eligible_tail, tail_items, pop_order, cutoffs, args.recent_window, args.transition_decay, args.seed + 1)
    rows = summarize_split("valid", valid_outputs, valid_tail, cutoffs) + summarize_split("test", test_outputs, test_tail, cutoffs)
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "eligible_tail_items": int(len(eligible_tail)), "tail_items": int(len(tail_items)), "transition_nonzero_edges": int(transition.nnz), "min_contexts": args.min_contexts, "min_predecessors": args.min_predecessors, "recent_window": args.recent_window, "transition_decay": args.transition_decay, "cutoffs": cutoffs}
    write_outputs(Path(args.out_prefix), rows, metadata, cutoffs)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix}_summary.csv, .md and _meta.json")


if __name__ == "__main__":
    main()
