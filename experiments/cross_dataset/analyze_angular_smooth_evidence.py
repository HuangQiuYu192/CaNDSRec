#!/usr/bin/env python3
"""Test whether AngularSmooth gains concentrate where tail targets have usable neighbor evidence.

The analysis is checkpoint-only: it compares matched CaNDS and AngularSmooth
models on the same test instances.  For every target, it measures direct
transition evidence from the current history and the evidence borrowed by its
angular neighbors. A popularity-matched random-neighbor control tests whether
the angular neighborhood is more context-relevant than chance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.analyze_group_metrics import build_model
from experiments.cross_dataset.analyze_transition_evidence_rerank import (
    build_transition_matrix,
    fmt,
    parse_list,
)


def target_groups(items: np.ndarray, pop: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-pop[items], kind="stable")
    return {"all": np.arange(len(items)), **dict(zip(["head", "mid", "tail"], np.array_split(order, 3)))}


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    hit = ranks <= k
    return float(hit.mean()), float((hit / np.log2(ranks + 1.0)).mean())


def build_neighbors(model, n_items: int, neighbor_k: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive-cosine angular neighbors, softmax weights, and mean cosine."""
    with torch.no_grad():
        item_dir = model._normalized_item_embedding().detach().float()
        item_dir[0] = 0.0
        all_ids, all_weights, all_cosines = [], [], []
        ids = torch.arange(n_items, device=item_dir.device)
        k = min(neighbor_k + 2, n_items)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            sim = item_dir[start:end] @ item_dir.transpose(0, 1)
            sim[:, 0] = -float("inf")
            sim[torch.arange(end - start, device=item_dir.device), ids[start:end]] = -float("inf")
            values, indices = torch.topk(sim, k=k, dim=1)
            values, indices = values[:, :neighbor_k], indices[:, :neighbor_k]
            positive = values.clamp_min(0.0)
            weights = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-8)
            all_ids.append(indices.cpu())
            all_weights.append(weights.cpu())
            all_cosines.append(torch.where(positive > 0, positive, torch.nan).nanmean(dim=1).cpu())
    return torch.cat(all_ids).long(), torch.cat(all_weights).float(), torch.cat(all_cosines).float()


def popularity_matched_random_neighbors(neighbor_ids: torch.Tensor, pop: np.ndarray, bins: int, seed: int) -> torch.Tensor:
    """Replace each angular neighbor with a random active item from its popularity bin."""
    active = pop > 0
    edges = np.quantile(pop[active], np.linspace(0.0, 1.0, bins + 1))
    item_bins = np.digitize(pop, edges[1:-1], right=True)
    pools = [np.where(active & (item_bins == bin_id))[0] for bin_id in range(bins)]
    rng = np.random.default_rng(seed)
    flat = neighbor_ids.numpy().reshape(-1)
    sampled = np.empty_like(flat)
    for bin_id, pool in enumerate(pools):
        mask = item_bins[flat] == bin_id
        if mask.any():
            sampled[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
    return torch.from_numpy(sampled.reshape(neighbor_ids.shape)).long()


def history_evidence(item_seq: torch.Tensor, transition: torch.Tensor, n_items: int, decay: float, window: int) -> torch.Tensor:
    if window > 0:
        item_seq = item_seq[:, -window:]
    batch, length = item_seq.shape
    positions = torch.arange(length, device=item_seq.device).view(1, -1)
    valid_len = (item_seq > 0).sum(dim=1, keepdim=True)
    distance = valid_len - 1 - positions
    weights = torch.pow(torch.tensor(decay, device=item_seq.device), distance.clamp_min(0).float())
    weights = weights * (item_seq > 0).float()
    history = torch.zeros(batch, n_items, device=item_seq.device)
    history.scatter_add_(1, item_seq, weights)
    history[:, 0] = 0.0
    history /= history.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.sparse.mm(transition.transpose(0, 1), history.transpose(0, 1)).transpose(0, 1)


def collect(
    base_model, smooth_model, eval_data, transition: torch.Tensor, neighbor_ids: torch.Tensor,
    random_neighbor_ids: torch.Tensor, neighbor_weights: torch.Tensor, neighbor_cosines: torch.Tensor, decay: float, window: int,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    device = next(base_model.parameters()).device
    transition = transition.to(device)
    neighbor_ids, random_neighbor_ids, neighbor_weights = neighbor_ids.to(device), random_neighbor_ids.to(device), neighbor_weights.to(device)
    output = {key: [] for key in ["items", "base_rank", "smooth_rank", "direct", "borrowed", "random_borrowed", "neighbor_cosine"]}
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(eval_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            base_scores = base_model.full_sort_predict(interaction)
            smooth_scores = smooth_model.full_sort_predict(interaction)
            if base_scores.dim() == 1:
                base_scores = base_scores.view(positive_i.size(0), -1)
                smooth_scores = smooth_scores.view(positive_i.size(0), -1)
            for scores in (base_scores, smooth_scores):
                scores[:, 0] = -float("inf")
                if history_index is not None:
                    scores[history_index] = -float("inf")
            base_pos, smooth_pos = base_scores[positive_u, positive_i], smooth_scores[positive_u, positive_i]
            evidence = history_evidence(interaction[base_model.ITEM_SEQ], transition, base_model.n_items, decay, window)
            direct = evidence[positive_u, positive_i]
            pos_neighbors = neighbor_ids[positive_i]
            borrowed = (evidence[positive_u.unsqueeze(1), pos_neighbors] * neighbor_weights[positive_i]).sum(dim=1)
            pos_random_neighbors = random_neighbor_ids[positive_i]
            random_borrowed = (evidence[positive_u.unsqueeze(1), pos_random_neighbors] * neighbor_weights[positive_i]).sum(dim=1)
            output["items"].extend(positive_i.cpu().tolist())
            output["base_rank"].extend(((base_scores[positive_u] > base_pos.unsqueeze(1)).sum(dim=1) + 1).cpu().tolist())
            output["smooth_rank"].extend(((smooth_scores[positive_u] > smooth_pos.unsqueeze(1)).sum(dim=1) + 1).cpu().tolist())
            output["direct"].extend(direct.cpu().tolist())
            output["borrowed"].extend(borrowed.cpu().tolist())
            output["random_borrowed"].extend(random_borrowed.cpu().tolist())
            output["neighbor_cosine"].extend(neighbor_cosines[positive_i.cpu()].tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def summarize(indices: np.ndarray, stats: dict[str, np.ndarray], cutoffs: list[int], label: str) -> dict:
    base_rank, smooth_rank = stats["base_rank"][indices], stats["smooth_rank"][indices]
    row = {"group": label, "n": int(len(indices))}
    for k in cutoffs:
        base_recall, base_ndcg = metric_at(base_rank, k)
        smooth_recall, smooth_ndcg = metric_at(smooth_rank, k)
        row.update({f"base_recall@{k}": base_recall, f"smooth_recall@{k}": smooth_recall,
                    f"delta_recall@{k}": smooth_recall - base_recall, f"base_ndcg@{k}": base_ndcg,
                    f"smooth_ndcg@{k}": smooth_ndcg, f"delta_ndcg@{k}": smooth_ndcg - base_ndcg})
    row["base_median_rank"] = float(np.median(base_rank))
    row["smooth_median_rank"] = float(np.median(smooth_rank))
    row["rank_win_rate"] = float((smooth_rank < base_rank).mean())
    row["rank_loss_rate"] = float((smooth_rank > base_rank).mean())
    for field in ["direct", "borrowed", "random_borrowed", "neighbor_cosine"]:
        values = stats[field][indices]
        row[f"{field}_mean"] = float(values.mean())
        row[f"{field}_nonzero_pct"] = float((values > 0).mean()) if field != "neighbor_cosine" else math.nan
    row["angular_minus_random_borrowed_mean"] = row["borrowed_mean"] - row["random_borrowed_mean"]
    return row


def write_outputs(prefix: Path, rows: list[dict]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(fmt(row.get(key, "")) for key in headers) + " |" for row in rows]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--smooth_checkpoint", required=True)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", required=True, type=int)
    parser.add_argument("--max_item_list_length", required=True, type=int)
    parser.add_argument("--cands_inner_size", required=True, type=int)
    parser.add_argument("--smooth_inner_size", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--angular_smooth_weight", required=True, type=float)
    parser.add_argument("--angular_smooth_k", required=True, type=int)
    parser.add_argument("--angular_smooth_temperature", required=True, type=float)
    parser.add_argument("--angular_smooth_pop_quantile", required=True, type=float)
    parser.add_argument("--angular_smooth_sim_threshold", required=True, type=float)
    parser.add_argument("--angular_smooth_pop_weight", default=False)
    parser.add_argument("--transition_mode", default="conditional", choices=["conditional", "pmi", "log_conditional"])
    parser.add_argument("--transition_topk", default=200, type=int)
    parser.add_argument("--edge_decay", default=0.8, type=float)
    parser.add_argument("--seq_decay", default=1.0, type=float)
    parser.add_argument("--recent_window", default=5, type=int)
    parser.add_argument("--neighbor_k", default=10, type=int)
    parser.add_argument("--neighbor_chunk_size", default=512, type=int)
    parser.add_argument("--random_popularity_bins", default=10, type=int)
    parser.add_argument("--cutoffs", default="10,50")
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    base_args = copy.copy(args)
    base_args.inner_size = args.cands_inner_size
    smooth_args = copy.copy(args)
    smooth_args.inner_size = args.smooth_inner_size
    config, dataset, train_data, _, test_data, base_model = build_model("CANDSSASRec", args.cands_checkpoint, base_args)
    _, _, _, _, _, smooth_model = build_model("AngularSmoothCANDSSASRec", args.smooth_checkpoint, smooth_args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    transition = build_transition_matrix(train_data.dataset, dataset.item_num, item_field, base_model.ITEM_SEQ,
                                         base_model.ITEM_SEQ_LEN, args.transition_mode, args.edge_decay, args.transition_topk)
    neighbor_ids, neighbor_weights, neighbor_cosines = build_neighbors(base_model, dataset.item_num, args.neighbor_k, args.neighbor_chunk_size)
    random_neighbor_ids = popularity_matched_random_neighbors(neighbor_ids, pop, args.random_popularity_bins, args.seed)
    stats = collect(base_model, smooth_model, test_data, transition, neighbor_ids, random_neighbor_ids, neighbor_weights, neighbor_cosines,
                    args.seq_decay, args.recent_window, args.max_batches)
    cutoffs = parse_list(args.cutoffs, int)
    rows = []
    for label, indices in target_groups(stats["items"].astype(np.int64), pop).items():
        rows.append(summarize(indices, stats, cutoffs, label))
        if label == "tail":
            borrowed = stats["borrowed"][indices]
            zero, nonzero = indices[borrowed <= 0], indices[borrowed > 0]
            if len(zero):
                rows.append(summarize(zero, stats, cutoffs, "tail_borrowed_zero"))
            if len(nonzero):
                rows.append(summarize(nonzero, stats, cutoffs, "tail_borrowed_nonzero"))
                ordered = nonzero[np.argsort(stats["borrowed"][nonzero], kind="stable")]
                for name, part in zip(["tail_borrowed_nonzero_low", "tail_borrowed_nonzero_high"], np.array_split(ordered, 2)):
                    if len(part):
                        rows.append(summarize(part, stats, cutoffs, name))
    meta = {"dataset": args.dataset, "hidden": args.hidden_size, "temperature": args.temperature,
            "neighbor_k": args.neighbor_k, "transition_mode": args.transition_mode,
            "recent_window": args.recent_window, "random_popularity_bins": args.random_popularity_bins,
            "base_checkpoint": args.cands_checkpoint,
            "smooth_checkpoint": args.smooth_checkpoint}
    rows = [{**meta, **row} for row in rows]
    write_outputs(Path(args.out_prefix), rows)
    print(f"wrote {len(rows)} rows to {Path(args.out_prefix).with_suffix('.csv')} and .md")


if __name__ == "__main__":
    main()
