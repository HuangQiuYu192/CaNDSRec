#!/usr/bin/env python3
"""Rerank CaNDS candidates with embedding-neighbor-smoothed transition evidence.

This diagnostic extends exact item-item transition reranking. Exact transition
evidence is sparse for tail items, so this script borrows evidence from a
candidate item's nearest neighbors in the normalized item embedding space:

    evidence'(j) = beta * evidence(j) + (1 - beta) * avg_{n in N(j)} evidence(n)
    final_score(j) = cands_score(j) + alpha * evidence'(j)

The base CaNDS model is not retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.analyze_transition_evidence_rerank import (
    build_model,
    build_transition_matrix,
    flatten_rows,
    fmt,
    grouped_metrics,
    history_transition_scores,
    normalize_candidate_evidence,
    parse_list,
    score_for_selection,
    write_csv,
)


def build_item_neighbors(model, n_items: int, neighbor_k: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if hasattr(model, "_normalized_item_embedding"):
            item_emb = model._normalized_item_embedding().detach()
        else:
            item_emb = F.normalize(model.item_embedding.weight.detach(), dim=-1)
        item_emb = item_emb.float()
        item_emb[0] = 0.0

        all_ids = []
        all_weights = []
        k = min(neighbor_k + 2, n_items)
        item_ids = torch.arange(n_items, device=item_emb.device)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            sim = torch.matmul(item_emb[start:end], item_emb.transpose(0, 1))
            sim[:, 0] = -float("inf")
            sim[torch.arange(end - start, device=item_emb.device), item_ids[start:end]] = -float("inf")
            values, indices = torch.topk(sim, k=k, dim=1)
            values = torch.clamp(values[:, :neighbor_k], min=0.0)
            indices = indices[:, :neighbor_k]
            denom = values.sum(dim=1, keepdim=True).clamp_min(1e-8)
            all_ids.append(indices.cpu())
            all_weights.append((values / denom).cpu())
    return torch.cat(all_ids, dim=0).long(), torch.cat(all_weights, dim=0).float()


def smoothed_candidate_evidence(
    evidence: torch.Tensor,
    candidate_items: torch.Tensor,
    neighbor_ids: torch.Tensor,
    neighbor_weights: torch.Tensor,
    direct_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = evidence.device
    neighbor_ids = neighbor_ids.to(device)
    neighbor_weights = neighbor_weights.to(device)

    direct = evidence.gather(1, candidate_items)
    candidate_neighbors = neighbor_ids[candidate_items]
    candidate_neighbor_weights = neighbor_weights[candidate_items]
    flat_neighbors = candidate_neighbors.reshape(candidate_items.size(0), -1)
    neighbor_evidence = evidence.gather(1, flat_neighbors).view_as(candidate_neighbor_weights)
    borrowed = (neighbor_evidence * candidate_neighbor_weights).sum(dim=-1)
    smoothed = direct_beta * direct + (1.0 - direct_beta) * borrowed
    return smoothed, direct, borrowed


def rerank_scores(
    scores: torch.Tensor,
    evidence: torch.Tensor,
    neighbor_ids: torch.Tensor,
    neighbor_weights: torch.Tensor,
    candidate_cutoff: int,
    alpha: float,
    direct_beta: float,
    evidence_norm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(candidate_cutoff, scores.size(1))
    candidate_items = torch.topk(scores, k=k, dim=1).indices
    candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
    candidate_mask.scatter_(1, candidate_items, True)

    smoothed, direct, borrowed = smoothed_candidate_evidence(
        evidence, candidate_items, neighbor_ids, neighbor_weights, direct_beta
    )
    dense_smoothed = torch.zeros_like(scores)
    dense_direct = torch.zeros_like(scores)
    dense_borrowed = torch.zeros_like(scores)
    dense_smoothed.scatter_(1, candidate_items, smoothed)
    dense_direct.scatter_(1, candidate_items, direct)
    dense_borrowed.scatter_(1, candidate_items, borrowed)
    dense_smoothed = normalize_candidate_evidence(dense_smoothed, candidate_mask, evidence_norm)

    reranked = scores.clone()
    if alpha != 0:
        reranked[candidate_mask] += float(alpha) * dense_smoothed[candidate_mask]
    return reranked, dense_smoothed, dense_direct, dense_borrowed, candidate_mask


def collect_eval(
    model,
    eval_data,
    transition: torch.Tensor,
    neighbor_ids: torch.Tensor,
    neighbor_weights: torch.Tensor,
    candidate_cutoff: int,
    alpha: float,
    seq_decay: float,
    recent_window: int,
    direct_beta: float,
    evidence_norm: str,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    transition = transition.to(device)
    output = {
        "items": [],
        "ranks": [],
        "direct_pos": [],
        "borrowed_pos": [],
        "smoothed_pos": [],
        "candidate_pos": [],
    }
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(eval_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()

            scores = model.full_sort_predict(interaction)
            if scores.dim() == 1:
                scores = scores.view(positive_i.size(0), -1)
            scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")

            evidence = history_transition_scores(
                interaction[model.ITEM_SEQ],
                transition,
                model.n_items,
                seq_decay=seq_decay,
                recent_window=recent_window,
            )
            evidence[:, 0] = 0.0
            scores, smoothed, direct, borrowed, candidate_mask = rerank_scores(
                scores,
                evidence,
                neighbor_ids,
                neighbor_weights,
                candidate_cutoff,
                alpha,
                direct_beta,
                evidence_norm,
            )
            pos_score = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1
            output["items"].extend(positive_i.cpu().numpy().tolist())
            output["ranks"].extend(ranks.cpu().numpy().tolist())
            output["direct_pos"].extend(direct[positive_u, positive_i].cpu().numpy().tolist())
            output["borrowed_pos"].extend(borrowed[positive_u, positive_i].cpu().numpy().tolist())
            output["smoothed_pos"].extend(smoothed[positive_u, positive_i].cpu().numpy().tolist())
            output["candidate_pos"].extend(candidate_mask[positive_u, positive_i].cpu().numpy().tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def grouped_metrics_with_smoothing(
    items: np.ndarray,
    ranks: np.ndarray,
    direct_pos: np.ndarray,
    borrowed_pos: np.ndarray,
    smoothed_pos: np.ndarray,
    candidate_pos: np.ndarray,
    pop: np.ndarray,
    cutoffs: list[int],
) -> list[dict]:
    rows = grouped_metrics(items, ranks, smoothed_pos, pop, cutoffs)
    for row in rows:
        group = row["group"]
        order = np.argsort(-pop[items], kind="stable")
        group_idx = {"all": np.arange(len(items)), **dict(zip(["head", "mid", "tail"], np.array_split(order, 3)))}[
            group
        ]
        row["direct_evidence_nonzero_pct"] = float((direct_pos[group_idx] > 0).mean()) if len(group_idx) else math.nan
        row["borrowed_evidence_nonzero_pct"] = (
            float((borrowed_pos[group_idx] > 0).mean()) if len(group_idx) else math.nan
        )
        row["smoothed_evidence_nonzero_pct"] = (
            float((smoothed_pos[group_idx] > 0).mean()) if len(group_idx) else math.nan
        )
        row["candidate_recall"] = float(candidate_pos[group_idx].mean()) if len(group_idx) else math.nan
        row["direct_evidence_mean"] = float(direct_pos[group_idx].mean()) if len(group_idx) else math.nan
        row["borrowed_evidence_mean"] = float(borrowed_pos[group_idx].mean()) if len(group_idx) else math.nan
        row["smoothed_evidence_mean"] = float(smoothed_pos[group_idx].mean()) if len(group_idx) else math.nan
    return rows


def write_markdown(path: Path, rows: list[dict], k: int) -> None:
    headers = [
        "dataset",
        "backbone",
        "hidden",
        "transition_mode",
        "candidate_cutoff",
        "neighbor_k",
        "direct_beta",
        "seq_decay",
        "recent_window",
        "selected_alpha",
        "selection_objective",
        "group",
        f"base_recall@{k}",
        f"rerank_recall@{k}",
        f"rel_recall@{k}",
        f"base_ndcg@{k}",
        f"rerank_ndcg@{k}",
        f"rel_ndcg@{k}",
        "base_median_rank",
        "rerank_median_rank",
        "rerank_candidate_recall",
        "rerank_direct_evidence_nonzero_pct",
        "rerank_borrowed_evidence_nonzero_pct",
        "rerank_smoothed_evidence_nonzero_pct",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--wearec_num_heads", default=1, type=int)
    parser.add_argument("--wearec_alpha", default=0.8, type=float)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=1024, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--transition_modes", default="conditional,pmi")
    parser.add_argument("--candidate_cutoffs", default="50,100")
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--seq_decays", default="0.8,1.0")
    parser.add_argument("--recent_windows", default="1,3,5")
    parser.add_argument("--direct_betas", default="0.0,0.25,0.5")
    parser.add_argument("--neighbor_ks", default="10,20,50")
    parser.add_argument("--neighbor_chunk_size", default=512, type=int)
    parser.add_argument("--edge_decay", default=0.8, type=float)
    parser.add_argument("--transition_topk", default=200, type=int)
    parser.add_argument("--evidence_norm", default="max", choices=["none", "max", "zscore"])
    parser.add_argument("--cutoffs", default="10,20,50,100")
    parser.add_argument("--selection_k", default=10, type=int)
    parser.add_argument(
        "--selection_objective",
        default="balanced_ndcg",
        choices=["overall_ndcg", "tail_ndcg", "balanced_ndcg", "overall_recall", "tail_recall"],
    )
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    config, dataset, train_data, valid_data, test_data, model = build_model(args.model, args.checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    item_seq_field = getattr(model, "ITEM_SEQ", f"{item_field}{config['LIST_SUFFIX']}")
    item_len_field = getattr(model, "ITEM_SEQ_LEN", config["ITEM_LIST_LENGTH_FIELD"])
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)

    cutoffs = parse_list(args.cutoffs, int)
    candidate_cutoffs = parse_list(args.candidate_cutoffs, int)
    alphas = parse_list(args.alphas, float)
    seq_decays = parse_list(args.seq_decays, float)
    recent_windows = parse_list(args.recent_windows, int)
    direct_betas = parse_list(args.direct_betas, float)
    neighbor_ks = parse_list(args.neighbor_ks, int)
    transition_modes = [mode.strip() for mode in args.transition_modes.split(",") if mode.strip()]

    max_neighbor_k = max(neighbor_ks)
    neighbor_ids, neighbor_weights = build_item_neighbors(
        model, dataset.item_num, neighbor_k=max_neighbor_k, chunk_size=args.neighbor_chunk_size
    )

    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "backbone": args.backbone or args.model.replace("CANDS", ""),
        "hidden": args.hidden_size,
        "max_len": args.max_item_list_length,
        "temperature": args.temperature,
        "edge_decay": args.edge_decay,
        "transition_topk": args.transition_topk,
        "evidence_norm": args.evidence_norm,
    }

    all_valid_rows = []
    selected_rows = []
    for transition_mode in transition_modes:
        transition = build_transition_matrix(
            train_data.dataset,
            dataset.item_num,
            item_field=item_field,
            item_seq_field=item_seq_field,
            item_len_field=item_len_field,
            mode=transition_mode,
            edge_decay=args.edge_decay,
            topk=args.transition_topk,
        )
        for neighbor_k in neighbor_ks:
            ids_k = neighbor_ids[:, :neighbor_k]
            weights_k = neighbor_weights[:, :neighbor_k]
            weights_k = weights_k / weights_k.sum(dim=1, keepdim=True).clamp_min(1e-8)
            for candidate_cutoff in candidate_cutoffs:
                for seq_decay in seq_decays:
                    for recent_window in recent_windows:
                        for direct_beta in direct_betas:
                            base_valid = collect_eval(
                                model,
                                valid_data,
                                transition,
                                ids_k,
                                weights_k,
                                candidate_cutoff,
                                alpha=0.0,
                                seq_decay=seq_decay,
                                recent_window=recent_window,
                                direct_beta=direct_beta,
                                evidence_norm=args.evidence_norm,
                                max_batches=args.max_batches,
                            )
                            base_valid_rows = grouped_metrics_with_smoothing(
                                base_valid["items"],
                                base_valid["ranks"],
                                base_valid["direct_pos"],
                                base_valid["borrowed_pos"],
                                base_valid["smoothed_pos"],
                                base_valid["candidate_pos"],
                                pop,
                                cutoffs,
                            )
                            best = None
                            for alpha in alphas:
                                valid_stats = collect_eval(
                                    model,
                                    valid_data,
                                    transition,
                                    ids_k,
                                    weights_k,
                                    candidate_cutoff,
                                    alpha=alpha,
                                    seq_decay=seq_decay,
                                    recent_window=recent_window,
                                    direct_beta=direct_beta,
                                    evidence_norm=args.evidence_norm,
                                    max_batches=args.max_batches,
                                )
                                valid_rows = grouped_metrics_with_smoothing(
                                    valid_stats["items"],
                                    valid_stats["ranks"],
                                    valid_stats["direct_pos"],
                                    valid_stats["borrowed_pos"],
                                    valid_stats["smoothed_pos"],
                                    valid_stats["candidate_pos"],
                                    pop,
                                    cutoffs,
                                )
                                score = score_for_selection(valid_rows, args.selection_objective, args.selection_k)
                                if best is None or score > best["score"]:
                                    best = {"alpha": alpha, "score": score}
                                setting = {
                                    "transition_mode": transition_mode,
                                    "candidate_cutoff": candidate_cutoff,
                                    "neighbor_k": neighbor_k,
                                    "seq_decay": seq_decay,
                                    "recent_window": recent_window,
                                    "direct_beta": direct_beta,
                                    "alpha": alpha,
                                    "selected_alpha": "",
                                    "selection_objective": args.selection_objective,
                                    "selection_score": score,
                                }
                                all_valid_rows.extend(
                                    flatten_rows(meta, "valid", setting, valid_rows, base_valid_rows)
                                )

                            assert best is not None
                            base_test = collect_eval(
                                model,
                                test_data,
                                transition,
                                ids_k,
                                weights_k,
                                candidate_cutoff,
                                alpha=0.0,
                                seq_decay=seq_decay,
                                recent_window=recent_window,
                                direct_beta=direct_beta,
                                evidence_norm=args.evidence_norm,
                                max_batches=args.max_batches,
                            )
                            base_test_rows = grouped_metrics_with_smoothing(
                                base_test["items"],
                                base_test["ranks"],
                                base_test["direct_pos"],
                                base_test["borrowed_pos"],
                                base_test["smoothed_pos"],
                                base_test["candidate_pos"],
                                pop,
                                cutoffs,
                            )
                            test_stats = collect_eval(
                                model,
                                test_data,
                                transition,
                                ids_k,
                                weights_k,
                                candidate_cutoff,
                                alpha=best["alpha"],
                                seq_decay=seq_decay,
                                recent_window=recent_window,
                                direct_beta=direct_beta,
                                evidence_norm=args.evidence_norm,
                                max_batches=args.max_batches,
                            )
                            test_rows = grouped_metrics_with_smoothing(
                                test_stats["items"],
                                test_stats["ranks"],
                                test_stats["direct_pos"],
                                test_stats["borrowed_pos"],
                                test_stats["smoothed_pos"],
                                test_stats["candidate_pos"],
                                pop,
                                cutoffs,
                            )
                            selected_setting = {
                                "transition_mode": transition_mode,
                                "candidate_cutoff": candidate_cutoff,
                                "neighbor_k": neighbor_k,
                                "seq_decay": seq_decay,
                                "recent_window": recent_window,
                                "direct_beta": direct_beta,
                                "alpha": best["alpha"],
                                "selected_alpha": best["alpha"],
                                "selection_objective": args.selection_objective,
                                "selection_score": best["score"],
                            }
                            selected_rows.extend(
                                flatten_rows(meta, "test", selected_setting, test_rows, base_test_rows)
                            )

    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_suffix(".all_valid.csv"), all_valid_rows)
    write_csv(out_prefix.with_suffix(".selected_test.csv"), selected_rows)
    write_markdown(out_prefix.with_suffix(".selected_test.md"), selected_rows, args.selection_k)
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"selected_test": selected_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote neighbor-smoothed transition rerank outputs with prefix {out_prefix}")


if __name__ == "__main__":
    main()
