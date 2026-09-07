#!/usr/bin/env python3
"""Diagnose whether AngularSmooth neighbors crowd out tail targets at test time."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.analyze_angular_smooth_evidence import build_neighbors, history_evidence, target_groups
from experiments.cross_dataset.analyze_group_metrics import build_model
from experiments.cross_dataset.analyze_transition_evidence_rerank import build_transition_matrix


def collect(base_model, smooth_model, eval_data, transition, neighbor_ids, neighbor_weights, decay, window, max_batches):
    device = next(base_model.parameters()).device
    transition, neighbor_ids, neighbor_weights = transition.to(device), neighbor_ids.to(device), neighbor_weights.to(device)
    rows = []
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(eval_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            base_scores, smooth_scores = base_model.full_sort_predict(interaction), smooth_model.full_sort_predict(interaction)
            if base_scores.dim() == 1:
                base_scores, smooth_scores = base_scores.view(positive_i.size(0), -1), smooth_scores.view(positive_i.size(0), -1)
            for scores in (base_scores, smooth_scores):
                scores[:, 0] = -float("inf")
                if history_index is not None:
                    scores[history_index] = -float("inf")
            evidence = history_evidence(interaction[base_model.ITEM_SEQ], transition, base_model.n_items, decay, window)
            pos_neighbors, weights = neighbor_ids[positive_i], neighbor_weights[positive_i]
            borrowed = (evidence[positive_u.unsqueeze(1), pos_neighbors] * weights).sum(dim=1)
            base_pos, smooth_pos = base_scores[positive_u, positive_i], smooth_scores[positive_u, positive_i]
            base_neighbor_scores = base_scores.gather(1, pos_neighbors)
            smooth_neighbor_scores = smooth_scores.gather(1, pos_neighbors)
            base_best_score, base_best_col = base_neighbor_scores.max(dim=1)
            smooth_best_score, smooth_best_col = smooth_neighbor_scores.max(dim=1)
            base_rank = (base_scores[positive_u] > base_pos.unsqueeze(1)).sum(dim=1) + 1
            smooth_rank = (smooth_scores[positive_u] > smooth_pos.unsqueeze(1)).sum(dim=1) + 1
            base_best_rank = (base_scores[positive_u] > base_best_score.unsqueeze(1)).sum(dim=1) + 1
            smooth_best_rank = (smooth_scores[positive_u] > smooth_best_score.unsqueeze(1)).sum(dim=1) + 1
            for index in range(positive_i.size(0)):
                rows.append({
                    "item": int(positive_i[index]), "borrowed": float(borrowed[index]),
                    "base_rank": int(base_rank[index]), "smooth_rank": int(smooth_rank[index]),
                    "rank_delta": int(base_rank[index] - smooth_rank[index]),
                    "base_best_neighbor": int(pos_neighbors[index, base_best_col[index]]),
                    "smooth_best_neighbor": int(pos_neighbors[index, smooth_best_col[index]]),
                    "base_best_neighbor_rank": int(base_best_rank[index]), "smooth_best_neighbor_rank": int(smooth_best_rank[index]),
                    "base_target_minus_neighbor": float(base_pos[index] - base_best_score[index]),
                    "smooth_target_minus_neighbor": float(smooth_pos[index] - smooth_best_score[index]),
                })
    return rows


def aggregate(rows, label):
    if not rows:
        return None
    values = lambda key: np.asarray([row[key] for row in rows])
    base_rank, smooth_rank = values("base_rank"), values("smooth_rank")
    base_gap, smooth_gap = values("base_target_minus_neighbor"), values("smooth_target_minus_neighbor")
    return {
        "group": label, "n": len(rows),
        "base_target_top10": float((base_rank <= 10).mean()), "smooth_target_top10": float((smooth_rank <= 10).mean()),
        "base_neighbor_top10": float((values("base_best_neighbor_rank") <= 10).mean()),
        "smooth_neighbor_top10": float((values("smooth_best_neighbor_rank") <= 10).mean()),
        "base_neighbor_beats_target": float((base_gap < 0).mean()), "smooth_neighbor_beats_target": float((smooth_gap < 0).mean()),
        "neighbor_overtakes_target": float(((base_gap >= 0) & (smooth_gap < 0)).mean()),
        "target_top10_lost_with_neighbor_top10": float(((base_rank <= 10) & (smooth_rank > 10) & (values("smooth_best_neighbor_rank") <= 10)).mean()),
        "target_rank_improved": float((smooth_rank < base_rank).mean()), "target_rank_worsened": float((smooth_rank > base_rank).mean()),
        "base_target_minus_neighbor_mean": float(base_gap.mean()), "smooth_target_minus_neighbor_mean": float(smooth_gap.mean()),
        "gap_delta_mean": float((smooth_gap - base_gap).mean()), "borrowed_mean": float(values("borrowed").mean()),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
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
    parser.add_argument("--transition_mode", default="conditional")
    parser.add_argument("--transition_topk", default=200, type=int)
    parser.add_argument("--edge_decay", default=0.8, type=float)
    parser.add_argument("--seq_decay", default=1.0, type=float)
    parser.add_argument("--recent_window", default=5, type=int)
    parser.add_argument("--neighbor_k", default=10, type=int)
    parser.add_argument("--neighbor_chunk_size", default=512, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    base_args, smooth_args = copy.copy(args), copy.copy(args)
    base_args.inner_size, smooth_args.inner_size = args.cands_inner_size, args.smooth_inner_size
    config, dataset, train_data, _, test_data, base_model = build_model("CANDSSASRec", args.cands_checkpoint, base_args)
    _, _, _, _, _, smooth_model = build_model("AngularSmoothCANDSSASRec", args.smooth_checkpoint, smooth_args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    transition = build_transition_matrix(train_data.dataset, dataset.item_num, item_field, base_model.ITEM_SEQ, base_model.ITEM_SEQ_LEN, args.transition_mode, args.edge_decay, args.transition_topk)
    neighbor_ids, neighbor_weights, _ = build_neighbors(base_model, dataset.item_num, args.neighbor_k, args.neighbor_chunk_size)
    records = collect(base_model, smooth_model, test_data, transition, neighbor_ids, neighbor_weights, args.seq_decay, args.recent_window, args.max_batches)
    items = np.asarray([row["item"] for row in records])
    tail_idx = target_groups(items, pop)["tail"]
    tail_records = [records[index] for index in tail_idx if records[index]["borrowed"] > 0]
    improved = [row for row in tail_records if row["smooth_rank"] < row["base_rank"]]
    worsened = [row for row in tail_records if row["smooth_rank"] > row["base_rank"]]
    summary = [row for row in [aggregate(tail_records, "tail_borrowed_nonzero"), aggregate(improved, "target_rank_improved"), aggregate(worsened, "target_rank_worsened")] if row]
    meta = {"dataset": args.dataset, "hidden": args.hidden_size, "temperature": args.temperature, "neighbor_k": args.neighbor_k}
    summary = [{**meta, **row} for row in summary]
    prefix = Path(args.out_prefix)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)
    write_csv(prefix.with_name(prefix.name + "_samples.csv"), [{**meta, **row} for row in tail_records])
    print(f"wrote {len(tail_records)} tail nonzero samples and {len(summary)} summary rows under {prefix.parent}")


if __name__ == "__main__":
    main()
