#!/usr/bin/env python3
"""Test whether CaNDS gains concentrate on radial-displacement examples.

The subgroup label is computed only from the dot-product SASRec checkpoint:
the target must be angularly preferred to the strongest dot-product competitor,
both cosines must be positive, and the competitor must still outrank it under
the dot product.  CaNDS is then evaluated on the same examples and catalogue.
This avoids defining the subgroup using the proposed model's outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_group_metrics import build_model, metric_at


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--inner_size", type=int, default=1024)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_item_list_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--out", required=True)


def masked_scores(model, interaction, history_index, positive_u, positive_i):
    scores = model.full_sort_predict(interaction)
    scores[:, 0] = -torch.inf
    if history_index is not None:
        scores[history_index] = -torch.inf
    selected = scores[positive_u]
    target_scores = selected[torch.arange(len(positive_i), device=scores.device), positive_i]
    ranks = (selected > target_scores.unsqueeze(1)).sum(dim=1) + 1
    return scores, ranks


def collect(base_model, cands_model, test_data):
    device = next(base_model.parameters()).device
    item = base_model.item_embedding.weight.detach()
    item_dir = F.normalize(item, dim=-1)
    rows = {"base_rank": [], "cands_rank": [], "displaced": []}
    with torch.no_grad():
        for batch in test_data:
            interaction, history_index, positive_u, positive_i = batch
            interaction = interaction.to(device)
            positive_u = torch.as_tensor(positive_u, device=device).long()
            positive_i = torch.as_tensor(positive_i, device=device).long()
            base_scores, base_rank = masked_scores(base_model, interaction, history_index, positive_u, positive_i)
            _, cands_rank = masked_scores(cands_model, interaction, history_index, positive_u, positive_i)

            seq = base_model.forward(interaction[base_model.ITEM_SEQ], interaction[base_model.ITEM_SEQ_LEN])
            cosine = F.normalize(seq, dim=-1) @ item_dir.T
            cosine[:, 0] = -torch.inf
            if history_index is not None:
                cosine[history_index] = -torch.inf
            chosen_dot = base_scores[positive_u].clone()
            chosen_dot[torch.arange(len(positive_i), device=device), positive_i] = -torch.inf
            competitor = chosen_dot.argmax(dim=1)
            target_cos = cosine[positive_u, positive_i]
            comp_cos = cosine[positive_u, competitor]
            target_dot = base_scores[positive_u, positive_i]
            comp_dot = base_scores[positive_u, competitor]
            displaced = (target_cos > 0) & (comp_cos > 0) & (target_cos > comp_cos) & (target_dot < comp_dot)
            rows["base_rank"].append(base_rank.cpu().numpy())
            rows["cands_rank"].append(cands_rank.cpu().numpy())
            rows["displaced"].append(displaced.cpu().numpy())
    return {key: np.concatenate(value) for key, value in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    _, _, _, _, test_data, base = build_model("SASRec", args.base_checkpoint, args)
    _, _, _, _, _, cands = build_model("CANDSSASRec", args.cands_checkpoint, args)
    values = collect(base, cands, test_data)
    masks = {
        "all": np.ones(len(values["base_rank"]), dtype=bool),
        "radial_displaced": values["displaced"],
        "not_radial_displaced": ~values["displaced"],
    }
    rows = []
    for group, mask in masks.items():
        row = {"dataset": args.dataset, "seed": args.seed, "group": group, "n": int(mask.sum())}
        for k in (10, 20, 50):
            base_recall, base_ndcg = metric_at(values["base_rank"][mask], k)
            cands_recall, cands_ndcg = metric_at(values["cands_rank"][mask], k)
            row.update({
                f"base_recall@{k}": base_recall, f"cands_recall@{k}": cands_recall,
                f"delta_recall@{k}": cands_recall - base_recall,
                f"base_ndcg@{k}": base_ndcg, f"cands_ndcg@{k}": cands_ndcg,
                f"delta_ndcg@{k}": cands_ndcg - base_ndcg,
            })
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Radial-displacement subgroup analysis", "", "| group | n | base NDCG@10 | CaNDS NDCG@10 | delta |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['n']} | {row['base_ndcg@10']:.4f} | {row['cands_ndcg@10']:.4f} | {row['delta_ndcg@10']:+.4f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
