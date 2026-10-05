#!/usr/bin/env python3
"""Fixed-checkpoint intervention: continuously reintroduce item radial scale."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_group_metrics import build_model


def checkpoint(root: Path, dataset: str, seed: int, variant: str) -> str:
    files = sorted((root / dataset / f"seed{seed}" / variant).glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"Missing checkpoint: {dataset}, seed={seed}, {variant}")
    return str(files[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("dot", "joint"), required=True)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    args.hidden_size = 256; args.inner_size = 1024; args.max_item_list_length = 50
    args.eval_batch_size = 512; args.train_batch_size = 1024; args.n_layers = 2; args.n_heads = 2
    args.hidden_dropout_prob = .5; args.attn_dropout_prob = .5; args.learning_rate = .001
    args.temperature = 10.; args.score_geometry = "both"; args.sequence_norm_power = 1.; args.item_norm_power = 1.

    model_name = "SASRec" if args.variant == "dot" else "GeometrySASRec"
    _, _, _, _, test_data, model = build_model(
        model_name, checkpoint(args.checkpoint_root, args.dataset, args.seed, args.variant), args
    )
    device = next(model.parameters()).device
    totals = {alpha: {"n": 0, "hit10": 0., "ndcg10": 0., "hit20": 0., "ndcg20": 0., "rank": 0.} for alpha in alphas}
    with torch.no_grad():
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            items = model.item_embedding.weight
            angular = F.normalize(sequence, dim=-1) @ F.normalize(items, dim=-1).T
            item_norm = items.norm(dim=1).clamp_min(1e-12)
            for alpha in alphas:
                # Sequence norm is candidate-common and therefore irrelevant to ranks.
                scores = angular * item_norm.pow(alpha).unsqueeze(0)
                scores[:, 0] = -torch.inf
                if history is not None:
                    scores[history] = -torch.inf
                rank = (scores[users] > scores[users, positives].unsqueeze(1)).sum(1).float() + 1
                record = totals[alpha]
                record["n"] += len(rank)
                record["hit10"] += (rank <= 10).sum().item()
                record["ndcg10"] += ((rank <= 10).float() / torch.log2(rank + 1)).sum().item()
                record["hit20"] += (rank <= 20).sum().item()
                record["ndcg20"] += ((rank <= 20).float() / torch.log2(rank + 1)).sum().item()
                record["rank"] += rank.sum().item()

    rows = []
    for alpha in alphas:
        value = totals[alpha]
        n = value["n"]
        rows.append({
            "dataset": args.dataset, "seed": args.seed, "trained_geometry": args.variant,
            "item_radial_exponent": alpha, "n": n,
            "recall@10": value["hit10"] / n, "ndcg@10": value["ndcg10"] / n,
            "recall@20": value["hit20"] / n, "ndcg@20": value["ndcg20"] / n,
            "mean_rank": value["rank"] / n,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
