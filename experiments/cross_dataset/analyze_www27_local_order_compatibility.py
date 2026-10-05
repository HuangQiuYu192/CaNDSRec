#!/usr/bin/env python3
"""Measure whether dot and angular rankings agree in the local candidate set."""
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--local_k", type=int, default=50)
    args = parser.parse_args()
    args.hidden_size = 256; args.inner_size = 1024; args.max_item_list_length = 50
    args.eval_batch_size = 512; args.train_batch_size = 1024; args.n_layers = 2; args.n_heads = 2
    args.hidden_dropout_prob = .5; args.attn_dropout_prob = .5; args.learning_rate = .001
    args.temperature = 10.; args.score_geometry = "both"; args.sequence_norm_power = 1.; args.item_norm_power = 1.

    model_name = "SASRec" if args.variant == "dot" else "GeometrySASRec"
    _, _, _, _, test_data, model = build_model(
        model_name, checkpoint(args.checkpoint_root, args.dataset, args.seed, args.variant), args
    )
    device = next(model.parameters()).device
    total = 0
    sums = {"top10_jaccard": 0.0, "top20_jaccard": 0.0, "top50_inversion_rate": 0.0, "dot_minus_angular_top10_log_norm": 0.0}
    with torch.no_grad():
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            items = model.item_embedding.weight
            dot = sequence @ items.T
            angular = F.normalize(sequence, dim=-1) @ F.normalize(items, dim=-1).T
            for scores in (dot, angular):
                scores[:, 0] = -torch.inf
                if history is not None:
                    scores[history] = -torch.inf

            k = min(args.local_k, dot.size(1) - 1)
            dot_ids = dot.topk(k, dim=1).indices
            angular_ids = angular.topk(k, dim=1).indices
            local_angular = angular.gather(1, dot_ids)
            # In dot-order i<j, an inversion occurs if candidate j has higher angular score.
            pairwise_inversion = local_angular.unsqueeze(2) < local_angular.unsqueeze(1)
            upper = torch.triu(torch.ones(k, k, dtype=torch.bool, device=device), diagonal=1)
            inversion_rate = pairwise_inversion[:, upper].float().mean(1)
            log_norm = torch.log(items.norm(dim=1).clamp_min(1e-12))

            batch_size = dot.size(0)
            for cutoff in (10, 20):
                dot_set, angular_set = dot_ids[:, :cutoff], angular_ids[:, :cutoff]
                matches = (dot_set.unsqueeze(2) == angular_set.unsqueeze(1)).any(2).sum(1).float()
                sums[f"top{cutoff}_jaccard"] += (matches / cutoff).sum().item()
            sums["top50_inversion_rate"] += inversion_rate.sum().item()
            sums["dot_minus_angular_top10_log_norm"] += (
                log_norm[dot_ids[:, :10]].mean(1) - log_norm[angular_ids[:, :10]].mean(1)
            ).sum().item()
            total += batch_size

    row = {"dataset": args.dataset, "seed": args.seed, "trained_geometry": args.variant, "n": total}
    row.update({key: value / total for key, value in sums.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader(); writer.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
