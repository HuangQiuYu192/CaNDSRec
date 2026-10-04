#!/usr/bin/env python3
"""Decompose frozen dot-to-angular ranking changes by the decisive competitor."""
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


def transition(dot_rank: int, angular_rank: int) -> str:
    if dot_rank > 10 and angular_rank <= 10:
        return "gain_top10"
    if dot_rank <= 10 and angular_rank > 10:
        return "loss_top10"
    return "stable_top10" if dot_rank <= 10 else "outside_top10"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("dot", "joint"), required=True)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
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
    rows: list[dict] = []
    with torch.no_grad():
        for batched in test_data:
            interaction = batched[0].to(device)
            history_index = batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            items = model.item_embedding.weight
            dot = sequence @ items.T
            angular = F.normalize(sequence, dim=-1) @ F.normalize(items, dim=-1).T
            for scores in (dot, angular):
                scores[:, 0] = -torch.inf
                if history_index is not None:
                    scores[history_index] = -torch.inf

            dot_rank = (dot[users] > dot[users, positives].unsqueeze(1)).sum(1) + 1
            angular_rank = (angular[users] > angular[users, positives].unsqueeze(1)).sum(1) + 1
            user_dot, user_angular = dot[users], angular[users]
            positive_dot = user_dot[torch.arange(len(users), device=device), positives]
            positive_angular = user_angular[torch.arange(len(users), device=device), positives]
            log_item_norm = torch.log(items.norm(dim=1).clamp_min(1e-12))

            # Pairwise reversals are the direct source of a rank change.  A recovered
            # competitor outranks the target by dot product but not by angle.
            recovered = (user_dot > positive_dot.unsqueeze(1)) & (user_angular < positive_angular.unsqueeze(1))
            introduced = (user_dot < positive_dot.unsqueeze(1)) & (user_angular > positive_angular.unsqueeze(1))
            recovered_count = recovered.sum(1)
            introduced_count = introduced.sum(1)
            angular_gap = positive_angular.unsqueeze(1) - user_angular
            norm_gap = log_item_norm[positives].unsqueeze(1) - log_item_norm.unsqueeze(0)
            recovered_margin = (angular_gap * recovered).sum(1) / recovered_count.clamp_min(1)
            recovered_norm_ratio = (norm_gap * recovered).sum(1) / recovered_count.clamp_min(1)
            introduced_disadvantage = ((-angular_gap) * introduced).sum(1) / introduced_count.clamp_min(1)
            introduced_norm_ratio = (norm_gap * introduced).sum(1) / introduced_count.clamp_min(1)
            dot_for_competitor = dot[users].clone()
            dot_for_competitor[torch.arange(len(users), device=device), positives] = -torch.inf
            competitor = dot_for_competitor.argmax(1)
            unit_sequence = F.normalize(sequence[users], dim=-1)
            unit_items = F.normalize(items, dim=-1)
            angular_margin = (unit_sequence * unit_items[positives]).sum(1) - (unit_sequence * unit_items[competitor]).sum(1)
            log_norm_ratio = torch.log(items[positives].norm(dim=1).clamp_min(1e-12)) - torch.log(items[competitor].norm(dim=1).clamp_min(1e-12))

            for d_rank, a_rank, margin, ratio, rec_n, rec_margin, rec_ratio, add_n, add_margin, add_ratio in zip(
                dot_rank.tolist(), angular_rank.tolist(), angular_margin.tolist(), log_norm_ratio.tolist(),
                recovered_count.tolist(), recovered_margin.tolist(), recovered_norm_ratio.tolist(),
                introduced_count.tolist(), introduced_disadvantage.tolist(), introduced_norm_ratio.tolist(),
            ):
                rows.append({
                    "dataset": args.dataset, "seed": args.seed, "trained_geometry": args.variant,
                    "transition": transition(d_rank, a_rank), "dot_rank": d_rank, "angular_rank": a_rank,
                    "rank_delta": d_rank - a_rank, "angular_margin_vs_dot_competitor": margin,
                    "log_target_over_competitor_norm": ratio,
                    "recovered_competitors": rec_n, "recovered_angular_margin": rec_margin,
                    "recovered_log_norm_ratio": rec_ratio, "introduced_competitors": add_n,
                    "introduced_angular_disadvantage": add_margin, "introduced_log_norm_ratio": add_ratio,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
