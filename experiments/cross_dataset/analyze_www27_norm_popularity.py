#!/usr/bin/env python3
"""Test whether learned item norms are reducible to training interaction frequency."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

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
    args = parser.parse_args()
    args.hidden_size = 256; args.inner_size = 1024; args.max_item_list_length = 50
    args.eval_batch_size = 512; args.train_batch_size = 1024; args.n_layers = 2; args.n_heads = 2
    args.hidden_dropout_prob = .5; args.attn_dropout_prob = .5; args.learning_rate = .001
    args.temperature = 10.; args.score_geometry = "both"; args.sequence_norm_power = 1.; args.item_norm_power = 1.

    model_name = "SASRec" if args.variant == "dot" else "GeometrySASRec"
    _, dataset, train_data, _, test_data, model = build_model(
        model_name, checkpoint(args.checkpoint_root, args.dataset, args.seed, args.variant), args
    )
    device = next(model.parameters()).device
    item_field = dataset.iid_field
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    log_popularity = np.log1p(popularity.astype(np.float64))
    item_norm_cpu = model.item_embedding.weight.detach().norm(dim=1).cpu().numpy()
    active = popularity > 0
    rho = float(spearmanr(log_popularity[active], item_norm_cpu[active]).statistic)
    pearson = float(np.corrcoef(log_popularity[active], item_norm_cpu[active])[0, 1])

    item_norm = model.item_embedding.weight.norm(dim=1)
    log_norm = torch.log(item_norm.clamp_min(1e-12))
    log_popularity_t = torch.as_tensor(log_popularity, device=device, dtype=log_norm.dtype)
    pair_sums = {"recovered_count": 0., "recovered_log_pop_gap": 0., "recovered_log_norm_gap": 0., "introduced_count": 0., "introduced_log_pop_gap": 0., "introduced_log_norm_gap": 0.}
    with torch.no_grad():
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            items = model.item_embedding.weight
            dot = sequence @ items.T
            angular = F.normalize(sequence, dim=-1) @ F.normalize(items, dim=-1).T
            for scores in (dot, angular):
                scores[:, 0] = -torch.inf
                if history is not None:
                    scores[history] = -torch.inf
            user_dot, user_angular = dot[users], angular[users]
            pos_dot = user_dot[torch.arange(len(users), device=device), positives]
            pos_angular = user_angular[torch.arange(len(users), device=device), positives]
            recovered = (user_dot > pos_dot.unsqueeze(1)) & (user_angular < pos_angular.unsqueeze(1))
            introduced = (user_dot < pos_dot.unsqueeze(1)) & (user_angular > pos_angular.unsqueeze(1))
            pop_gap = log_popularity_t[positives].unsqueeze(1) - log_popularity_t.unsqueeze(0)
            norm_gap = log_norm[positives].unsqueeze(1) - log_norm.unsqueeze(0)
            for prefix, mask in (("recovered", recovered), ("introduced", introduced)):
                pair_sums[f"{prefix}_count"] += mask.sum().item()
                pair_sums[f"{prefix}_log_pop_gap"] += torch.where(mask, pop_gap, torch.zeros_like(pop_gap)).sum().item()
                pair_sums[f"{prefix}_log_norm_gap"] += torch.where(mask, norm_gap, torch.zeros_like(norm_gap)).sum().item()

    row = {
        "dataset": args.dataset, "seed": args.seed, "trained_geometry": args.variant,
        "active_items": int(active.sum()), "norm_logpop_spearman": rho, "norm_logpop_pearson": pearson,
    }
    for prefix in ("recovered", "introduced"):
        count = pair_sums[f"{prefix}_count"]
        row[f"{prefix}_pair_count"] = int(count)
        row[f"{prefix}_mean_log_pop_gap"] = pair_sums[f"{prefix}_log_pop_gap"] / max(count, 1.)
        row[f"{prefix}_mean_log_norm_gap"] = pair_sums[f"{prefix}_log_norm_gap"] / max(count, 1.)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader(); writer.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
