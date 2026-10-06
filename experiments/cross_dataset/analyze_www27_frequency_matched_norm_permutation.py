#!/usr/bin/env python3
"""Frequency-exact norm permutation control for dot versus angular ranking."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_group_metrics import build_model


METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")


def checkpoint(root: Path, dataset: str, seed: int, variant: str) -> str:
    files = sorted((root / dataset / f"seed{seed}" / variant).glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"Missing checkpoint: {dataset}, seed={seed}, {variant}")
    return str(files[-1])


def frequency_exact_permutation(norm: np.ndarray, popularity: np.ndarray, seed: int) -> np.ndarray:
    """Permute norms only among items with exactly equal train interaction count."""
    result = norm.copy()
    rng = np.random.default_rng(seed)
    for count in np.unique(popularity[1:]):
        indices = np.flatnonzero(popularity == count)
        indices = indices[indices != 0]
        if len(indices) > 1:
            result[indices] = norm[rng.permutation(indices)]
    return result


def add_metrics(accumulator: dict[str, float], ranks: torch.Tensor) -> None:
    accumulator["n"] += len(ranks)
    for cutoff in (10, 20):
        hit = (ranks <= cutoff).float()
        accumulator[f"recall@{cutoff}"] += hit.sum().item()
        accumulator[f"ndcg@{cutoff}"] += (hit / torch.log2(ranks + 1)).sum().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("dot", "joint"), required=True)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=3)
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
    observed_norm = model.item_embedding.weight.detach().norm(dim=1).cpu().numpy()
    permuted_norms = [torch.as_tensor(frequency_exact_permutation(observed_norm, popularity, args.seed * 100 + rep), device=device) for rep in range(args.permutations)]
    totals = {name: {"n": 0., **{metric: 0. for metric in METRICS}} for name in ("angular", "observed_radial", "frequency_matched_permuted_radial")}

    with torch.no_grad():
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            items = model.item_embedding.weight
            angular = F.normalize(sequence, dim=-1) @ F.normalize(items, dim=-1).T
            score_sets = {
                "angular": [angular],
                "observed_radial": [angular * items.norm(dim=1).clamp_min(1e-12).unsqueeze(0)],
                "frequency_matched_permuted_radial": [angular * norm.clamp_min(1e-12).unsqueeze(0) for norm in permuted_norms],
            }
            for name, scores_list in score_sets.items():
                for scores in scores_list:
                    scores = scores.clone()
                    scores[:, 0] = -torch.inf
                    if history is not None:
                        scores[history] = -torch.inf
                    rank = (scores[users] > scores[users, positives].unsqueeze(1)).sum(1).float() + 1
                    add_metrics(totals[name], rank)

    rows = []
    for name, values in totals.items():
        divisor = values["n"]
        # The permuted variant evaluates three full test passes; average them.
        if name == "frequency_matched_permuted_radial":
            divisor /= args.permutations
        row = {"dataset": args.dataset, "seed": args.seed, "trained_geometry": args.variant, "score_variant": name, "n": int(divisor)}
        row.update({metric: values[metric] / values["n"] for metric in METRICS})
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
