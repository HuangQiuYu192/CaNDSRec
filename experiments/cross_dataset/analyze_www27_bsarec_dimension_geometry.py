#!/usr/bin/env python3
"""Frozen score-geometry diagnostics across BSARec representation dimensions.

This is deliberately a checkpoint analysis: it does not train, select a
dimension, or select a temperature on test data.  For a fixed BSARec or
CANDSBSARec checkpoint it measures the angular target--hard-negative margin,
dot--angular local-order disagreement, item-norm residuals, and the effect of
reintroducing candidate radius ``||e_i||**alpha`` into angular scoring.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argument_parser import build_config_dict, parse_args
from models import get_model_class
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})


def build_model(args: argparse.Namespace):
    """Rebuild the exact BSARec data/model configuration for a checkpoint."""
    cli = [
        sys.argv[0], "--model", args.model, "--dataset", args.dataset,
        "--gpu_id", str(args.gpu_id), "--seed", str(args.seed),
        "--hidden_size", str(args.dimension), "--n_layers", "2", "--n_heads", "2",
        "--inner_size", str(4 * args.dimension), "--alpha", "0.5", "--c", "9",
        "--learning_rate", "0.001", "--weight_decay", "0",
        "--max_item_list_length", "50", "--eval_batch_size", str(args.eval_batch_size),
        "--train_batch_size", "1024", "--verbose", "False", "--show_progress", "False",
    ]
    if args.model == "CANDSBSARec":
        cli += ["--temperature", "10"]
    sys.argv = cli
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu_id)
    model_class = get_model_class(parsed.model)
    config = Config(model=model_class, dataset=parsed.dataset, config_dict=build_config_dict(parsed))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)
    model = model_class(config, test_data.dataset).to(config["device"])
    checkpoint = torch.load(args.checkpoint, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    return test_data, model


def masked(scores: torch.Tensor, history, users: torch.Tensor) -> torch.Tensor:
    scores = scores.clone()
    scores[:, 0] = -torch.inf
    if history is not None:
        scores[history] = -torch.inf
    return scores[users]


def ranks_and_metrics(scores: torch.Tensor, positives: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    pos_scores = scores[torch.arange(len(positives), device=scores.device), positives]
    rank = (scores > pos_scores.unsqueeze(1)).sum(dim=1).float() + 1
    recall10 = (rank <= 10).float().sum().item()
    ndcg10 = ((rank <= 10).float() / torch.log2(rank + 1)).sum().item()
    return rank, recall10, ndcg10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("BSARec", "CANDSBSARec"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--local_k", type=int, default=50)
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    test_data, model = build_model(args)
    device = next(model.parameters()).device

    totals = {"n": 0, "angular_margin": 0.0, "pos_log_norm_minus_hard_neg": 0.0,
              "top10_jaccard": 0.0, "local_inversion": 0.0, "local_pairs": 0}
    radial = {alpha: {"recall10": 0.0, "ndcg10": 0.0} for alpha in alphas}
    with torch.no_grad():
        items = model.item_embedding.weight
        item_norm = items.norm(dim=1).clamp_min(1e-12)
        unit_items = F.normalize(items, dim=-1)
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            angular_all = F.normalize(sequence, dim=-1) @ unit_items.T
            dot_all = sequence @ items.T
            angular = masked(angular_all, history, users)
            dot = masked(dot_all, history, users)
            batch = positives.numel()
            row_index = torch.arange(batch, device=device)

            # Directional target margin against its hardest angular competitor.
            angular_without_positive = angular.clone()
            angular_without_positive[row_index, positives] = -torch.inf
            hard_neg = angular_without_positive.argmax(dim=1)
            margin = angular[row_index, positives] - angular[row_index, hard_neg]
            totals["angular_margin"] += margin.sum().item()
            totals["pos_log_norm_minus_hard_neg"] += (
                item_norm[positives].log() - item_norm[hard_neg].log()
            ).sum().item()

            # Candidate-list compatibility and pairwise disagreement within the
            # angular local candidate set.  This evaluates order, not metrics.
            k10 = min(10, angular.size(1))
            klocal = min(args.local_k, angular.size(1))
            a10 = angular.topk(k10, dim=1).indices
            d10 = dot.topk(k10, dim=1).indices
            overlap = (a10.unsqueeze(2) == d10.unsqueeze(1)).any(dim=2).sum(dim=1).float()
            totals["top10_jaccard"] += (overlap / (2 * k10 - overlap)).sum().item()
            local = angular.topk(klocal, dim=1).indices
            local_a = angular.gather(1, local)
            local_d = dot.gather(1, local)
            da = local_a.unsqueeze(2) - local_a.unsqueeze(1)
            dd = local_d.unsqueeze(2) - local_d.unsqueeze(1)
            tri = torch.triu(torch.ones((klocal, klocal), device=device, dtype=torch.bool), diagonal=1)
            disagree = ((da * dd) < 0)[:, tri].sum().item()
            totals["local_inversion"] += disagree
            totals["local_pairs"] += batch * (klocal * (klocal - 1) // 2)

            for alpha in alphas:
                score = angular * item_norm.pow(alpha).unsqueeze(0)
                _, hit, ndcg = ranks_and_metrics(score, positives)
                radial[alpha]["recall10"] += hit
                radial[alpha]["ndcg10"] += ndcg
            totals["n"] += batch

    n = totals["n"]
    row = {
        "dataset": args.dataset, "seed": args.seed, "model": args.model,
        "dimension": args.dimension, "n": n, "item_log_norm_std": float(item_norm.log()[1:].std().item()),
        "angular_margin": totals["angular_margin"] / n,
        "pos_log_norm_minus_hard_neg": totals["pos_log_norm_minus_hard_neg"] / n,
        "top10_jaccard_dot_vs_angular": totals["top10_jaccard"] / n,
        "local_inversion_rate": totals["local_inversion"] / totals["local_pairs"],
    }
    for alpha, values in radial.items():
        label = f"alpha_{alpha:g}".replace(".", "p")
        row[f"recall10_{label}"] = values["recall10"] / n
        row[f"ndcg10_{label}"] = values["ndcg10"] / n
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
