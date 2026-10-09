#!/usr/bin/env python3
"""Read-only sequence-scale strata diagnostics for BSARec score heads.

For each completed checkpoint, test examples are partitioned by quintiles of
the *unnormalized* sequence norm produced by its encoder.  The script then
reports the actual deployed logit dispersion, entropy, target cross-entropy,
and ranking metrics within each stratum.  It never trains or selects a model.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

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
    cli = [
        sys.argv[0], "--model", args.model, "--dataset", args.dataset,
        "--gpu_id", str(args.gpu_id), "--seed", str(args.seed),
        "--hidden_size", "256", "--n_layers", "2", "--n_heads", "2",
        "--inner_size", "1024", "--alpha", "0.5", "--c", "9",
        "--learning_rate", "0.001", "--weight_decay", "0",
        "--max_item_list_length", "50", "--eval_batch_size", str(args.eval_batch_size),
        "--train_batch_size", "1024", "--verbose", "False", "--show_progress", "False",
    ]
    if args.model != "BSARec":
        cli += ["--temperature", str(args.temperature)]
    if args.model in {"GeometryBSARec", "StopGradSequenceGeometryBSARec"}:
        cli += ["--score_geometry", "sequence"]
    sys.argv = cli
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu_id)
    cls = get_model_class(parsed.model)
    config = Config(model=cls, dataset=parsed.dataset, config_dict=build_config_dict(parsed))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)
    model = cls(config, test_data.dataset).to(config["device"])
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


def bins_for(norms: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Quantile cut points are fixed before collecting any score statistics.
    cuts = torch.quantile(norms, torch.linspace(0, 1, groups + 1, device=norms.device)[1:-1])
    return torch.bucketize(norms, cuts, right=True), cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("BSARec", "ScaledDotBSARec", "GeometryBSARec", "StopGradSequenceGeometryBSARec", "CANDSBSARec"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    test_data, model = build_model(args)
    device = next(model.parameters()).device

    # Pass 1: define strata solely from raw sequence radii.
    all_norms = []
    with torch.no_grad():
        for batched in test_data:
            interaction = batched[0].to(device)
            users = torch.as_tensor(batched[2], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            all_norms.append(sequence[users].norm(dim=1))
    all_norms = torch.cat(all_norms)
    _, cuts = bins_for(all_norms, args.groups)

    totals = [{"n": 0, "norm": 0.0, "logit_std": 0.0, "entropy": 0.0, "ce": 0.0, "recall10": 0.0, "ndcg10": 0.0} for _ in range(args.groups)]
    with torch.no_grad():
        for batched in test_data:
            interaction, history = batched[0].to(device), batched[1]
            users = torch.as_tensor(batched[2], device=device).long()
            positives = torch.as_tensor(batched[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            raw_norm = sequence[users].norm(dim=1)
            group = torch.bucketize(raw_norm, cuts, right=True)
            scores = masked(model.full_sort_predict(interaction), history, users)
            finite = torch.isfinite(scores)
            count = finite.sum(dim=1).clamp_min(1)
            clean = torch.where(finite, scores, torch.zeros_like(scores))
            mean = clean.sum(dim=1) / count
            std = (((clean - mean.unsqueeze(1)).square() * finite).sum(dim=1) / count).sqrt()
            log_prob = torch.log_softmax(scores, dim=1)
            prob = log_prob.exp()
            entropy = -(prob * log_prob.masked_fill(~finite, 0)).sum(dim=1)
            ce = -log_prob[torch.arange(positives.numel(), device=device), positives]
            pos = scores[torch.arange(positives.numel(), device=device), positives]
            rank = (scores > pos.unsqueeze(1)).sum(dim=1).float() + 1
            for index in range(args.groups):
                take = group == index
                if not take.any():
                    continue
                total = totals[index]
                total["n"] += int(take.sum().item())
                total["norm"] += raw_norm[take].sum().item()
                total["logit_std"] += std[take].sum().item()
                total["entropy"] += entropy[take].sum().item()
                total["ce"] += ce[take].sum().item()
                total["recall10"] += (rank[take] <= 10).float().sum().item()
                total["ndcg10"] += ((rank[take] <= 10).float() / torch.log2(rank[take] + 1)).sum().item()

    rows = []
    for index, total in enumerate(totals, start=1):
        n = total["n"]
        rows.append({
            "dataset": args.dataset, "seed": args.seed, "variant": args.variant,
            "model": args.model, "temperature": args.temperature, "stratum": f"Q{index}", "n": n,
            "raw_sequence_norm_mean": total["norm"] / n,
            "logit_std_mean": total["logit_std"] / n,
            "entropy_mean": total["entropy"] / n,
            "target_cross_entropy_mean": total["ce"] / n,
            "recall@10": total["recall10"] / n,
            "ndcg@10": total["ndcg10"] / n,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
