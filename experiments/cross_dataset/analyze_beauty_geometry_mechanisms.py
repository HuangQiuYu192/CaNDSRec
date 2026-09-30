#!/usr/bin/env python3
"""Read-only mechanism diagnostics for a Beauty SASRec score checkpoint.

For one checkpoint this script jointly measures: (1) the fixed-representation
sequence-normalization ranking counterfactual; (2) representation/score-scale,
entropy and top-score-margin statistics; and (3) test metrics after grouping
targets by global training-popularity item tertiles.
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

from argument_parser import build_config_dict, parse_args
from models import get_model_class
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})


def build_model(args: argparse.Namespace):
    cli = [
        sys.argv[0], "--model", args.model, "--dataset", "Beauty", "--gpu_id", str(args.gpu_id),
        "--seed", str(args.seed), "--hidden_size", "256", "--n_layers", "2", "--n_heads", "2",
        "--inner_size", "1024", "--hidden_dropout_prob", "0.5", "--attn_dropout_prob", "0.5",
        "--learning_rate", "0.001", "--max_item_list_length", "50", "--train_batch_size", "1024",
        "--eval_batch_size", str(args.eval_batch_size), "--temperature", str(args.temperature),
        "--verbose", "False", "--show_progress", "False",
    ]
    if args.model == "GeometrySASRec":
        cli += [
            "--score_geometry", args.geometry,
            "--sequence_norm_power", str(args.sequence_norm_power),
            "--item_norm_power", str(args.item_norm_power),
        ]
    sys.argv = cli
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu_id)
    model_class = get_model_class(parsed.model)
    config = Config(model=model_class, dataset=parsed.dataset, config_dict=build_config_dict(parsed))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    checkpoint = torch.load(args.checkpoint, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    return config, dataset, train_data, test_data, model


def masked(scores: torch.Tensor, history_index) -> torch.Tensor:
    scores = scores.clone()
    scores[:, 0] = -float("inf")
    if history_index is not None:
        scores[history_index] = -float("inf")
    return scores


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    if not len(ranks):
        return math.nan, math.nan
    values = ranks.astype(np.float64)
    hit = values <= k
    return float(hit.mean()), float((hit / np.log2(values + 1.0)).mean())


def popularity_groups(targets: np.ndarray, popularity: np.ndarray) -> dict[str, np.ndarray]:
    active = np.flatnonzero(popularity > 0)
    active = active[active != 0]
    order = active[np.argsort(-popularity[active], kind="stable")]
    labels = np.full(len(popularity), -1, dtype=np.int8)
    for label, part in enumerate(np.array_split(order, 3)):
        labels[part] = label
    return {
        "all": np.arange(len(targets)),
        "head": np.flatnonzero(labels[targets] == 0),
        "mid": np.flatnonzero(labels[targets] == 1),
        "tail": np.flatnonzero(labels[targets] == 2),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["SASRec", "GeometrySASRec"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--geometry", default="dot")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--sequence_norm_power", default=0.0, type=float)
    parser.add_argument("--item_norm_power", default=0.0, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu_id", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    config, dataset, train_data, test_data, model = build_model(args)
    device = config["device"]
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    item_weight = model.item_embedding.weight.detach()
    item_norms = item_weight[1:].norm(dim=-1).cpu().numpy()

    target_items: list[int] = []
    ranks_all: list[int] = []
    sequence_norms: list[float] = []
    score_stds: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    rank_equal = topk_equal = sample_count = 0
    item_topk_jaccard_total = 0.0

    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            item_seq = interaction[model.ITEM_SEQ]
            item_seq_len = interaction[model.ITEM_SEQ_LEN]
            sequence = model.forward(item_seq, item_seq_len)

            # Actual trained geometry: score spread, predictive entropy and margin.
            actual = model.full_sort_predict(interaction)
            if actual.dim() == 1:
                actual = actual.view(positive_i.size(0), -1)
            score_stds.extend(actual[:, 1:].std(dim=1, unbiased=False).cpu().tolist())
            visible = masked(actual, history_index)
            probability = torch.softmax(visible, dim=1)
            entropies.extend((-(probability * torch.log(probability.clamp_min(1e-30))).sum(dim=1)).cpu().tolist())
            top_two = torch.topk(visible, k=2, dim=1).values
            margins.extend((top_two[:, 0] - top_two[:, 1]).cpu().tolist())
            positive_score = visible[positive_u, positive_i]
            ranks = (visible[positive_u] > positive_score.unsqueeze(1)).sum(dim=1) + 1
            target_items.extend(positive_i.cpu().tolist())
            ranks_all.extend(ranks.cpu().tolist())
            sequence_norms.extend(sequence.norm(dim=-1).cpu().tolist())

            # Fixed-representation counterfactuals.  Sequence normalization is
            # candidate-common positive scaling, whereas item normalization can
            # alter candidate order.  Both use identical masks.
            raw = masked(sequence @ item_weight.transpose(0, 1), history_index)
            sequence_only = masked(F.normalize(sequence, dim=-1) @ item_weight.transpose(0, 1), history_index)
            item_only = masked(sequence @ F.normalize(item_weight, dim=-1).transpose(0, 1), history_index)
            raw_pos = raw[positive_u, positive_i]
            seq_pos = sequence_only[positive_u, positive_i]
            raw_rank = (raw[positive_u] > raw_pos.unsqueeze(1)).sum(dim=1) + 1
            seq_rank = (sequence_only[positive_u] > seq_pos.unsqueeze(1)).sum(dim=1) + 1
            rank_equal += int((raw_rank == seq_rank).sum().item())
            raw_topk = torch.topk(raw, k=args.topk, dim=1).indices
            seq_topk = torch.topk(sequence_only, k=args.topk, dim=1).indices
            item_topk = torch.topk(item_only, k=args.topk, dim=1).indices
            topk_equal += int((raw_topk == seq_topk).all(dim=1).sum().item())
            item_topk_jaccard_total += float(sum(
                len(set(a.tolist()).intersection(b.tolist())) / args.topk
                for a, b in zip(raw_topk, item_topk)
            ))
            sample_count += positive_i.numel()

    targets = np.asarray(target_items, dtype=np.int64)
    ranks_array = np.asarray(ranks_all, dtype=np.int64)
    diagnostics = {
        "tag": args.tag, "seed": args.seed, "model": args.model, "geometry": args.geometry,
        "temperature": args.temperature, "sequence_norm_power": args.sequence_norm_power,
        "item_norm_power": args.item_norm_power, "checkpoint": args.checkpoint, "n_test": sample_count,
        "sequence_norm_mean": mean_sd(sequence_norms)[0], "sequence_norm_sd": mean_sd(sequence_norms)[1],
        "sequence_norm_cv": mean_sd(sequence_norms)[1] / max(mean_sd(sequence_norms)[0], 1e-12),
        "item_norm_mean": float(item_norms.mean()), "item_norm_sd": float(item_norms.std()),
        "item_norm_cv": float(item_norms.std() / max(item_norms.mean(), 1e-12)),
        "score_std_mean": mean_sd(score_stds)[0], "score_std_sd": mean_sd(score_stds)[1],
        "predictive_entropy_mean": mean_sd(entropies)[0], "predictive_entropy_sd": mean_sd(entropies)[1],
        "top1_top2_margin_mean": mean_sd(margins)[0], "top1_top2_margin_sd": mean_sd(margins)[1],
        "counterfactual_seq_rank_equal_rate": rank_equal / max(sample_count, 1),
        "counterfactual_seq_topk_equal_rate": topk_equal / max(sample_count, 1),
        "counterfactual_item_topk_jaccard": item_topk_jaccard_total / max(sample_count, 1),
    }

    group_rows = []
    for group, idx in popularity_groups(targets, popularity).items():
        row = {"tag": args.tag, "seed": args.seed, "geometry": args.geometry, "temperature": args.temperature,
               "group": group, "n": int(len(idx)), "mean_rank": float(ranks_array[idx].mean()) if len(idx) else math.nan,
               "median_rank": float(np.median(ranks_array[idx])) if len(idx) else math.nan}
        for cutoff in (5, 10, 20, 50):
            recall, ndcg = metric_at(ranks_array[idx], cutoff)
            row[f"recall@{cutoff}"] = recall
            row[f"ndcg@{cutoff}"] = ndcg
        group_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    with (args.out_dir / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)
    (args.out_dir / "README.md").write_text(
        "# Geometry mechanism diagnostic\n\n"
        "`counterfactual_seq_*` compares raw dot scores with sequence-normalized scores from the same frozen representation. "
        "`counterfactual_item_topk_jaccard` compares raw dot with item-normalized scores; lower values indicate changed candidate order. "
        "Groups are global item-popularity tertiles computed only from training interactions.\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
