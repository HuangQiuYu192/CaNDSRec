#!/usr/bin/env python3
"""Analyze full-sort positive target rank distribution for a RecBole checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed

from argument_parser import build_config_dict, parse_args
from models import get_model_class


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})


def summarize_ranks(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks, dtype=np.int64)
    out = {
        "n": int(len(ranks)),
        "mean_rank": float(ranks.mean()) if len(ranks) else 0.0,
        "median_rank": float(np.median(ranks)) if len(ranks) else 0.0,
        "p75_rank": float(np.percentile(ranks, 75)) if len(ranks) else 0.0,
        "p90_rank": float(np.percentile(ranks, 90)) if len(ranks) else 0.0,
        "p95_rank": float(np.percentile(ranks, 95)) if len(ranks) else 0.0,
        "p99_rank": float(np.percentile(ranks, 99)) if len(ranks) else 0.0,
        "mrr": float((1.0 / ranks).mean()) if len(ranks) else 0.0,
    }
    for k in [1, 3, 5, 10, 20, 50, 100]:
        out[f"hit@{k}"] = float((ranks <= k).mean()) if len(ranks) else 0.0
        out[f"ndcg@{k}"] = float(((ranks <= k) / np.log2(ranks + 1.0)).mean()) if len(ranks) else 0.0
    buckets = [
        ("rank_1", ranks == 1),
        ("rank_2_5", (ranks >= 2) & (ranks <= 5)),
        ("rank_6_10", (ranks >= 6) & (ranks <= 10)),
        ("rank_11_20", (ranks >= 11) & (ranks <= 20)),
        ("rank_21_50", (ranks >= 21) & (ranks <= 50)),
        ("rank_51_100", (ranks >= 51) & (ranks <= 100)),
        ("rank_gt100", ranks > 100),
    ]
    for name, mask in buckets:
        out[name] = int(mask.sum())
        out[f"{name}_rate"] = float(mask.mean()) if len(ranks) else 0.0
    return out


def collect_ranks(args, checkpoint_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model_class = get_model_class(args.model)
    config_dict = build_config_dict(args)
    config = Config(model=model_class, dataset=args.dataset, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], args.model)(config, model)

    checkpoint = torch.load(checkpoint_path, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    trainer.tot_item_num = test_data._dataset.item_num
    trainer.item_tensor = test_data._dataset.get_item_feature().to(config["device"])

    all_users = []
    all_items = []
    all_ranks = []
    uid_field = config["USER_ID_FIELD"]
    with torch.no_grad():
        for batched_data in test_data:
            interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
            positive_u = torch.as_tensor(positive_u, device=scores.device).long()
            positive_i = torch.as_tensor(positive_i, device=scores.device).long()
            pos_scores = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_scores.unsqueeze(1)).sum(dim=1) + 1
            users = interaction[uid_field][positive_u.detach().cpu()]
            all_users.extend(users.detach().cpu().numpy().tolist())
            all_items.extend(positive_i.detach().cpu().numpy().tolist())
            all_ranks.extend(ranks.detach().cpu().numpy().tolist())
    return (
        np.asarray(all_users, dtype=np.int64),
        np.asarray(all_items, dtype=np.int64),
        np.asarray(all_ranks, dtype=np.int64),
    )


def main() -> None:
    wrapper = argparse.ArgumentParser()
    wrapper.add_argument("--checkpoint", required=True)
    wrapper.add_argument("--label", required=True)
    wrapper.add_argument("--output_dir", required=True)
    known, remaining = wrapper.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()

    users, items, ranks = collect_ranks(args, known.checkpoint)
    output_dir = Path(known.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_ranks(ranks)
    summary.update(
        {
            "label": known.label,
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint": known.checkpoint,
            "max_item_list_length": args.max_item_list_length,
            "temperature": getattr(args, "temperature", None),
        }
    )
    with open(output_dir / f"{known.label}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / f"{known.label}.ranks.tsv", "w", encoding="utf-8") as f:
        f.write("user_id\titem_id\trank\n")
        for user, item, rank in zip(users, items, ranks):
            f.write(f"{user}\t{item}\t{rank}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
