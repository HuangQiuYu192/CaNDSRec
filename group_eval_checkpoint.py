#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed

from argument_parser import build_config_dict, parse_args
from models import get_model_class


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})


def metric_from_ranks(ranks, topk):
    out = {}
    ranks = np.asarray(ranks)
    for k in topk:
        hit = ranks <= k
        out[f"recall@{k}"] = float(hit.mean()) if len(ranks) else 0.0
        ndcg = hit / np.log2(ranks + 1.0)
        out[f"ndcg@{k}"] = float(ndcg.mean()) if len(ranks) else 0.0
    return out


def split_by_test_target_pop(items, pop):
    order = np.argsort(-pop[items], kind="stable")
    groups = {}
    names = ["head", "mid", "tail"]
    for name, idx in zip(names, np.array_split(order, 3)):
        groups[name] = idx
    return groups


def main():
    wrapper = argparse.ArgumentParser()
    wrapper.add_argument("--checkpoint", required=True)
    known, remaining = wrapper.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    model_class = get_model_class(args.model)
    config_dict = build_config_dict(args)
    config = Config(model=model_class, dataset=args.dataset, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], args.model)(config, model)

    checkpoint = torch.load(known.checkpoint, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    trainer.tot_item_num = test_data._dataset.item_num
    trainer.item_tensor = test_data._dataset.get_item_feature().to(config["device"])

    pop = np.bincount(
        train_data.dataset.inter_feat[config["ITEM_ID_FIELD"]].cpu().numpy(),
        minlength=dataset.item_num,
    )

    all_items = []
    all_ranks = []
    with torch.no_grad():
        for batched_data in test_data:
            interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
            positive_u = torch.as_tensor(positive_u, device=scores.device).long()
            positive_i = torch.as_tensor(positive_i, device=scores.device).long()
            pos_scores = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_scores.unsqueeze(1)).sum(dim=1) + 1
            all_items.extend(positive_i.detach().cpu().numpy().tolist())
            all_ranks.extend(ranks.detach().cpu().numpy().tolist())

    items = np.asarray(all_items, dtype=np.int64)
    ranks = np.asarray(all_ranks, dtype=np.int64)
    groups = split_by_test_target_pop(items, pop)

    topk = [5, 10, 20]
    print("group\tn\test_pop_mean\test_pop_min\test_pop_max\t" + "\t".join([f"{m}@{k}" for m in ["recall", "ndcg"] for k in topk]))
    for name in ["all", "head", "mid", "tail"]:
        idx = np.arange(len(items)) if name == "all" else groups[name]
        metrics = metric_from_ranks(ranks[idx], topk)
        values = []
        for m in ["recall", "ndcg"]:
            for k in topk:
                values.append(metrics[f"{m}@{k}"])
        group_pop = pop[items[idx]]
        print(
            "{}\t{}\t{:.4f}\t{}\t{}\t{}".format(
                name,
                len(idx),
                float(group_pop.mean()) if len(idx) else math.nan,
                int(group_pop.min()) if len(idx) else 0,
                int(group_pop.max()) if len(idx) else 0,
                "\t".join(f"{v:.6f}" for v in values),
            )
        )


if __name__ == "__main__":
    main()
