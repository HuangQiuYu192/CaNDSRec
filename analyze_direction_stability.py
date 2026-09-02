#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from argument_parser import build_config_dict, parse_args
from models import get_model_class


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})


def build_dataset(dataset_name, hidden_size, inner_size):
    sys.argv = [
        sys.argv[0],
        "--model", "CANDSSASRec",
        "--dataset", dataset_name,
        "--gpu_id", "0",
        "--hidden_size", str(hidden_size),
        "--inner_size", str(inner_size),
        "--temperature", "10.0",
    ]
    args = parse_args()
    model_class = get_model_class(args.model)
    config = Config(model=model_class, dataset=args.dataset, config_dict=build_config_dict(args))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    pop = np.bincount(
        train_data.dataset.inter_feat[config["ITEM_ID_FIELD"]].cpu().numpy(),
        minlength=dataset.item_num,
    )
    return dataset.item_num, pop


def load_item_embedding(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    weight = state["item_embedding.weight"].float()
    return F.normalize(weight, dim=-1)


def orthogonal_align(src, tgt, anchors):
    x = src[anchors]
    y = tgt[anchors]
    m = x.t().matmul(y)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    r = u.matmul(vh)
    return F.normalize(src.matmul(r), dim=-1)


def pairwise_same_item_cos(embs, item_ids):
    vals = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            vals.append((embs[i][item_ids] * embs[j][item_ids]).sum(dim=-1).numpy())
    return np.stack(vals, axis=0).mean(axis=0)


def summarize(name, values, pop_values):
    values = np.asarray(values)
    pop_values = np.asarray(pop_values)
    print(
        "{}\tn={}\tpop_mean={:.4f}\tcos_mean={:.6f}\tcos_median={:.6f}\tcos_p10={:.6f}\tcos_p25={:.6f}\tcos_p75={:.6f}".format(
            name,
            len(values),
            float(pop_values.mean()),
            float(values.mean()),
            float(np.median(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()

    item_num, pop = build_dataset(args.dataset, args.hidden_size, args.inner_size)
    embs = [load_item_embedding(path) for path in args.checkpoints]
    for path, emb in zip(args.checkpoints, embs):
        if emb.shape[0] != item_num:
            raise RuntimeError(f"checkpoint item size mismatch: {path} has {emb.shape[0]} expected {item_num}")

    active = np.arange(1, item_num)
    active = active[pop[active] > 0]
    item_order = active[np.argsort(-pop[active], kind="stable")]
    item_groups = {name: idx for name, idx in zip(["head_items", "mid_items", "tail_items"], np.array_split(item_order, 3))}

    test_target_items = []
    # The test-target grouping is not needed for direction itself; item-level groups are cleaner here.

    raw_cos = pairwise_same_item_cos(embs, active)
    anchors = item_groups["head_items"]
    aligned = [embs[0]]
    for emb in embs[1:]:
        aligned.append(orthogonal_align(emb, embs[0], anchors))
    aligned_cos = pairwise_same_item_cos(aligned, active)

    print("DIRECTION_STABILITY_RAW")
    print("group\tn\tpop_mean\tcos_mean\tcos_median\tcos_p10\tcos_p25\tcos_p75")
    for name, idx in {"all_items": active, **item_groups}.items():
        positions = np.searchsorted(active, idx)
        summarize(name, raw_cos[positions], pop[idx])

    print("DIRECTION_STABILITY_ALIGNED_BY_HEAD")
    print("group\tn\tpop_mean\tcos_mean\tcos_median\tcos_p10\tcos_p25\tcos_p75")
    for name, idx in {"all_items": active, **item_groups}.items():
        positions = np.searchsorted(active, idx)
        summarize(name, aligned_cos[positions], pop[idx])

    print("CHECKPOINTS")
    for path in args.checkpoints:
        print(path)


if __name__ == "__main__":
    main()
