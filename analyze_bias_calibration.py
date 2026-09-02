#!/usr/bin/env python3
import argparse
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


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def ranks_of_values(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x, y):
    return pearson(ranks_of_values(x), ranks_of_values(y))


def metric_from_ranks(ranks, topk=(5, 10, 20)):
    ranks = np.asarray(ranks)
    out = {}
    for k in topk:
        hit = ranks <= k
        out[f"recall@{k}"] = float(hit.mean()) if len(ranks) else 0.0
        out[f"ndcg@{k}"] = float((hit / np.log2(ranks + 1.0)).mean()) if len(ranks) else 0.0
    return out


def build_runtime(model_name, dataset_name, gpu_id, hidden_size, inner_size, temperature):
    sys.argv = [
        sys.argv[0],
        "--model",
        model_name,
        "--dataset",
        dataset_name,
        "--gpu_id",
        str(gpu_id),
        "--hidden_size",
        str(hidden_size),
        "--inner_size",
        str(inner_size),
        "--temperature",
        str(temperature),
    ]
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model_class = get_model_class(args.model)
    config = Config(model=model_class, dataset=args.dataset, config_dict=build_config_dict(args))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], args.model)(config, model)
    trainer.tot_item_num = test_data._dataset.item_num
    trainer.item_tensor = test_data._dataset.get_item_feature().to(config["device"])
    return config, dataset, train_data, test_data, model, trainer


def load_model(model, checkpoint):
    ckpt = torch.load(checkpoint, map_location=next(model.parameters()).device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.load_other_parameter(ckpt.get("other_parameter"))
    model.eval()


def target_ranks(model, trainer, test_data):
    items = []
    ranks = []
    with torch.no_grad():
        for batched_data in test_data:
            interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
            positive_u = torch.as_tensor(positive_u, device=scores.device).long()
            positive_i = torch.as_tensor(positive_i, device=scores.device).long()
            pos_scores = scores[positive_u, positive_i]
            rank = (scores[positive_u] > pos_scores.unsqueeze(1)).sum(dim=1) + 1
            items.extend(positive_i.detach().cpu().numpy().tolist())
            ranks.extend(rank.detach().cpu().numpy().tolist())
    return np.asarray(items, dtype=np.int64), np.asarray(ranks, dtype=np.int64)


def split_test_groups(items, pop):
    order = np.argsort(-pop[items], kind="stable")
    return {name: idx for name, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))}


def format_metrics(metrics):
    return "\t".join(f"{metrics[k]:.6f}" for k in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosine_checkpoint", required=True)
    parser.add_argument("--cosine_temperature", required=True, type=float)
    parser.add_argument("--bias_checkpoint", required=True)
    parser.add_argument("--bias_temperature", default=10.0, type=float)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    args = parser.parse_args()

    cfg, dataset, train_data, test_data, cosine_model, cosine_trainer = build_runtime(
        "CANDSSASRec", args.dataset, args.gpu_id, args.hidden_size, args.inner_size, args.cosine_temperature
    )
    pop = np.bincount(
        train_data.dataset.inter_feat[cfg["ITEM_ID_FIELD"]].cpu().numpy(),
        minlength=dataset.item_num,
    )
    load_model(cosine_model, args.cosine_checkpoint)
    items, cosine_ranks = target_ranks(cosine_model, cosine_trainer, test_data)

    _, _, _, test_data_b, bias_model, bias_trainer = build_runtime(
        "CalibratedCANDSSASRec", args.dataset, args.gpu_id, args.hidden_size, args.inner_size, args.bias_temperature
    )
    load_model(bias_model, args.bias_checkpoint)
    bias_items, bias_ranks = target_ranks(bias_model, bias_trainer, test_data_b)
    if not np.array_equal(items, bias_items):
        raise RuntimeError("test item order differs between checkpoints")

    bias = bias_model.item_bias.detach().cpu().numpy()
    active = np.arange(1, len(pop))
    active = active[pop[active] > 0]
    logpop = np.log1p(pop[active])
    active_bias = bias[active]

    print("BIAS_POP_CORRELATION")
    print(f"pearson_bias_logpop\t{pearson(active_bias, logpop):.6f}")
    print(f"spearman_bias_pop\t{spearman(active_bias, pop[active]):.6f}")
    print(f"bias_mean\t{active_bias.mean():.6f}")
    print(f"bias_std\t{active_bias.std():.6f}")
    print(f"bias_min\t{active_bias.min():.6f}")
    print(f"bias_max\t{active_bias.max():.6f}")

    print("ITEM_POP_GROUP_BIAS")
    item_order = active[np.argsort(-pop[active], kind="stable")]
    for name, idx in zip(["head_items", "mid_items", "tail_items"], np.array_split(item_order, 3)):
        print(
            "{}\tn={}\tpop_mean={:.4f}\tbias_mean={:.6f}\tbias_std={:.6f}\tbias_min={:.6f}\tbias_max={:.6f}".format(
                name, len(idx), float(pop[idx].mean()), float(bias[idx].mean()), float(bias[idx].std()), float(bias[idx].min()), float(bias[idx].max())
            )
        )

    print("TEST_TARGET_GROUP_METRICS")
    print("model\tgroup\tn\tpop_mean\trecall@5\trecall@10\trecall@20\tndcg@5\tndcg@10\tndcg@20")
    groups = split_test_groups(items, pop)
    for model_name, ranks in [("cosine", cosine_ranks), ("bias", bias_ranks)]:
        for group_name in ["all", "head", "mid", "tail"]:
            idx = np.arange(len(items)) if group_name == "all" else groups[group_name]
            metrics = metric_from_ranks(ranks[idx])
            print(f"{model_name}\t{group_name}\t{len(idx)}\t{pop[items[idx]].mean():.4f}\t{format_metrics(metrics)}")

    print("RANK_DELTA")
    print("group\tn\tmean_delta_rank\tmedian_delta_rank\timproved_pct\tworse_pct\ttie_pct\thit10_gain\thit10_loss\thit20_gain\thit20_loss")
    delta = cosine_ranks - bias_ranks
    for group_name in ["all", "head", "mid", "tail"]:
        idx = np.arange(len(items)) if group_name == "all" else groups[group_name]
        d = delta[idx]
        c = cosine_ranks[idx]
        b = bias_ranks[idx]
        print(
            "{}\t{}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\t{}\t{}\t{}".format(
                group_name,
                len(idx),
                float(d.mean()),
                float(np.median(d)),
                float((d > 0).mean()),
                float((d < 0).mean()),
                float((d == 0).mean()),
                int(((c > 10) & (b <= 10)).sum()),
                int(((c <= 10) & (b > 10)).sum()),
                int(((c > 20) & (b <= 20)).sum()),
                int(((c <= 20) & (b > 20)).sum()),
            )
        )

    print("TARGET_BIAS_BY_GROUP")
    print("group\tn\tbias_mean\tbias_std\tbias_min\tbias_max")
    for group_name in ["all", "head", "mid", "tail"]:
        idx = np.arange(len(items)) if group_name == "all" else groups[group_name]
        vals = bias[items[idx]]
        print(f"{group_name}\t{len(idx)}\t{vals.mean():.6f}\t{vals.std():.6f}\t{vals.min():.6f}\t{vals.max():.6f}")


if __name__ == "__main__":
    main()
