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


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def rankdata(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def hit_metrics(ranks):
    out = {}
    for k in [5, 10, 20]:
        hit = ranks <= k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float((hit / np.log2(ranks + 1.0)).mean())
    return out


def build_runtime(model_name, checkpoint, args):
    sys.argv = [
        sys.argv[0],
        "--model", model_name,
        "--dataset", args.dataset,
        "--gpu_id", str(args.gpu_id),
        "--hidden_size", str(args.hidden_size),
        "--inner_size", str(args.inner_size),
        "--temperature", str(args.temperature),
    ]
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu_id)
    model_class = get_model_class(parsed.model)
    config = Config(model=model_class, dataset=parsed.dataset, config_dict=build_config_dict(parsed))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    ckpt = torch.load(checkpoint, map_location=config["device"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.load_other_parameter(ckpt.get("other_parameter"))
    model.eval()
    return config, dataset, train_data, test_data, model


def groups_by_target_pop(items, pop):
    order = np.argsort(-pop[items], kind="stable")
    groups = {"all": np.arange(len(items))}
    groups.update({n: idx for n, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    args = parser.parse_args()

    config, dataset, train_data, test_data, model = build_runtime("SASRec", args.checkpoint, args)
    device = config["device"]
    item_emb = model.item_embedding.weight.detach()
    item_norm = item_emb.norm(dim=-1)
    item_dir = F.normalize(item_emb, dim=-1)
    pop = np.bincount(
        train_data.dataset.inter_feat[config["ITEM_ID_FIELD"]].cpu().numpy(),
        minlength=dataset.item_num,
    )

    target_items, dot_ranks, cos_ranks, inorm_ranks = [], [], [], []
    target_cos, target_hnorm, target_inorm, target_dot = [], [], [], []
    top_dot_cos, top_dot_hnorm, top_dot_inorm, top_dot_pop = [], [], [], []
    top_cos_pop, top_inorm_pop = [], []

    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            seq = interaction[model.ITEM_SEQ]
            seq_len = interaction[model.ITEM_SEQ_LEN]
            h = model.forward(seq, seq_len)
            hnorm = h.norm(dim=-1)
            hdir = F.normalize(h, dim=-1)
            cos_scores = torch.matmul(hdir, item_dir.t())
            dot_scores = torch.matmul(h, item_emb.t())
            inorm_scores = item_norm.unsqueeze(0).expand_as(dot_scores).clone()
            for scores in [cos_scores, dot_scores, inorm_scores]:
                scores[:, 0] = -float("inf")
                if history_index is not None:
                    scores[history_index] = -float("inf")

            pos_dot = dot_scores[positive_u, positive_i]
            pos_cos = cos_scores[positive_u, positive_i]
            pos_inorm = item_norm[positive_i]
            pos_hnorm = hnorm[positive_u]

            dot_rank = (dot_scores[positive_u] > pos_dot.unsqueeze(1)).sum(dim=1) + 1
            cos_rank = (cos_scores[positive_u] > pos_cos.unsqueeze(1)).sum(dim=1) + 1
            inorm_rank = (inorm_scores[positive_u] > pos_inorm.unsqueeze(1)).sum(dim=1) + 1

            top_dot = torch.topk(dot_scores[positive_u], 20, dim=1).indices
            top_cos = torch.topk(cos_scores[positive_u], 20, dim=1).indices
            top_inorm = torch.topk(inorm_scores[positive_u], 20, dim=1).indices

            target_items.extend(positive_i.cpu().numpy().tolist())
            dot_ranks.extend(dot_rank.cpu().numpy().tolist())
            cos_ranks.extend(cos_rank.cpu().numpy().tolist())
            inorm_ranks.extend(inorm_rank.cpu().numpy().tolist())
            target_dot.extend(pos_dot.cpu().numpy().tolist())
            target_cos.extend(pos_cos.cpu().numpy().tolist())
            target_inorm.extend(pos_inorm.cpu().numpy().tolist())
            target_hnorm.extend(pos_hnorm.cpu().numpy().tolist())
            top_dot_cos.extend(cos_scores[positive_u].gather(1, top_dot).mean(dim=1).cpu().numpy().tolist())
            top_dot_hnorm.extend(hnorm[positive_u].cpu().numpy().tolist())
            top_dot_inorm.extend(item_norm[top_dot].mean(dim=1).cpu().numpy().tolist())
            top_dot_pop.extend(pop[top_dot.cpu().numpy()].mean(axis=1).tolist())
            top_cos_pop.extend(pop[top_cos.cpu().numpy()].mean(axis=1).tolist())
            top_inorm_pop.extend(pop[top_inorm.cpu().numpy()].mean(axis=1).tolist())

    arrays = {k: np.asarray(v) for k, v in {
        "items": target_items,
        "dot_ranks": dot_ranks,
        "cos_ranks": cos_ranks,
        "inorm_ranks": inorm_ranks,
        "target_dot": target_dot,
        "target_cos": target_cos,
        "target_inorm": target_inorm,
        "target_hnorm": target_hnorm,
        "top_dot_cos": top_dot_cos,
        "top_dot_hnorm": top_dot_hnorm,
        "top_dot_inorm": top_dot_inorm,
        "top_dot_pop": top_dot_pop,
        "top_cos_pop": top_cos_pop,
        "top_inorm_pop": top_inorm_pop,
    }.items()}
    groups = groups_by_target_pop(arrays["items"].astype(np.int64), pop)

    print("ITEM_NORM_RELATION")
    active = np.arange(1, dataset.item_num)
    active = active[pop[active] > 0]
    print(f"pearson_itemnorm_logpop\t{pearson(item_norm.cpu().numpy()[active], np.log1p(pop[active])):.6f}")
    print(f"spearman_itemnorm_pop\t{spearman(item_norm.cpu().numpy()[active], pop[active]):.6f}")

    print("RANKING_BY_COMPONENT")
    print("component\tgroup\trecall@5\trecall@10\trecall@20\tndcg@5\tndcg@10\tndcg@20")
    for comp, rank_key in [("dot", "dot_ranks"), ("cos_only", "cos_ranks"), ("item_norm_only", "inorm_ranks")]:
        for group, idx in groups.items():
            m = hit_metrics(arrays[rank_key][idx])
            print(f"{comp}\t{group}\t{m['recall@5']:.6f}\t{m['recall@10']:.6f}\t{m['recall@20']:.6f}\t{m['ndcg@5']:.6f}\t{m['ndcg@10']:.6f}\t{m['ndcg@20']:.6f}")

    print("TARGET_SCORE_COMPONENTS_BY_GROUP")
    print("group\tpop_mean\thnorm_mean\tinorm_mean\tcos_mean\tdot_mean")
    for group, idx in groups.items():
        items = arrays["items"][idx].astype(np.int64)
        print(
            f"{group}\t{pop[items].mean():.6f}\t{arrays['target_hnorm'][idx].mean():.6f}\t"
            f"{arrays['target_inorm'][idx].mean():.6f}\t{arrays['target_cos'][idx].mean():.6f}\t{arrays['target_dot'][idx].mean():.6f}"
        )

    print("CORRELATION_WITH_TARGET_RANK_NEGATIVE_IS_BETTER")
    rank_quality = -arrays["dot_ranks"]
    for name in ["target_dot", "target_cos", "target_hnorm", "target_inorm"]:
        print(f"{name}\tpearson={pearson(arrays[name], rank_quality):.6f}\tspearman={spearman(arrays[name], rank_quality):.6f}")

    print("TOP20_AVERAGES")
    print(f"dot_top20_pop_mean\t{arrays['top_dot_pop'].mean():.6f}")
    print(f"cosonly_top20_pop_mean\t{arrays['top_cos_pop'].mean():.6f}")
    print(f"itemnorm_top20_pop_mean\t{arrays['top_inorm_pop'].mean():.6f}")
    print(f"dot_top20_itemnorm_mean\t{arrays['top_dot_inorm'].mean():.6f}")
    print(f"dot_top20_cos_mean\t{arrays['top_dot_cos'].mean():.6f}")


if __name__ == "__main__":
    main()
