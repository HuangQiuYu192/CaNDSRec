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
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    xr = np.empty(len(x), dtype=np.float64)
    yr = np.empty(len(y), dtype=np.float64)
    xr[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    yr[np.argsort(y, kind="mergesort")] = np.arange(len(y))
    return pearson(xr, yr)


def build_model(model_name, checkpoint, args):
    sys.argv = [
        sys.argv[0],
        "--model", model_name,
        "--dataset", args.dataset,
        "--gpu_id", str(args.gpu_id),
        "--hidden_size", str(args.hidden_size),
        "--inner_size", str(args.inner_size),
        "--temperature", str(args.temperature),
        "--max_item_list_length", str(args.max_item_list_length),
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


def get_group_indices(items, pop):
    order = np.argsort(-pop[items], kind="stable")
    return {"all": np.arange(len(items)), **{n: i for n, i in zip(["head", "mid", "tail"], np.array_split(order, 3))}}


def summarize_group(name, values, groups):
    print(name)
    print("group\tn\tmean\tmedian\tp25\tp75")
    for group, idx in groups.items():
        v = np.asarray(values)[idx]
        print(f"{group}\t{len(v)}\t{v.mean():.6f}\t{np.median(v):.6f}\t{np.percentile(v,25):.6f}\t{np.percentile(v,75):.6f}")


def collect_stats(model, test_data, mode, max_batches=None, topk=20):
    device = next(model.parameters()).device
    item_emb = model.item_embedding.weight
    item_dir = F.normalize(item_emb, dim=-1)
    item_norm = item_emb.norm(dim=-1)
    all_items, all_ranks, pos_cos, hard_neg_cos, margin = [], [], [], [], []
    pos_score_raw, top_item_norm, top_item_pop_idx = [], [], []
    top_items = []

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            seq = interaction[model.ITEM_SEQ]
            seq_len = interaction[model.ITEM_SEQ_LEN]
            seq_out = model.forward(seq, seq_len)
            seq_dir = F.normalize(seq_out, dim=-1)
            cos_scores = torch.matmul(seq_dir, item_dir.t())
            if mode == "dot":
                scores = torch.matmul(seq_out, item_emb.t())
            else:
                scores = float(getattr(model, "temperature", 1.0)) * cos_scores

            scores[:, 0] = -float("inf")
            history_index = batched_data[1]
            if history_index is not None:
                scores[history_index] = -float("inf")

            pos_s = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_s.unsqueeze(1)).sum(dim=1) + 1
            masked_cos = cos_scores[positive_u].clone()
            masked_cos[:, 0] = -float("inf")
            masked_cos[torch.arange(len(positive_i), device=device), positive_i] = -float("inf")
            hard_cos = masked_cos.max(dim=1).values
            pcos = cos_scores[positive_u, positive_i]
            top = torch.topk(scores[positive_u], k=topk, dim=1).indices

            all_items.extend(positive_i.cpu().numpy().tolist())
            all_ranks.extend(ranks.cpu().numpy().tolist())
            pos_cos.extend(pcos.cpu().numpy().tolist())
            hard_neg_cos.extend(hard_cos.cpu().numpy().tolist())
            margin.extend((pcos - hard_cos).cpu().numpy().tolist())
            pos_score_raw.extend(pos_s.cpu().numpy().tolist())
            top_item_norm.extend(item_norm[top].mean(dim=1).cpu().numpy().tolist())
            top_items.extend(top.cpu().numpy().reshape(-1).tolist())

    return {
        "items": np.asarray(all_items, dtype=np.int64),
        "ranks": np.asarray(all_ranks, dtype=np.int64),
        "pos_cos": np.asarray(pos_cos),
        "hard_neg_cos": np.asarray(hard_neg_cos),
        "margin": np.asarray(margin),
        "pos_score_raw": np.asarray(pos_score_raw),
        "top_item_norm": np.asarray(top_item_norm),
        "top_items": np.asarray(top_items, dtype=np.int64),
    }


def hit_metrics(ranks):
    out = {}
    for k in [5, 10, 20]:
        hit = ranks <= k
        out[f"recall@{k}"] = hit.mean()
        out[f"ndcg@{k}"] = (hit / np.log2(ranks + 1.0)).mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_checkpoint", required=True)
    parser.add_argument("--cosine_checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    args = parser.parse_args()

    cfg, dataset, train_data, test_data, sasrec = build_model("SASRec", args.sasrec_checkpoint, args)
    _, _, _, test_data_c, cosine = build_model("CANDSSASRec", args.cosine_checkpoint, args)
    pop = np.bincount(
        train_data.dataset.inter_feat[cfg["ITEM_ID_FIELD"]].cpu().numpy(),
        minlength=dataset.item_num,
    )
    active = np.arange(1, dataset.item_num)
    active = active[pop[active] > 0]

    print("ITEM_NORM_CORRELATION")
    for name, model in [("sasrec_dot", sasrec), ("cosine", cosine)]:
        norm = model.item_embedding.weight.detach().norm(dim=-1).cpu().numpy()
        print(
            f"{name}\tpearson_norm_logpop={pearson(norm[active], np.log1p(pop[active])):.6f}"
            f"\tspearman_norm_pop={spearman(norm[active], pop[active]):.6f}"
            f"\tnorm_mean={norm[active].mean():.6f}\tnorm_std={norm[active].std():.6f}"
        )

    dot_stats = collect_stats(sasrec, test_data, "dot")
    cos_stats = collect_stats(cosine, test_data_c, "cosine")
    groups = get_group_indices(dot_stats["items"], pop)

    print("TEST_METRICS_BY_GROUP")
    print("model\tgroup\tn\trecall@5\trecall@10\trecall@20\tndcg@5\tndcg@10\tndcg@20")
    for name, stats in [("sasrec_dot", dot_stats), ("cosine", cos_stats)]:
        for group, idx in groups.items():
            m = hit_metrics(stats["ranks"][idx])
            print(f"{name}\t{group}\t{len(idx)}\t{m['recall@5']:.6f}\t{m['recall@10']:.6f}\t{m['recall@20']:.6f}\t{m['ndcg@5']:.6f}\t{m['ndcg@10']:.6f}\t{m['ndcg@20']:.6f}")

    print("TOPK_RECOMMENDED_ITEM_BIAS")
    for name, stats, model in [("sasrec_dot", dot_stats, sasrec), ("cosine", cos_stats, cosine)]:
        item_norm = model.item_embedding.weight.detach().norm(dim=-1).cpu().numpy()
        top_pop = pop[stats["top_items"]]
        top_norm = item_norm[stats["top_items"]]
        target_pop = pop[stats["items"]]
        print(
            f"{name}\ttop20_pop_mean={top_pop.mean():.6f}\ttarget_pop_mean={target_pop.mean():.6f}"
            f"\ttop20_norm_mean={top_norm.mean():.6f}\ttarget_norm_mean={item_norm[stats['items']].mean():.6f}"
        )

    summarize_group("POSITIVE_COSINE_BY_TARGET_GROUP_SASREC_DOT", dot_stats["pos_cos"], groups)
    summarize_group("POSITIVE_COSINE_BY_TARGET_GROUP_COSINE", cos_stats["pos_cos"], groups)
    summarize_group("HARD_NEG_COSINE_BY_TARGET_GROUP_SASREC_DOT", dot_stats["hard_neg_cos"], groups)
    summarize_group("HARD_NEG_COSINE_BY_TARGET_GROUP_COSINE", cos_stats["hard_neg_cos"], groups)
    summarize_group("ANGULAR_MARGIN_POS_MINUS_HARDNEG_SASREC_DOT", dot_stats["margin"], groups)
    summarize_group("ANGULAR_MARGIN_POS_MINUS_HARDNEG_COSINE", cos_stats["margin"], groups)

    print("RANK_DELTA_COSINE_VS_DOT")
    print("group\tn\tmean_delta_rank\tmedian_delta_rank\timproved_pct\tworse_pct\thit10_gain\thit10_loss\thit20_gain\thit20_loss")
    delta = dot_stats["ranks"] - cos_stats["ranks"]
    for group, idx in groups.items():
        d = delta[idx]
        dr = dot_stats["ranks"][idx]
        cr = cos_stats["ranks"][idx]
        print(
            f"{group}\t{len(idx)}\t{d.mean():.6f}\t{np.median(d):.6f}\t{(d>0).mean():.6f}\t{(d<0).mean():.6f}"
            f"\t{int(((dr>10)&(cr<=10)).sum())}\t{int(((dr<=10)&(cr>10)).sum())}"
            f"\t{int(((dr>20)&(cr<=20)).sum())}\t{int(((dr<=20)&(cr>20)).sum())}"
        )


if __name__ == "__main__":
    main()
