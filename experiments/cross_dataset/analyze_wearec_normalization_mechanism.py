#!/usr/bin/env python3
"""Mechanism diagnostics for WEARec vs CANDSWEARec.

The script compares a dot-product WEARec checkpoint with a cosine-temperature
CANDSWEARec checkpoint on the same dataset/configuration. It reports item norm
statistics, head/mid/tail target-group metrics, angular score distributions, and
rank changes. This is intended to test whether the normalization effect observed
on SASRec also appears with a frequency/wavelet sequential backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x, y):
    return pearson(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
    }


def hit_metrics(ranks):
    output = {}
    for k in [5, 10, 20]:
        hit = ranks <= k
        output[f"recall@{k}"] = float(hit.mean())
        output[f"ndcg@{k}"] = float((hit / np.log2(ranks + 1.0)).mean())
    return output


def build_model(model_name, checkpoint, args):
    sys.argv = [
        sys.argv[0],
        "--model",
        model_name,
        "--dataset",
        args.dataset,
        "--gpu_id",
        str(args.gpu_id),
        "--seed",
        str(args.seed),
        "--hidden_size",
        str(args.hidden_size),
        "--n_layers",
        str(args.n_layers),
        "--num_heads",
        str(args.num_heads),
        "--inner_size",
        str(args.inner_size),
        "--alpha",
        str(args.alpha),
        "--hidden_dropout_prob",
        str(args.hidden_dropout_prob),
        "--learning_rate",
        str(args.learning_rate),
        "--max_item_list_length",
        str(args.max_item_list_length),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--train_batch_size",
        str(args.train_batch_size),
        "--temperature",
        str(args.temperature),
        "--verbose",
        "False",
        "--show_progress",
        "False",
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


def target_groups(items, pop):
    order = np.argsort(-pop[items], kind="stable")
    groups = {"all": np.arange(len(items))}
    groups.update({name: idx for name, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))})
    return groups


def collect_rank_stats(model, test_data, mode, max_batches=None, topk=20):
    device = next(model.parameters()).device
    item_emb = model.item_embedding.weight.detach()
    item_dir = F.normalize(item_emb, dim=-1)
    item_norm = item_emb.norm(dim=-1)

    output = {
        "items": [],
        "ranks": [],
        "pos_score": [],
        "pos_cos": [],
        "hard_neg_cos": [],
        "cos_margin": [],
        "seq_norm": [],
        "item_norm": [],
        "top_items": [],
        "top_item_norm": [],
    }

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            seq = interaction[model.ITEM_SEQ]
            seq_len = interaction[model.ITEM_SEQ_LEN]
            seq_out = model.forward(seq, seq_len)
            seq_dir = F.normalize(seq_out, dim=-1)

            cos_scores = torch.matmul(seq_dir, item_dir.transpose(0, 1))
            if mode == "dot":
                scores = torch.matmul(seq_out, item_emb.transpose(0, 1))
            else:
                scores = float(getattr(model, "temperature", 1.0)) * cos_scores

            scores[:, 0] = -float("inf")
            cos_scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")
                cos_scores[history_index] = -float("inf")

            pos_score = scores[positive_u, positive_i]
            pos_cos = cos_scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1
            masked_cos = cos_scores[positive_u].clone()
            masked_cos[torch.arange(len(positive_i), device=device), positive_i] = -float("inf")
            hard_neg_cos = masked_cos.max(dim=1).values
            top_items = torch.topk(scores[positive_u], k=topk, dim=1).indices

            output["items"].extend(positive_i.cpu().numpy().tolist())
            output["ranks"].extend(ranks.cpu().numpy().tolist())
            output["pos_score"].extend(pos_score.cpu().numpy().tolist())
            output["pos_cos"].extend(pos_cos.cpu().numpy().tolist())
            output["hard_neg_cos"].extend(hard_neg_cos.cpu().numpy().tolist())
            output["cos_margin"].extend((pos_cos - hard_neg_cos).cpu().numpy().tolist())
            output["seq_norm"].extend(seq_out[positive_u].norm(dim=-1).cpu().numpy().tolist())
            output["item_norm"].extend(item_norm[positive_i].cpu().numpy().tolist())
            output["top_items"].extend(top_items.cpu().numpy().reshape(-1).tolist())
            output["top_item_norm"].extend(item_norm[top_items].mean(dim=1).cpu().numpy().tolist())

    return {key: np.asarray(value) for key, value in output.items()}


def add_prefixed(row, prefix, values):
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wearec_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--alpha", default=0.8, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=1024, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config, dataset, train_data, test_data, wearec = build_model("WEARec", args.wearec_checkpoint, args)
    _, _, _, test_data_cands, cands = build_model("CANDSWEARec", args.cands_checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(
        train_data.dataset.inter_feat[item_field].cpu().numpy(),
        minlength=dataset.item_num,
    )
    active = np.arange(1, dataset.item_num)
    active = active[pop[active] > 0]

    dot_stats = collect_rank_stats(wearec, test_data, "dot", args.max_batches)
    cands_stats = collect_rank_stats(cands, test_data_cands, "cosine", args.max_batches)
    groups = target_groups(dot_stats["items"].astype(np.int64), pop)

    rows = []
    base = {
        "dataset": args.dataset,
        "hidden": args.hidden_size,
        "max_len": args.max_item_list_length,
        "temperature": args.temperature,
        "wearec_checkpoint": args.wearec_checkpoint,
        "cands_checkpoint": args.cands_checkpoint,
    }

    for model_name, model, stats in [
        ("WEARec", wearec, dot_stats),
        ("CANDSWEARec", cands, cands_stats),
    ]:
        item_norm = model.item_embedding.weight.detach().norm(dim=-1).cpu().numpy()
        for group, idx in groups.items():
            row = dict(base)
            row["model"] = model_name
            row["group"] = group
            row["n"] = int(len(idx))
            add_prefixed(row, "rank", hit_metrics(stats["ranks"][idx]))
            add_prefixed(row, "pos_score", summarize(stats["pos_score"][idx]))
            add_prefixed(row, "pos_cos", summarize(stats["pos_cos"][idx]))
            add_prefixed(row, "hard_neg_cos", summarize(stats["hard_neg_cos"][idx]))
            add_prefixed(row, "cos_margin", summarize(stats["cos_margin"][idx]))
            add_prefixed(row, "seq_norm", summarize(stats["seq_norm"][idx]))
            add_prefixed(row, "target_item_norm", summarize(stats["item_norm"][idx]))
            row["item_norm_pearson_logpop"] = pearson(item_norm[active], np.log1p(pop[active]))
            row["item_norm_spearman_pop"] = spearman(item_norm[active], pop[active])
            rows.append(row)

    delta_rank = dot_stats["ranks"] - cands_stats["ranks"]
    for group, idx in groups.items():
        row = dict(base)
        row["model"] = "CANDSWEARec_minus_WEARec"
        row["group"] = group
        row["n"] = int(len(idx))
        row["mean_delta_rank"] = float(delta_rank[idx].mean())
        row["median_delta_rank"] = float(np.median(delta_rank[idx]))
        row["improved_pct"] = float((delta_rank[idx] > 0).mean())
        row["worse_pct"] = float((delta_rank[idx] < 0).mean())
        dot_rank = dot_stats["ranks"][idx]
        cands_rank = cands_stats["ranks"][idx]
        row["hit10_gain"] = int(((dot_rank > 10) & (cands_rank <= 10)).sum())
        row["hit10_loss"] = int(((dot_rank <= 10) & (cands_rank > 10)).sum())
        row["hit20_gain"] = int(((dot_rank > 20) & (cands_rank <= 20)).sum())
        row["hit20_loss"] = int(((dot_rank <= 20) & (cands_rank > 20)).sum())
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    headers = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output} and {json_path}")


if __name__ == "__main__":
    main()
