#!/usr/bin/env python3
"""Grouped full-sort evaluation for a trained sequential recommendation model."""

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


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def parse_list(text: str, cast=int) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def build_cli_for_model(model_name: str, args: argparse.Namespace) -> list[str]:
    cli = [
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
        "--inner_size",
        str(args.inner_size),
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
    if model_name in {"SASRec", "CANDSSASRec", "AngularSmoothCANDSSASRec", "SemanticCANDSSASRec", "TailSimCANDSSASRec"}:
        cli.extend(["--n_heads", str(args.n_heads), "--attn_dropout_prob", str(args.attn_dropout_prob)])
    if model_name in {"WEARec", "CANDSWEARec"}:
        cli.extend(["--num_heads", str(args.wearec_num_heads), "--alpha", str(args.wearec_alpha)])
    if model_name == "AngularSmoothCANDSSASRec":
        cli.extend(
            [
                "--angular_smooth_weight",
                str(args.angular_smooth_weight),
                "--angular_smooth_k",
                str(args.angular_smooth_k),
                "--angular_smooth_temperature",
                str(args.angular_smooth_temperature),
                "--angular_smooth_pop_quantile",
                str(args.angular_smooth_pop_quantile),
                "--angular_smooth_sim_threshold",
                str(args.angular_smooth_sim_threshold),
                "--angular_smooth_pop_weight",
                str(args.angular_smooth_pop_weight),
            ]
        )
    if model_name == "SemanticCANDSSASRec":
        cli.extend(
            [
                "--semantic_embedding_path",
                str(args.semantic_embedding_path),
                "--semantic_fusion_mode",
                str(args.semantic_fusion_mode),
                "--semantic_weight",
                str(args.semantic_weight),
                "--semantic_gate",
                str(args.semantic_gate),
                "--semantic_freeze",
                str(args.semantic_freeze),
            ]
        )
    return cli


def build_model(model_name: str, checkpoint: str, args: argparse.Namespace):
    sys.argv = build_cli_for_model(model_name, args)
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
    return config, dataset, train_data, valid_data, test_data, model


def target_quantile_groups(items: np.ndarray, pop: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-pop[items], kind="stable")
    groups = {"all": np.arange(len(items))}
    groups.update({name: idx for name, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))})
    return groups


def global_item_tertile_groups(items: np.ndarray, pop: np.ndarray) -> dict[str, np.ndarray]:
    """Group test examples by their target's training-popularity item tertile.

    Unlike target quantiles, this first partitions unique active items and then
    maps every test target to its item's membership.  Group sizes are therefore
    intentionally unequal: that is the expected consequence of popularity.
    """
    active = np.flatnonzero(pop > 0)
    active = active[active != 0]
    order = active[np.argsort(-pop[active], kind="stable")]
    labels = np.full(len(pop), -1, dtype=np.int8)
    for label, part in enumerate(np.array_split(order, 3)):
        labels[part] = label
    return {
        "all": np.arange(len(items)),
        "head": np.flatnonzero(labels[items] == 0),
        "mid": np.flatnonzero(labels[items] == 1),
        "tail": np.flatnonzero(labels[items] == 2),
    }


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    ranks = ranks.astype(np.float64)
    if len(ranks) == 0:
        return math.nan, math.nan
    hit = ranks <= k
    return float(hit.mean()), float((hit / np.log2(ranks + 1.0)).mean())


def collect_eval(model, eval_data, max_batches: int | None = None) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    output = {"items": [], "ranks": []}
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(eval_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()

            scores = model.full_sort_predict(interaction)
            if scores.dim() == 1:
                scores = scores.view(positive_i.size(0), -1)
            scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")

            pos_score = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1
            output["items"].extend(positive_i.cpu().numpy().tolist())
            output["ranks"].extend(ranks.cpu().numpy().tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def grouped_metrics(items: np.ndarray, ranks: np.ndarray, pop: np.ndarray, cutoffs: list[int], group_mode: str) -> list[dict]:
    rows = []
    if group_mode == "global_item_tertile":
        groups = global_item_tertile_groups(items.astype(np.int64), pop)
    else:
        groups = target_quantile_groups(items.astype(np.int64), pop)
    for group, idx in groups.items():
        group_ranks = ranks[idx]
        row = {"group": group, "n": int(len(idx))}
        for k in cutoffs:
            recall, ndcg = metric_at(group_ranks, k)
            row[f"recall@{k}"] = recall
            row[f"ndcg@{k}"] = ndcg
        row["mean_rank"] = float(group_ranks.mean()) if len(group_ranks) else math.nan
        row["median_rank"] = float(np.median(group_ranks)) if len(group_ranks) else math.nan
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], cutoffs: list[int]) -> None:
    headers = ["tag", "model", "hidden", "temperature", "group", "n", "median_rank"]
    for k in cutoffs:
        headers.extend([f"recall@{k}", f"ndcg@{k}"])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--wearec_num_heads", default=1, type=int)
    parser.add_argument("--wearec_alpha", default=0.8, type=float)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=1024, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--angular_smooth_weight", default=0.0, type=float)
    parser.add_argument("--angular_smooth_k", default=10, type=int)
    parser.add_argument("--angular_smooth_temperature", default=0.2, type=float)
    parser.add_argument("--angular_smooth_pop_quantile", default=0.67, type=float)
    parser.add_argument("--angular_smooth_sim_threshold", default=0.0, type=float)
    parser.add_argument("--angular_smooth_pop_weight", default=True)
    parser.add_argument("--semantic_embedding_path", default="")
    parser.add_argument("--semantic_fusion_mode", default="score")
    parser.add_argument("--semantic_weight", default=0.1, type=float)
    parser.add_argument("--semantic_gate", default="constant")
    parser.add_argument("--semantic_freeze", default=True)
    parser.add_argument("--cutoffs", default="5,10,20,50,100")
    parser.add_argument("--group_mode", choices=["target_quantile", "global_item_tertile"], default="target_quantile")
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    config, dataset, train_data, _, test_data, model = build_model(args.model, args.checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    cutoffs = parse_list(args.cutoffs, int)
    stats = collect_eval(model, test_data, args.max_batches)
    rows = grouped_metrics(stats["items"], stats["ranks"], pop, cutoffs, args.group_mode)
    meta = {
        "tag": args.tag or args.model,
        "dataset": args.dataset,
        "model": args.model,
        "group_mode": args.group_mode,
        "hidden": args.hidden_size,
        "max_len": args.max_item_list_length,
        "temperature": args.temperature,
        "angular_smooth_weight": args.angular_smooth_weight,
        "angular_smooth_k": args.angular_smooth_k,
        "angular_smooth_temperature": args.angular_smooth_temperature,
        "angular_smooth_pop_quantile": args.angular_smooth_pop_quantile,
        "angular_smooth_sim_threshold": args.angular_smooth_sim_threshold,
        "semantic_embedding_path": args.semantic_embedding_path,
        "semantic_fusion_mode": args.semantic_fusion_mode,
        "semantic_weight": args.semantic_weight,
        "semantic_gate": args.semantic_gate,
        "semantic_freeze": args.semantic_freeze,
    }
    rows = [{**meta, **row} for row in rows]

    out_prefix = Path(args.out_prefix)
    write_csv(Path(f"{out_prefix}.csv"), rows)
    write_markdown(Path(f"{out_prefix}.md"), rows, cutoffs)
    Path(f"{out_prefix}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote grouped metrics to {out_prefix}.csv and {out_prefix}.md")


if __name__ == "__main__":
    main()
