#!/usr/bin/env python3
"""Validate simple group-aware reranking on top of CaNDS full-sort scores.

The reranker uses only observable training-set popularity groups. For each user,
it first computes CaNDS full-sort scores, keeps a top-M candidate pool, and adds
a constant alpha bias to eligible item groups inside that pool. Alpha is selected
on validation data and then evaluated on test data.
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


def parse_list(text: str, cast=float) -> list:
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
    if model_name in {"SASRec", "CANDSSASRec"}:
        cli.extend(["--n_heads", str(args.n_heads), "--attn_dropout_prob", str(args.attn_dropout_prob)])
    if model_name in {"WEARec", "CANDSWEARec"}:
        cli.extend(["--num_heads", str(args.wearec_num_heads), "--alpha", str(args.wearec_alpha)])
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


def popularity_groups(pop: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    item_ids = np.arange(len(pop))
    active = item_ids[(item_ids > 0) & (pop > 0)]
    order = active[np.argsort(-pop[active], kind="stable")]
    group_ids = np.full(len(pop), -1, dtype=np.int64)
    masks = {}
    for idx, (name, ids) in enumerate(zip(["head", "mid", "tail"], np.array_split(order, 3))):
        mask = np.zeros(len(pop), dtype=bool)
        mask[ids] = True
        masks[name] = mask
        group_ids[ids] = idx
    masks["all"] = pop > 0
    return masks, group_ids


def target_groups(items: np.ndarray, pop: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-pop[items], kind="stable")
    groups = {"all": np.arange(len(items))}
    groups.update({name: idx for name, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))})
    return groups


def apply_group_bias(
    scores: torch.Tensor,
    item_group_mask: torch.Tensor,
    candidate_cutoff: int,
    alpha: float,
) -> torch.Tensor:
    if alpha == 0:
        return scores
    reranked = scores.clone()
    k = min(candidate_cutoff, scores.size(1))
    top_items = torch.topk(scores, k=k, dim=1).indices
    candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
    candidate_mask.scatter_(1, top_items, True)
    reranked[candidate_mask & item_group_mask.unsqueeze(0)] += float(alpha)
    return reranked


def collect_eval(
    model,
    eval_data,
    item_group_mask: torch.Tensor,
    candidate_cutoff: int,
    alpha: float,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
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

            scores = apply_group_bias(scores, item_group_mask, candidate_cutoff, alpha)
            pos_score = scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1
            output["items"].extend(positive_i.cpu().numpy().tolist())
            output["ranks"].extend(ranks.cpu().numpy().tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def metric_at(ranks: np.ndarray, k: int) -> tuple[float, float]:
    ranks = ranks.astype(np.float64)
    if len(ranks) == 0:
        return math.nan, math.nan
    hit = ranks <= k
    return float(hit.mean()), float((hit / np.log2(ranks + 1.0)).mean())


def grouped_metrics(items: np.ndarray, ranks: np.ndarray, pop: np.ndarray, cutoffs: list[int]) -> list[dict]:
    groups = target_groups(items.astype(np.int64), pop)
    rows = []
    for group, idx in groups.items():
        row = {"group": group, "n": int(len(idx))}
        group_ranks = ranks[idx]
        for k in cutoffs:
            recall, ndcg = metric_at(group_ranks, k)
            row[f"recall@{k}"] = recall
            row[f"ndcg@{k}"] = ndcg
        row["mean_rank"] = float(group_ranks.mean()) if len(group_ranks) else math.nan
        row["median_rank"] = float(np.median(group_ranks)) if len(group_ranks) else math.nan
        rows.append(row)
    return rows


def score_for_selection(rows: list[dict], objective: str, k: int) -> float:
    lookup = {row["group"]: row for row in rows}
    if objective == "overall_ndcg":
        return lookup["all"][f"ndcg@{k}"]
    if objective == "tail_ndcg":
        return lookup["tail"][f"ndcg@{k}"]
    if objective == "balanced_ndcg":
        return float(np.mean([lookup[g][f"ndcg@{k}"] for g in ["head", "mid", "tail"]]))
    if objective == "overall_recall":
        return lookup["all"][f"recall@{k}"]
    if objective == "tail_recall":
        return lookup["tail"][f"recall@{k}"]
    raise ValueError(f"Unknown objective: {objective}")


def flatten_rows(meta: dict, split: str, setting: dict, metric_rows: list[dict], base_rows: list[dict]) -> list[dict]:
    base_by_group = {row["group"]: row for row in base_rows}
    rows = []
    for metric_row in metric_rows:
        group = metric_row["group"]
        row = {**meta, **setting, "split": split, "group": group, "n": metric_row["n"]}
        for key, value in metric_row.items():
            if key in {"group", "n"}:
                continue
            row[f"rerank_{key}"] = value
            if key in base_by_group[group]:
                base_value = base_by_group[group][key]
                row[f"base_{key}"] = base_value
                if isinstance(value, float) and isinstance(base_value, float):
                    row[f"delta_{key}"] = value - base_value
                    row[f"rel_{key}"] = value / base_value - 1 if base_value > 0 else math.nan
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


def write_markdown(path: Path, rows: list[dict], k: int) -> None:
    headers = [
        "dataset",
        "backbone",
        "hidden",
        "eligible",
        "candidate_cutoff",
        "selected_alpha",
        "selection_objective",
        "group",
        f"base_recall@{k}",
        f"rerank_recall@{k}",
        f"rel_recall@{k}",
        f"base_ndcg@{k}",
        f"rerank_ndcg@{k}",
        f"rel_ndcg@{k}",
        "base_median_rank",
        "rerank_median_rank",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default=None)
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
    parser.add_argument("--candidate_cutoffs", default="50,100")
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--eligible_sets", default="tail:tail,mid_tail:mid+tail")
    parser.add_argument("--cutoffs", default="10,20,50,100")
    parser.add_argument("--selection_k", default=10, type=int)
    parser.add_argument(
        "--selection_objective",
        default="balanced_ndcg",
        choices=["overall_ndcg", "tail_ndcg", "balanced_ndcg", "overall_recall", "tail_recall"],
    )
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    config, dataset, train_data, valid_data, test_data, model = build_model(args.model, args.checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    item_group_masks, _ = popularity_groups(pop)

    cutoffs = parse_list(args.cutoffs, int)
    candidate_cutoffs = parse_list(args.candidate_cutoffs, int)
    alphas = parse_list(args.alphas, float)
    eligible_sets = []
    for spec in args.eligible_sets.split(","):
        if not spec.strip():
            continue
        label, groups = spec.split(":", 1)
        eligible_sets.append((label, groups.split("+")))

    device = next(model.parameters()).device
    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "backbone": args.backbone or args.model.replace("CANDS", ""),
        "hidden": args.hidden_size,
        "max_len": args.max_item_list_length,
        "temperature": args.temperature,
    }

    all_rows = []
    selected_rows = []
    for eligible_label, group_names in eligible_sets:
        eligible_mask_np = np.zeros(dataset.item_num, dtype=bool)
        for group_name in group_names:
            eligible_mask_np |= item_group_masks[group_name]
        eligible_mask = torch.as_tensor(eligible_mask_np, device=device)

        for candidate_cutoff in candidate_cutoffs:
            base_valid = collect_eval(model, valid_data, eligible_mask, candidate_cutoff, alpha=0.0, max_batches=args.max_batches)
            base_valid_rows = grouped_metrics(base_valid["items"], base_valid["ranks"], pop, cutoffs)
            best = None
            valid_cache = {}
            for alpha in alphas:
                valid_stats = collect_eval(
                    model, valid_data, eligible_mask, candidate_cutoff, alpha=alpha, max_batches=args.max_batches
                )
                valid_rows = grouped_metrics(valid_stats["items"], valid_stats["ranks"], pop, cutoffs)
                valid_cache[alpha] = valid_rows
                score = score_for_selection(valid_rows, args.selection_objective, args.selection_k)
                if best is None or score > best["score"]:
                    best = {"alpha": alpha, "score": score}
                setting = {
                    "eligible": eligible_label,
                    "candidate_cutoff": candidate_cutoff,
                    "alpha": alpha,
                    "selected_alpha": "",
                    "selection_objective": args.selection_objective,
                    "selection_score": score,
                }
                all_rows.extend(flatten_rows(meta, "valid", setting, valid_rows, base_valid_rows))

            assert best is not None
            base_test = collect_eval(model, test_data, eligible_mask, candidate_cutoff, alpha=0.0, max_batches=args.max_batches)
            base_test_rows = grouped_metrics(base_test["items"], base_test["ranks"], pop, cutoffs)
            test_stats = collect_eval(
                model, test_data, eligible_mask, candidate_cutoff, alpha=best["alpha"], max_batches=args.max_batches
            )
            test_rows = grouped_metrics(test_stats["items"], test_stats["ranks"], pop, cutoffs)
            selected_setting = {
                "eligible": eligible_label,
                "candidate_cutoff": candidate_cutoff,
                "alpha": best["alpha"],
                "selected_alpha": best["alpha"],
                "selection_objective": args.selection_objective,
                "selection_score": best["score"],
            }
            selected_rows.extend(flatten_rows(meta, "test", selected_setting, test_rows, base_test_rows))

    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_suffix(".all_valid.csv"), all_rows)
    write_csv(out_prefix.with_suffix(".selected_test.csv"), selected_rows)
    write_markdown(out_prefix.with_suffix(".selected_test.md"), selected_rows, args.selection_k)
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"selected_test": selected_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote group-bias rerank outputs with prefix {out_prefix}")


if __name__ == "__main__":
    main()
