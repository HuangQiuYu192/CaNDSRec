#!/usr/bin/env python3
"""Analyze rank transitions from dot-product scoring to CaNDS scoring.

The script compares a base sequential recommender and its CaNDS counterpart on
the same test set, then reports how target-item ranks move across top-k
decision boundaries. It is intended to explain cases where many tail items move
up in rank but Recall/NDCG do not improve accordingly.
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


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def hit_metrics(ranks: np.ndarray, ks=(5, 10, 20)) -> dict[str, float]:
    ranks = ranks.astype(np.float64)
    output = {}
    for k in ks:
        hit = ranks <= k
        output[f"recall@{k}"] = float(hit.mean()) if len(ranks) else math.nan
        output[f"ndcg@{k}"] = float((hit / np.log2(ranks + 1.0)).mean()) if len(ranks) else math.nan
    return output


def append_prefixed(row: dict, prefix: str, values: dict) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


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
        cli.extend(
            [
                "--n_heads",
                str(args.n_heads),
                "--attn_dropout_prob",
                str(args.attn_dropout_prob),
            ]
        )
    if model_name in {"WEARec", "CANDSWEARec"}:
        cli.extend(
            [
                "--num_heads",
                str(args.wearec_num_heads),
                "--alpha",
                str(args.wearec_alpha),
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
    return config, dataset, train_data, test_data, model


def target_groups(items: np.ndarray, pop: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-pop[items], kind="stable")
    groups = {"all": np.arange(len(items))}
    groups.update({name: idx for name, idx in zip(["head", "mid", "tail"], np.array_split(order, 3))})
    return groups


def collect_ranks(model, test_data, scoring: str, max_batches: int | None = None) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    item_emb = model.item_embedding.weight.detach()
    item_dir = F.normalize(item_emb, dim=-1)
    output = {"items": [], "ranks": [], "pos_score": [], "pos_cos": []}

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
            if scoring == "dot":
                scores = torch.matmul(seq_out, item_emb.transpose(0, 1))
            elif scoring == "cosine":
                scores = float(getattr(model, "temperature", 1.0)) * cos_scores
            else:
                raise ValueError(f"Unknown scoring: {scoring}")

            scores[:, 0] = -float("inf")
            cos_scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")
                cos_scores[history_index] = -float("inf")

            pos_score = scores[positive_u, positive_i]
            pos_cos = cos_scores[positive_u, positive_i]
            ranks = (scores[positive_u] > pos_score.unsqueeze(1)).sum(dim=1) + 1

            output["items"].extend(positive_i.cpu().numpy().tolist())
            output["ranks"].extend(ranks.cpu().numpy().tolist())
            output["pos_score"].extend(pos_score.cpu().numpy().tolist())
            output["pos_cos"].extend(pos_cos.cpu().numpy().tolist())

    return {key: np.asarray(value) for key, value in output.items()}


def bucket_counts(base_rank: np.ndarray, cands_rank: np.ndarray) -> list[dict[str, int]]:
    buckets = [
        ("1-10", 1, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        ("501-1000", 501, 1000),
        (">1000", 1001, np.inf),
    ]
    rows = []
    for label, low, high in buckets:
        before = (base_rank >= low) & (base_rank <= high)
        after = (cands_rank >= low) & (cands_rank <= high)
        rows.append(
            {
                "bucket": label,
                "base_count": int(before.sum()),
                "cands_count": int(after.sum()),
                "count_delta": int(after.sum() - before.sum()),
            }
        )
    return rows


def rank_bucket_labels(ranks: np.ndarray) -> np.ndarray:
    labels = np.full(len(ranks), ">1000", dtype=object)
    ranges = [
        ("1-10", 1, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        ("501-1000", 501, 1000),
    ]
    for label, low, high in ranges:
        labels[(ranks >= low) & (ranks <= high)] = label
    return labels


def transition_matrix_rows(group: str, base_rank: np.ndarray, cands_rank: np.ndarray) -> list[dict]:
    labels = ["1-10", "11-20", "21-50", "51-100", "101-500", "501-1000", ">1000"]
    base_labels = rank_bucket_labels(base_rank)
    cands_labels = rank_bucket_labels(cands_rank)
    rows = []
    for base_label in labels:
        for cands_label in labels:
            count = int(((base_labels == base_label) & (cands_labels == cands_label)).sum())
            if count:
                rows.append(
                    {
                        "group": group,
                        "base_bucket": base_label,
                        "cands_bucket": cands_label,
                        "count": count,
                        "pct_in_group": float(count / len(base_rank)) if len(base_rank) else math.nan,
                    }
                )
    return rows


def delta_bin_rows(group: str, delta_rank: np.ndarray) -> list[dict]:
    bins = [
        ("<=-1000", -np.inf, -1000),
        ("-999:-101", -999, -101),
        ("-100:-21", -100, -21),
        ("-20:-1", -20, -1),
        ("0", 0, 0),
        ("1:20", 1, 20),
        ("21:100", 21, 100),
        ("101:500", 101, 500),
        ("501:1000", 501, 1000),
        (">1000", 1001, np.inf),
    ]
    rows = []
    for label, low, high in bins:
        mask = (delta_rank >= low) & (delta_rank <= high)
        count = int(mask.sum())
        rows.append(
            {
                "group": group,
                "delta_rank_bin": label,
                "count": count,
                "pct_in_group": float(count / len(delta_rank)) if len(delta_rank) else math.nan,
            }
        )
    return rows


def transition_summary(group: str, base_rank: np.ndarray, cands_rank: np.ndarray) -> dict:
    delta_rank = base_rank - cands_rank
    base_hit10 = base_rank <= 10
    cands_hit10 = cands_rank <= 10
    miss10_stay = (~base_hit10) & (~cands_hit10)
    miss10_improved = miss10_stay & (delta_rank > 0)
    miss10_worse = miss10_stay & (delta_rank < 0)
    hit10_gain = (~base_hit10) & cands_hit10
    hit10_loss = base_hit10 & (~cands_hit10)
    hit10_stay = base_hit10 & cands_hit10

    row = {"group": group, "n": int(len(delta_rank))}
    append_prefixed(row, "base", hit_metrics(base_rank))
    append_prefixed(row, "cands", hit_metrics(cands_rank))
    append_prefixed(row, "delta_rank", summarize(delta_rank))
    row["improved_pct"] = float((delta_rank > 0).mean()) if len(delta_rank) else math.nan
    row["worse_pct"] = float((delta_rank < 0).mean()) if len(delta_rank) else math.nan
    row["unchanged_pct"] = float((delta_rank == 0).mean()) if len(delta_rank) else math.nan
    row["hit10_stay"] = int(hit10_stay.sum())
    row["hit10_gain"] = int(hit10_gain.sum())
    row["hit10_loss"] = int(hit10_loss.sum())
    row["miss10_stay"] = int(miss10_stay.sum())
    row["miss10_improved"] = int(miss10_improved.sum())
    row["miss10_worse"] = int(miss10_worse.sum())
    row["net_hit10"] = row["hit10_gain"] - row["hit10_loss"]
    row["miss10_improved_pct"] = float(miss10_improved.sum() / miss10_stay.sum()) if miss10_stay.sum() else math.nan
    row["miss10_worse_pct"] = float(miss10_worse.sum() / miss10_stay.sum()) if miss10_stay.sum() else math.nan
    row["avg_new_hit_rank"] = float(cands_rank[hit10_gain].mean()) if hit10_gain.sum() else math.nan
    row["avg_lost_hit_rank"] = float(base_rank[hit10_loss].mean()) if hit10_loss.sum() else math.nan
    row["base_to_cands_recall@10_delta"] = row["cands_recall@10"] - row["base_recall@10"]
    row["base_to_cands_ndcg@10_delta"] = row["cands_ndcg@10"] - row["base_ndcg@10"]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    headers = [
        "group",
        "n",
        "base_recall@10",
        "cands_recall@10",
        "base_ndcg@10",
        "cands_ndcg@10",
        "delta_rank_mean",
        "delta_rank_p50",
        "delta_rank_p75",
        "improved_pct",
        "worse_pct",
        "hit10_gain",
        "hit10_loss",
        "net_hit10",
        "miss10_improved",
        "miss10_worse",
        "miss10_improved_pct",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(path: Path, summary_rows: list[dict], bucket_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib is unavailable, skip plot: {exc}")
        return

    groups = [row["group"] for row in summary_rows if row["group"] != "all"]
    if not groups:
        return
    net_hit = [next(row for row in summary_rows if row["group"] == g)["net_hit10"] for g in groups]
    miss_improved = [next(row for row in summary_rows if row["group"] == g)["miss10_improved_pct"] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    axes[0].bar(groups, net_hit, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Net Hit@10 Transition")
    axes[0].set_ylabel("gain - loss")

    axes[1].bar(groups, miss_improved, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Miss@10 But Rank Improved")
    axes[1].set_ylabel("fraction")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--cands_model", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
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
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    config, dataset, train_data, test_data, base_model = build_model(args.base_model, args.base_checkpoint, args)
    _, _, _, test_data_cands, cands_model = build_model(args.cands_model, args.cands_checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(
        train_data.dataset.inter_feat[item_field].cpu().numpy(),
        minlength=dataset.item_num,
    )

    base_stats = collect_ranks(base_model, test_data, "dot", args.max_batches)
    cands_stats = collect_ranks(cands_model, test_data_cands, "cosine", args.max_batches)
    if len(base_stats["ranks"]) != len(cands_stats["ranks"]):
        raise RuntimeError("Base and CaNDS runs produced different numbers of test targets.")
    if not np.array_equal(base_stats["items"], cands_stats["items"]):
        raise RuntimeError("Base and CaNDS test targets are not aligned.")

    groups = target_groups(base_stats["items"].astype(np.int64), pop)
    summary_rows = []
    bucket_rows = []
    matrix_rows = []
    delta_rows = []
    sample_rows = []
    for group, idx in groups.items():
        base_rank = base_stats["ranks"][idx].astype(np.int64)
        cands_rank = cands_stats["ranks"][idx].astype(np.int64)
        delta_rank = base_rank - cands_rank
        summary_rows.append(transition_summary(group, base_rank, cands_rank))
        for bucket_row in bucket_counts(base_rank, cands_rank):
            bucket_row.update({"group": group, "n": int(len(idx))})
            bucket_rows.append(bucket_row)
        matrix_rows.extend(transition_matrix_rows(group, base_rank, cands_rank))
        delta_rows.extend(delta_bin_rows(group, delta_rank))
        for local_id in idx:
            base_rank_i = int(base_stats["ranks"][local_id])
            cands_rank_i = int(cands_stats["ranks"][local_id])
            sample_rows.append(
                {
                    "group": group,
                    "item_id": int(base_stats["items"][local_id]),
                    "item_popularity": int(pop[int(base_stats["items"][local_id])]),
                    "base_rank": base_rank_i,
                    "cands_rank": cands_rank_i,
                    "delta_rank": base_rank_i - cands_rank_i,
                    "base_hit10": int(base_rank_i <= 10),
                    "cands_hit10": int(cands_rank_i <= 10),
                    "base_pos_cos": float(base_stats["pos_cos"][local_id]),
                    "cands_pos_cos": float(cands_stats["pos_cos"][local_id]),
                }
            )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_prefix.with_suffix(".summary.csv"), summary_rows)
    write_csv(out_prefix.with_suffix(".rank_buckets.csv"), bucket_rows)
    write_csv(out_prefix.with_suffix(".rank_transition_matrix.csv"), matrix_rows)
    write_csv(out_prefix.with_suffix(".delta_bins.csv"), delta_rows)
    write_csv(out_prefix.with_suffix(".samples.csv"), sample_rows)
    write_markdown(out_prefix.with_suffix(".summary.md"), summary_rows)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "rank_buckets": bucket_rows,
                "rank_transition_matrix": matrix_rows,
                "delta_bins": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    maybe_plot(out_prefix.with_suffix(".png"), summary_rows, bucket_rows)
    print(f"wrote rank transition outputs with prefix {out_prefix}")


if __name__ == "__main__":
    main()
