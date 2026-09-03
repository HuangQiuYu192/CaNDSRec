#!/usr/bin/env python3
"""Estimate temperature from SASRec dot/cosine score distributions.

The script does not retrain a model. It loads a SASRec checkpoint, samples
candidate items on validation or test batches, and compares the score scale of
dot-product logits with raw cosine logits:

    temp_by_logit_std = std(dot_logits) / std(cosine_logits)

This provides a lightweight diagnostic for whether temperature mainly restores
the logit scale removed by normalization.
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


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p25": percentile(arr, 25),
        "p50": percentile(arr, 50),
        "p75": percentile(arr, 75),
    }


def build_runtime(args: argparse.Namespace):
    sys.argv = [
        sys.argv[0],
        "--model",
        "SASRec",
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
        "--n_heads",
        str(args.n_heads),
        "--inner_size",
        str(args.inner_size),
        "--hidden_dropout_prob",
        str(args.hidden_dropout_prob),
        "--attn_dropout_prob",
        str(args.attn_dropout_prob),
        "--learning_rate",
        str(args.learning_rate),
        "--max_item_list_length",
        str(args.max_item_list_length),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--train_batch_size",
        str(args.train_batch_size),
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
    checkpoint = torch.load(args.checkpoint, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    return config, dataset, valid_data, test_data, model


def sample_negative_items(
    item_num: int,
    positive_items: torch.Tensor,
    sample_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    sample_size = max(1, min(sample_size, item_num - 1))
    sampled = torch.randint(1, item_num, (len(positive_items), sample_size), generator=generator, device=device)
    collision = sampled.eq(positive_items.unsqueeze(1))
    if collision.any():
        sampled[collision] = sampled[collision].remainder(item_num - 1) + 1
    return sampled


def collect_stats(args: argparse.Namespace) -> dict[str, float | int | str]:
    config, dataset, valid_data, test_data, model = build_runtime(args)
    data = valid_data if args.split == "valid" else test_data
    device = config["device"]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.sample_seed)

    item_emb = model.item_embedding.weight.detach()
    item_dir = F.normalize(item_emb, dim=-1)
    dot_logits: list[float] = []
    cosine_logits: list[float] = []
    pos_dot: list[float] = []
    pos_cos: list[float] = []
    neg_dot: list[float] = []
    neg_cos: list[float] = []
    margin_dot: list[float] = []
    margin_cos: list[float] = []

    batches = 0
    examples = 0
    with torch.no_grad():
        for batched_data in data:
            if args.max_batches is not None and batches >= args.max_batches:
                break
            interaction = batched_data[0].to(device)
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            seq = interaction[model.ITEM_SEQ]
            seq_len = interaction[model.ITEM_SEQ_LEN]
            seq_out = model.forward(seq, seq_len)[positive_u]
            seq_dir = F.normalize(seq_out, dim=-1)

            sampled_i = sample_negative_items(
                dataset.item_num, positive_i, args.sample_items, generator, device
            )
            cand_i = torch.cat([positive_i.unsqueeze(1), sampled_i], dim=1)
            cand_emb = item_emb[cand_i]
            cand_dir = item_dir[cand_i]

            dot = torch.einsum("bd,bkd->bk", seq_out, cand_emb)
            cos = torch.einsum("bd,bkd->bk", seq_dir, cand_dir)
            dot[:, 0] = dot[:, 0]
            cos[:, 0] = cos[:, 0]

            p_dot = dot[:, 0]
            p_cos = cos[:, 0]
            n_dot = dot[:, 1:]
            n_cos = cos[:, 1:]
            hard_dot = n_dot.max(dim=1).values
            hard_cos = n_cos.max(dim=1).values

            dot_logits.extend(dot.flatten().detach().cpu().numpy().tolist())
            cosine_logits.extend(cos.flatten().detach().cpu().numpy().tolist())
            pos_dot.extend(p_dot.detach().cpu().numpy().tolist())
            pos_cos.extend(p_cos.detach().cpu().numpy().tolist())
            neg_dot.extend(n_dot.flatten().detach().cpu().numpy().tolist())
            neg_cos.extend(n_cos.flatten().detach().cpu().numpy().tolist())
            margin_dot.extend((p_dot - hard_dot).detach().cpu().numpy().tolist())
            margin_cos.extend((p_cos - hard_cos).detach().cpu().numpy().tolist())
            examples += len(positive_i)
            batches += 1

    dot_s = summarize(dot_logits)
    cos_s = summarize(cosine_logits)
    margin_dot_s = summarize(margin_dot)
    margin_cos_s = summarize(margin_cos)
    pos_dot_s = summarize(pos_dot)
    pos_cos_s = summarize(pos_cos)
    neg_dot_s = summarize(neg_dot)
    neg_cos_s = summarize(neg_cos)

    temp_by_logit_std = dot_s["std"] / cos_s["std"] if cos_s["std"] > 0 else 0.0
    temp_by_margin_std = margin_dot_s["std"] / margin_cos_s["std"] if margin_cos_s["std"] > 0 else 0.0
    temp_by_pos_neg_gap = (
        (pos_dot_s["mean"] - neg_dot_s["mean"]) / (pos_cos_s["mean"] - neg_cos_s["mean"])
        if abs(pos_cos_s["mean"] - neg_cos_s["mean"]) > 1e-12
        else 0.0
    )

    return {
        "dataset": args.dataset,
        "hidden": args.hidden_size,
        "max_len": args.max_item_list_length,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "sample_items": args.sample_items,
        "batches": batches,
        "examples": examples,
        "dot_std": dot_s["std"],
        "cosine_std": cos_s["std"],
        "temp_by_logit_std": temp_by_logit_std,
        "margin_dot_std": margin_dot_s["std"],
        "margin_cosine_std": margin_cos_s["std"],
        "temp_by_margin_std": temp_by_margin_std,
        "pos_dot_mean": pos_dot_s["mean"],
        "neg_dot_mean": neg_dot_s["mean"],
        "pos_cosine_mean": pos_cos_s["mean"],
        "neg_cosine_mean": neg_cos_s["mean"],
        "temp_by_pos_neg_gap": temp_by_pos_neg_gap,
        "dot_margin_mean": margin_dot_s["mean"],
        "cosine_margin_mean": margin_cos_s["mean"],
        "dot_p25": dot_s["p25"],
        "dot_p50": dot_s["p50"],
        "dot_p75": dot_s["p75"],
        "cosine_p25": cos_s["p25"],
        "cosine_p50": cos_s["p50"],
        "cosine_p75": cos_s["p75"],
    }


def write_outputs(row: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--inner_size", default=256, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--sample_items", default=1024, type=int)
    parser.add_argument("--sample_seed", default=2026, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    row = collect_stats(args)
    write_outputs(row, Path(args.output))
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
