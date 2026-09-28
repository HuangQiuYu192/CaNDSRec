#!/usr/bin/env python3
"""Measure score-head radial gradient energy for Beauty SASRec geometries.

The diagnostic differentiates the next-item cross-entropy only through the
final sequence state or only through the candidate embedding table.  It thus
isolates the final scoring geometry from gradients flowing through historical
item lookups in the encoder.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argument_parser import build_config_dict, parse_args
from models import get_model_class
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})


def radial_energy(vector: torch.Tensor, gradient: torch.Tensor) -> tuple[float, float]:
    """Return squared radial-gradient energy and total squared gradient energy."""
    direction = F.normalize(vector, dim=-1)
    radial = (gradient * direction).sum(dim=-1, keepdim=True) * direction
    return float(radial.square().sum().item()), float(gradient.square().sum().item())


def power_normalize(vector: torch.Tensor, power: float) -> torch.Tensor:
    if power == 0.0:
        return vector
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12).pow(power)


def scores(seq: torch.Tensor, items: torch.Tensor, tau: float, seq_power: float, item_power: float) -> torch.Tensor:
    return tau * power_normalize(seq, seq_power) @ power_normalize(items, item_power).transpose(0, 1)


def build_model(args: argparse.Namespace):
    cli = [
        sys.argv[0], "--model", args.model, "--dataset", "Beauty", "--gpu_id", str(args.gpu_id),
        "--seed", str(args.seed), "--hidden_size", "256", "--n_layers", "2", "--n_heads", "2",
        "--inner_size", "1024", "--hidden_dropout_prob", "0.5", "--attn_dropout_prob", "0.5",
        "--learning_rate", "0.001", "--max_item_list_length", "50", "--train_batch_size", str(args.batch_size),
        "--eval_batch_size", "512", "--temperature", str(args.temperature), "--verbose", "False",
        "--show_progress", "False",
    ]
    if args.model == "GeometrySASRec":
        cli += [
            "--score_geometry", args.geometry,
            "--sequence_norm_power", str(args.sequence_norm_power),
            "--item_norm_power", str(args.item_norm_power),
        ]
    sys.argv = cli
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu_id)
    model_class = get_model_class(parsed.model)
    config = Config(model=model_class, dataset=parsed.dataset, config_dict=build_config_dict(parsed))
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    checkpoint = torch.load(args.checkpoint, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    return config, train_data, model


def interaction_from_batch(batch):
    # RecBole train loaders yield Interaction; some versions wrap it in a tuple.
    return batch[0] if isinstance(batch, tuple) else batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["SASRec", "GeometrySASRec"], required=True)
    parser.add_argument("--geometry", default="dot")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--sequence_norm_power", default=0.0, type=float)
    parser.add_argument("--item_norm_power", default=0.0, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu_id", default=3, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_batches", default=32, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config, train_data, model = build_model(args)
    device = config["device"]
    seq_radial = seq_total = item_radial = item_total = 0.0
    losses = []
    batches = 0

    for batch in train_data:
        if batches >= args.max_batches:
            break
        interaction = interaction_from_batch(batch).to(device)
        item_seq = interaction[model.ITEM_SEQ]
        item_seq_len = interaction[model.ITEM_SEQ_LEN]
        positive = interaction[model.POS_ITEM_ID]

        # Sequence-side score-head gradient: candidate table is held fixed.
        sequence = model.forward(item_seq, item_seq_len)
        loss_seq = F.cross_entropy(
            scores(sequence, model.item_embedding.weight.detach(), args.temperature,
                   args.sequence_norm_power, args.item_norm_power),
            positive,
        )
        grad_seq = torch.autograd.grad(loss_seq, sequence, retain_graph=False)[0]
        radial, total = radial_energy(sequence.detach(), grad_seq)
        seq_radial += radial
        seq_total += total
        losses.append(float(loss_seq.item()))

        # Item-side score-head gradient: the encoder output is held fixed.
        sequence = model.forward(item_seq, item_seq_len).detach()
        items = model.item_embedding.weight
        loss_item = F.cross_entropy(
            scores(sequence, items, args.temperature, args.sequence_norm_power, args.item_norm_power),
            positive,
        )
        grad_item = torch.autograd.grad(loss_item, items, retain_graph=False)[0]
        radial, total = radial_energy(items.detach()[1:], grad_item[1:])
        item_radial += radial
        item_total += total
        batches += 1

    row = {
        "dataset": "Beauty", "seed": args.seed, "model": args.model, "geometry": args.geometry,
        "temperature": args.temperature, "sequence_norm_power": args.sequence_norm_power,
        "item_norm_power": args.item_norm_power, "checkpoint": args.checkpoint, "batches": batches,
        "mean_cross_entropy": sum(losses) / len(losses),
        "sequence_radial_energy_ratio": seq_radial / max(seq_total, 1e-30),
        "item_radial_energy_ratio": item_radial / max(item_total, 1e-30),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
