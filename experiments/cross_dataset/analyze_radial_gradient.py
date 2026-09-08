#!/usr/bin/env python3
"""Measure radial versus tangential item-embedding gradients.

For a loaded checkpoint and training mini-batches, this script decomposes the
gradient of the *final matching loss* into the component parallel to each
item embedding and the orthogonal remainder.  For CaNDS, Eq. (tangent) in the
paper predicts a numerically zero radial component (up to floating-point
error); dot-product SASRec has no such constraint.  This is a theory check,
not an endpoint evaluation and it never updates model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from analyze_group_metrics import build_model


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--inner_size", type=int, default=1024)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_item_list_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--dot_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--output", required=True)


def probe(model, train_data, max_batches: int) -> dict[str, float]:
    model.train()  # dropout mode matches the training loss geometry.
    radial_energy = tangential_energy = total_energy = 0.0
    active_rows = batches = 0
    device = next(model.parameters()).device
    for interaction in train_data:
        if batches >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = model.calculate_loss(interaction.to(device))
        loss.backward()
        embedding = model.item_embedding.weight.detach()
        grad = model.item_embedding.weight.grad.detach()
        unit = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radial = (grad * unit).sum(dim=-1, keepdim=True) * unit
        tangent = grad - radial
        active = grad.norm(dim=-1) > 0
        radial_energy += float(radial.square().sum().item())
        tangential_energy += float(tangent.square().sum().item())
        total_energy += float(grad.square().sum().item())
        active_rows += int(active.sum().item())
        batches += 1
    if batches == 0 or total_energy <= 0:
        raise RuntimeError("No nonzero item-embedding gradients were collected.")
    return {
        "batches": batches,
        "mean_active_item_rows_per_batch": active_rows / batches,
        "radial_gradient_energy": radial_energy,
        "tangential_gradient_energy": tangential_energy,
        "total_gradient_energy": total_energy,
        "radial_energy_fraction": radial_energy / total_energy,
        "tangential_energy_fraction": tangential_energy / total_energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    rows = []
    for name, checkpoint in (("SASRec", args.dot_checkpoint), ("CANDSSASRec", args.cands_checkpoint)):
        _, _, train_data, _, _, model = build_model(name, checkpoint, args)
        row = probe(model, train_data, args.max_batches)
        row.update({
            "dataset": args.dataset,
            "seed": args.seed,
            "model": name,
            "checkpoint": checkpoint,
            "hidden_size": args.hidden_size,
            "temperature": args.temperature,
        })
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Radial-gradient decomposition",
        "",
        "| model | batches | radial-energy fraction | tangential-energy fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['batches']} | {row['radial_energy_fraction']:.8f} | "
            f"{row['tangential_energy_fraction']:.8f} |"
        )
    lines += [
        "",
        "The result is a loss-gradient identity check. It is not a recommendation metric and must be interpreted alongside matched endpoint and counterfactual analyses.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}, {output.with_suffix('.json')} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
