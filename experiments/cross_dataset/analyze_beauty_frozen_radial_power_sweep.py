#!/usr/bin/env python3
"""Frozen-representation radial-power sweep for the score-geometry theorem.

For a trained dot-product SASRec, evaluate ``h/||h||**p_h`` against
``e/||e||**p_e`` without changing a parameter.  Any ``p_h`` with ``p_e=0`` is
a positive candidate-common rescaling and must preserve all ranks.  Candidate
specific item scaling (``p_e>0``) need not do so.  This is intentionally an
inference diagnostic rather than a training result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.analyze_beauty_geometry_mechanisms import build_model, masked


VARIANTS = [
    ("sequence_p025", 0.25, 0.0), ("sequence_p050", 0.50, 0.0),
    ("sequence_p075", 0.75, 0.0), ("sequence_p100", 1.00, 0.0),
    ("item_p025", 0.0, 0.25), ("item_p050", 0.0, 0.50),
    ("item_p075", 0.0, 0.75), ("item_p100", 0.0, 1.00),
    ("joint_p025", 0.25, 0.25), ("joint_p050", 0.50, 0.50),
    ("joint_p075", 0.75, 0.75), ("joint_p100", 1.00, 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def score(sequence: torch.Tensor, items: torch.Tensor, p_h: float, p_e: float) -> torch.Tensor:
    if p_h:
        sequence = sequence / sequence.norm(dim=-1, keepdim=True).clamp_min(1e-12).pow(p_h)
    if p_e:
        items = items / items.norm(dim=-1, keepdim=True).clamp_min(1e-12).pow(p_e)
    return sequence @ items.transpose(0, 1)


def main() -> None:
    args = parse_args()
    # ``build_model`` owns the RecBole configuration construction used by the
    # prior frozen-geometry diagnostics.
    args.model = "SASRec"
    args.tag = "frozen_radial_power_sweep"
    args.geometry = "dot"
    args.temperature = 1.0
    args.sequence_norm_power = 0.0
    args.item_norm_power = 0.0
    config, _, train_data, test_data, model = build_model(args)
    device = config["device"]
    item_weight = model.item_embedding.weight.detach()

    totals = {tag: {"rank_equal": 0, "topk_equal": 0, "jaccard_sum": 0.0,
                    "rank_abs_change_sum": 0.0, "n": 0} for tag, _, _ in VARIANTS}
    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0].to(device)
            history_index = batched_data[1]
            positive_u = torch.as_tensor(batched_data[2], device=device).long()
            positive_i = torch.as_tensor(batched_data[3], device=device).long()
            sequence = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            raw = masked(score(sequence, item_weight, 0.0, 0.0), history_index)
            raw_target = raw[positive_u, positive_i]
            raw_rank = (raw[positive_u] > raw_target.unsqueeze(1)).sum(dim=1) + 1
            raw_topk = torch.topk(raw, k=args.topk, dim=1).indices
            for tag, p_h, p_e in VARIANTS:
                candidate = masked(score(sequence, item_weight, p_h, p_e), history_index)
                candidate_target = candidate[positive_u, positive_i]
                candidate_rank = (candidate[positive_u] > candidate_target.unsqueeze(1)).sum(dim=1) + 1
                candidate_topk = torch.topk(candidate, k=args.topk, dim=1).indices
                total = totals[tag]
                total["rank_equal"] += int((candidate_rank == raw_rank).sum())
                total["topk_equal"] += int((candidate_topk == raw_topk).all(dim=1).sum())
                total["rank_abs_change_sum"] += float((candidate_rank - raw_rank).abs().sum())
                total["jaccard_sum"] += sum(
                    len(set(a.tolist()).intersection(b.tolist())) / args.topk
                    for a, b in zip(raw_topk, candidate_topk)
                )
                total["n"] += positive_i.numel()

    rows = []
    for tag, p_h, p_e in VARIANTS:
        total = totals[tag]
        n = total["n"]
        rows.append({
            "tag": tag, "sequence_norm_power": p_h, "item_norm_power": p_e, "n_test": n,
            "target_rank_equal_rate": total["rank_equal"] / n,
            "topk_exact_equal_rate": total["topk_equal"] / n,
            "topk_jaccard_vs_dot": total["jaccard_sum"] / n,
            "mean_abs_target_rank_change": total["rank_abs_change_sum"] / n,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Frozen radial-power sweep on Beauty", "",
        "A trained dot-product SASRec is held fixed. Each row changes only the final score; no model is retrained.", "",
        "| variant | `(p_h, p_e)` | target-rank equal to dot | exact top-10 equal | mean top-10 Jaccard | mean absolute target-rank change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['tag']} | ({row['sequence_norm_power']:.2f}, {row['item_norm_power']:.2f}) | "
            f"{row['target_rank_equal_rate']:.6f} | {row['topk_exact_equal_rate']:.6f} | "
            f"{row['topk_jaccard_vs_dot']:.6f} | {row['mean_abs_target_rank_change']:.2f} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "For `p_e=0`, score multiplication by the positive, sequence-dependent factor "
        "`||h||^{-p_h}` must preserve every candidate order. Nonzero `p_e` instead rescales candidates "
        "by their own norms and can change ranks. Any deviation of the sequence-only rows from one is numerical noise.",
    ])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}, {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
