#!/usr/bin/env python3
"""Paired user-level bootstrap for SASRec versus CaNDS checkpoints.

Each leave-one-out test example corresponds to one user. Resampling the common
example indices therefore preserves the paired comparison while estimating
uncertainty for metric deltas. This analysis never selects a checkpoint or
hyperparameter.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_group_metrics import build_model, collect_eval, metric_at


def parse_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--base_model", default="SASRec")
    parser.add_argument("--cands_model", default="CANDSSASRec")
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
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260908)
    parser.add_argument("--out", required=True)


def bootstrap_delta(base_ranks: np.ndarray, cands_ranks: np.ndarray, k: int, samples: int, rng: np.random.Generator) -> dict[str, float]:
    n = len(base_ranks)
    base_recall, base_ndcg = metric_at(base_ranks, k)
    cands_recall, cands_ndcg = metric_at(cands_ranks, k)
    draws = rng.integers(0, n, size=(samples, n), endpoint=False)
    b = base_ranks[draws]
    c = cands_ranks[draws]
    delta_recall = (c <= k).mean(axis=1) - (b <= k).mean(axis=1)
    delta_ndcg = ((c <= k) / np.log2(c + 1.0)).mean(axis=1) - ((b <= k) / np.log2(b + 1.0)).mean(axis=1)
    return {
        "k": k,
        "n_users": n,
        "base_recall": base_recall,
        "cands_recall": cands_recall,
        "delta_recall": cands_recall - base_recall,
        "delta_recall_ci_low": float(np.quantile(delta_recall, 0.025)),
        "delta_recall_ci_high": float(np.quantile(delta_recall, 0.975)),
        "base_ndcg": base_ndcg,
        "cands_ndcg": cands_ndcg,
        "delta_ndcg": cands_ndcg - base_ndcg,
        "delta_ndcg_ci_low": float(np.quantile(delta_ndcg, 0.025)),
        "delta_ndcg_ci_high": float(np.quantile(delta_ndcg, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    args = parser.parse_args()
    _, _, _, _, test_data, base_model = build_model(args.base_model, args.base_checkpoint, args)
    base = collect_eval(base_model, test_data)
    _, _, _, _, test_data, cands_model = build_model(args.cands_model, args.cands_checkpoint, args)
    cands = collect_eval(cands_model, test_data)
    if not np.array_equal(base["items"], cands["items"]):
        raise RuntimeError("paired evaluation examples differ between checkpoints")
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = [bootstrap_delta(base["ranks"], cands["ranks"], k, args.bootstrap_samples, rng) for k in (10, 20, 50)]
    for row in rows:
        row.update({"dataset": args.dataset, "seed": args.seed, "base_model": args.base_model,
                    "cands_model": args.cands_model, "bootstrap_samples": args.bootstrap_samples})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote paired bootstrap to {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
