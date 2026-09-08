#!/usr/bin/env python3
"""Counterfactual diagnostics for popularity-correlated radial confounding.

The core comparison is deliberately *within a trained model*.  For a fixed
SASRec representation, we rank the same candidates once with the ordinary
dot product and once with cosine similarity.  This separates the effect of
the scoring geometry from differences in optimization, initialization, or
early stopping between two independently trained checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_dataset.analyze_sasrec_normalization_mechanism import (
    build_model,
    pearson,
    spearman,
)


KS = (5, 10, 20, 50, 100)


def population_groups(pop: np.ndarray) -> np.ndarray:
    """Return 0/1/2 for head/mid/tail among items observed in training."""
    labels = np.full(len(pop), -1, dtype=np.int8)
    active = np.flatnonzero(pop > 0)
    active = active[active != 0]
    order = active[np.argsort(-pop[active], kind="stable")]
    for label, part in enumerate(np.array_split(order, 3)):
        labels[part] = label
    return labels


def add_metrics(row, prefix: str, ranks: torch.Tensor) -> None:
    ranks_np = ranks.detach().cpu().numpy()
    for k in KS:
        hit = ranks_np <= k
        row[f"{prefix}_recall@{k}"] = float(hit.mean())
        row[f"{prefix}_ndcg@{k}"] = float((hit / np.log2(ranks_np + 1.0)).mean())
    row[f"{prefix}_mean_rank"] = float(ranks_np.mean())
    row[f"{prefix}_median_rank"] = float(np.median(ranks_np))


def safe_mean(values: torch.Tensor) -> float:
    return float(values.float().mean().item()) if values.numel() else float("nan")


def collect_counterfactual(model, test_data, group_labels, max_batches=None):
    """Evaluate dot/cosine rankings and identify angular wins displaced by norm."""
    device = next(model.parameters()).device
    item_emb = model.item_embedding.weight.detach()
    item_norm = item_emb.norm(dim=-1).clamp_min(1e-12)
    item_dir = item_emb / item_norm.unsqueeze(1)
    chunks = []

    with torch.no_grad():
        for batch_id, batch in enumerate(test_data):
            if max_batches is not None and batch_id >= max_batches:
                break
            interaction, history_index, positive_u, positive_i = batch
            interaction = interaction.to(device)
            positive_u = torch.as_tensor(positive_u, device=device).long()
            positive_i = torch.as_tensor(positive_i, device=device).long()
            seq_out = model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
            seq_norm = seq_out.norm(dim=-1).clamp_min(1e-12)
            seq_dir = seq_out / seq_norm.unsqueeze(1)
            dot = seq_out @ item_emb.T
            cosine = seq_dir @ item_dir.T
            dot[:, 0] = -torch.inf
            cosine[:, 0] = -torch.inf
            if history_index is not None:
                dot[history_index] = -torch.inf
                cosine[history_index] = -torch.inf

            dot_pos = dot[positive_u, positive_i]
            cos_pos = cosine[positive_u, positive_i]
            dot_rank = (dot[positive_u] > dot_pos.unsqueeze(1)).sum(dim=1) + 1
            cos_rank = (cosine[positive_u] > cos_pos.unsqueeze(1)).sum(dim=1) + 1

            # Hardest dot-product competitor, excluding the actual target.
            dot_without_pos = dot[positive_u].clone()
            dot_without_pos[torch.arange(len(positive_i), device=device), positive_i] = -torch.inf
            competitor = dot_without_pos.argmax(dim=1)
            comp_dot = dot_without_pos.max(dim=1).values
            comp_cos = cosine[positive_u, competitor]
            angular_win = cos_pos > comp_cos
            radial_displacement = angular_win & (dot_pos < comp_dot)
            target_group = torch.as_tensor(group_labels[positive_i.cpu().numpy()], device=device)
            comp_group = torch.as_tensor(group_labels[competitor.cpu().numpy()], device=device)
            chunks.append(
                {
                    "target_group": target_group.cpu(),
                    "competitor_group": comp_group.cpu(),
                    "dot_rank": dot_rank.cpu(),
                    "cos_rank": cos_rank.cpu(),
                    "target_norm": item_norm[positive_i].cpu(),
                    "competitor_norm": item_norm[competitor].cpu(),
                    "target_cos": cos_pos.cpu(),
                    "competitor_cos": comp_cos.cpu(),
                    "seq_norm": seq_norm[positive_u].cpu(),
                    "angular_win": angular_win.cpu(),
                    "radial_displacement": radial_displacement.cpu(),
                }
            )
    if not chunks:
        raise RuntimeError("No test examples were collected; check --max_batches.")
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def rows_for_counterfactual(model_name, values, base):
    rows = []
    names = ("all", "head", "mid", "tail")
    masks = [torch.ones_like(values["target_group"], dtype=torch.bool)]
    masks += [values["target_group"] == i for i in range(3)]
    for group, mask in zip(names, masks):
        row = {**base, "analysis": "within_model_counterfactual", "model": model_name, "group": group}
        row["n"] = int(mask.sum())
        add_metrics(row, "dot", values["dot_rank"][mask])
        add_metrics(row, "cosine", values["cos_rank"][mask])
        for k in KS:
            row[f"cosine_minus_dot_recall@{k}"] = row[f"cosine_recall@{k}"] - row[f"dot_recall@{k}"]
            row[f"cosine_minus_dot_ndcg@{k}"] = row[f"cosine_ndcg@{k}"] - row[f"dot_ndcg@{k}"]
            row[f"rank_recovery@{k}"] = float(((values["dot_rank"][mask] > k) & (values["cos_rank"][mask] <= k)).float().mean())
            row[f"rank_regression@{k}"] = float(((values["dot_rank"][mask] <= k) & (values["cos_rank"][mask] > k)).float().mean())
        row["angular_win_rate_vs_dot_competitor"] = safe_mean(values["angular_win"][mask])
        row["radial_displacement_rate"] = safe_mean(values["radial_displacement"][mask])
        row["mean_competitor_to_target_norm_ratio"] = safe_mean(
            values["competitor_norm"][mask] / values["target_norm"][mask].clamp_min(1e-12)
        )
        row["mean_target_cos_minus_competitor_cos"] = safe_mean(
            values["target_cos"][mask] - values["competitor_cos"][mask]
        )
        row["mean_seq_norm"] = safe_mean(values["seq_norm"][mask])
        row["head_competitor_rate"] = safe_mean(values["competitor_group"][mask] == 0)
        rows.append(row)
    return rows


def rows_for_item_norms(model_name, model, pop, group_labels, base):
    norms = model.item_embedding.weight.detach().norm(dim=-1).cpu().numpy()
    active = np.flatnonzero(group_labels >= 0)
    rows = []
    for group, label in [("all", None), ("head", 0), ("mid", 1), ("tail", 2)]:
        idx = active if label is None else active[group_labels[active] == label]
        rows.append(
            {
                **base,
                "analysis": "item_norm_population",
                "model": model_name,
                "group": group,
                "n": int(len(idx)),
                "mean_item_norm": float(norms[idx].mean()),
                "median_item_norm": float(np.median(norms[idx])),
                "p90_item_norm": float(np.percentile(norms[idx], 90)),
                "mean_log1p_popularity": float(np.log1p(pop[idx]).mean()),
                "item_norm_pearson_logpop_all": pearson(norms[active], np.log1p(pop[active])),
                "item_norm_spearman_pop_all": spearman(norms[active], pop[active]),
            }
        )
    return rows


def write_markdown(path: Path, rows):
    lookup = {(r["analysis"], r["model"], r["group"]): r for r in rows}
    lines = [
        "# Directional calibration diagnostics",
        "",
        "The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.",
        "",
        "## Item-norm / popularity coupling",
        "",
        "| model | group | n | mean item norm | Spearman(norm, popularity) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in ("SASRec", "CANDSSASRec"):
        for group in ("all", "head", "mid", "tail"):
            r = lookup[("item_norm_population", model, group)]
            lines.append(f"| {model} | {group} | {r['n']} | {r['mean_item_norm']:.4f} | {r['item_norm_spearman_pop_all']:.4f} |")
    lines += [
        "",
        "## Within-checkpoint dot-versus-cosine counterfactual",
        "",
        "| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("SASRec", "CANDSSASRec"):
        for group in ("all", "head", "mid", "tail"):
            r = lookup[("within_model_counterfactual", model, group)]
            lines.append(
                f"| {model} | {group} | {r['n']} | {r['dot_recall@10']:.4f} | {r['cosine_recall@10']:.4f} | "
                f"{r['cosine_minus_dot_recall@10']:+.4f} | {r['dot_ndcg@10']:.4f} | {r['cosine_ndcg@10']:.4f} | "
                f"{r['angular_win_rate_vs_dot_competitor']:.4f} / {r['radial_displacement_rate']:.4f} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.",
        "- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.",
        "- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_checkpoint", required=True)
    parser.add_argument("--cands_checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    config, dataset, train_data, test_data, sasrec = build_model("SASRec", args.sasrec_checkpoint, args)
    _, _, _, cands_test_data, cands = build_model("CANDSSASRec", args.cands_checkpoint, args)
    item_field = config["ITEM_ID_FIELD"]
    pop = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    labels = population_groups(pop)
    base = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "max_item_list_length": args.max_item_list_length,
        "temperature": args.temperature,
        "sasrec_checkpoint": args.sasrec_checkpoint,
        "cands_checkpoint": args.cands_checkpoint,
    }
    sas_values = collect_counterfactual(sasrec, test_data, labels, args.max_batches)
    cands_values = collect_counterfactual(cands, cands_test_data, labels, args.max_batches)
    rows = []
    rows += rows_for_item_norms("SASRec", sasrec, pop, labels, base)
    rows += rows_for_item_norms("CANDSSASRec", cands, pop, labels, base)
    rows += rows_for_counterfactual("SASRec", sas_values, base)
    rows += rows_for_counterfactual("CANDSSASRec", cands_values, base)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(output.with_suffix(".md"), rows)
    print(f"wrote {len(rows)} rows to {output}, {output.with_suffix('.md')} and JSON")


if __name__ == "__main__":
    main()
