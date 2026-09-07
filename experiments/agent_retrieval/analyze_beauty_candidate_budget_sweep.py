#!/usr/bin/env python3
"""Budget-matched sweep for metadata/review candidate evidence on Beauty.

Unlike the initial diagnostic, this script makes the 200-candidate comparison
fair by reporting both ``CaNDS Top-200`` and the union of ``CaNDS Top-100``
with ``evidence Top-100``.  At the fixed 100-candidate budget it replaces
exactly q deep CaNDS slots with q evidence candidates, for q in a sweep, and
compares every q with random replacements under the identical budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import (
    align_architecture_to_checkpoint,
    item_documents,
    make_text_matrix,
    target_groups,
    top_ids,
)
from experiments.agent_retrieval.analyze_beauty_train_review_candidate_retrieval import (
    raw_id_lookup,
    review_documents,
    train_review_keys,
)


def merge_to_budget(*parts: np.ndarray, budget: int) -> np.ndarray:
    """Prefer supplement, then restore deep CaNDS IDs until budget is exact."""
    seen, merged = set(), []
    for part in parts:
        for item in part.tolist():
            if item not in seen:
                seen.add(item)
                merged.append(item)
                if len(merged) == budget:
                    return np.asarray(merged, dtype=np.int64)
    return np.asarray(merged, dtype=np.int64)


def candidates_for_sweep(model, eval_data, matrix, budget: int, quotas: list[int], seed: int, max_batches: int | None):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    result = {"items": [], f"cands_top{budget}": [], f"cands_top{2 * budget}": [], f"evidence_top{budget}": [], f"union_cands{budget}_evidence{budget}": []}
    for quota in quotas:
        result[f"fixed_cands{budget - quota}_evidence{quota}"] = []
        result[f"fixed_cands{budget - quota}_random{quota}"] = []

    for batch_index, batched_data in enumerate(eval_data):
        if max_batches is not None and batch_index >= max_batches:
            break
        interaction = batched_data[0].to(device)
        history_index = batched_data[1]
        positive_u = torch.as_tensor(batched_data[2], device=device).long()
        positive_i = torch.as_tensor(batched_data[3], device=device).long()
        with torch.no_grad():
            scores = model.full_sort_predict(interaction)
            if scores.dim() == 1:
                scores = scores.view(positive_i.size(0), -1)
            scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")
            cands_scores = scores[positive_u].detach().float().cpu().numpy()
        sequences = interaction[model.ITEM_SEQ].detach().cpu().numpy()
        for row, target in enumerate(positive_i.cpu().numpy().tolist()):
            cands_200 = top_ids(cands_scores[row], 2 * budget)
            history = sequences[row][sequences[row] > 0]
            if len(history):
                query = matrix[history].mean(axis=0)
                evidence_scores = np.asarray(query @ matrix.T).ravel().astype(np.float32, copy=False)
            else:
                evidence_scores = np.zeros(matrix.shape[0], dtype=np.float32)
            evidence_scores[0] = -np.inf
            evidence_scores[history] = -np.inf
            evidence = top_ids(evidence_scores, budget)
            cands_100 = cands_200[:budget]
            result["items"].append(target)
            result[f"cands_top{budget}"].append(cands_100)
            result[f"cands_top{2 * budget}"].append(cands_200)
            result[f"evidence_top{budget}"].append(evidence)
            result[f"union_cands{budget}_evidence{budget}"].append(merge_to_budget(cands_100, evidence, budget=2 * budget))
            for quota in quotas:
                base = cands_100[:budget - quota]
                # Evidence candidates that duplicate CaNDS are skipped; the
                # remaining deep CaNDS candidates fill any unused slots.
                fusion = merge_to_budget(base, evidence, cands_100[budget - quota:], budget=budget)
                available = np.flatnonzero(np.isfinite(cands_scores[row]))
                # Random replacements must not reuse *any* Top-100 CaNDS ID.
                pool = available[~np.isin(available, cands_100)]
                random_items = rng.choice(pool, size=quota, replace=False)
                random_fusion = merge_to_budget(base, random_items, cands_100[budget - quota:], budget=budget)
                result[f"fixed_cands{budget - quota}_evidence{quota}"].append(fusion)
                result[f"fixed_cands{budget - quota}_random{quota}"].append(random_fusion)
    result["items"] = np.asarray(result["items"], dtype=np.int64)
    return result


def metric_rows(stats: dict, popularity: np.ndarray, budget: int) -> list[dict]:
    rows = []
    for group, indices in target_groups(stats["items"], popularity).items():
        for variant, candidates in stats.items():
            if variant == "items":
                continue
            hits = np.asarray([stats["items"][i] in candidates[i] for i in indices], dtype=np.float32)
            rows.append({
                "group": group,
                "n": int(len(indices)),
                "variant": variant,
                "candidate_budget": int(round(np.mean([len(candidates[i]) for i in indices]))) if len(indices) else 0,
                f"candidate_recall@{budget}": float(hits.mean()) if len(hits) else float("nan"),
                "net_hits": int(hits.sum()),
            })
    return rows


def write_report(out_prefix: Path, rows: list[dict], metadata: dict, budget: int) -> None:
    import csv

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "n", "variant", "candidate_budget", f"candidate_recall@{budget}", "net_hits"]
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Beauty {metadata['evidence_source']} candidate budget sweep", "", metadata["safety"], "", "| group | n | variant | candidate budget | Candidate Recall@%d | net hits |" % budget, "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| {group} | {n} | {variant} | {candidate_budget} | {recall:.4f} | {net_hits} |".format(**row, recall=row[f"candidate_recall@{budget}"]))
    lines += ["", "## Required comparisons", "", f"- Compare `union_cands{budget}_evidence{budget}` against `cands_top{2 * budget}`: both have a 200-candidate budget.", "- For each q, compare `fixed_cands(100-q)_evidenceq` against `fixed_cands(100-q)_randomq`; this isolates evidence quality at a fixed 100-candidate budget.", ""]
    Path(f"{out_prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_evidence(args, dataset, train_data, config):
    if args.evidence_source == "metadata":
        documents = item_documents(Path(args.raw_meta), Path(args.item_map), dataset.item_num)
        return documents, {
            "evidence_source": "static metadata", "safety": "Uses title, category paths, and description only; no reviews are indexed.",
        }
    user_map, item_map = raw_id_lookup(Path(args.user_map)), raw_id_lookup(Path(args.item_map))
    allowed = train_review_keys(train_data, config, user_map, item_map)
    raw_to_item = {raw_id: item_id for item_id, raw_id in item_map.items()}
    documents, audit = review_documents(Path(args.raw_reviews), allowed, raw_to_item, dataset.item_num, args.max_reviews_per_item, args.max_words_per_review)
    return documents, {
        "evidence_source": "split-safe training reviews",
        "safety": "Only reviews matched to RecBole training-target interactions are indexed; validation/test interaction reviews are excluded.",
        **audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence_source", choices=["metadata", "reviews"], required=True)
    parser.add_argument("--raw_meta", default="dataset/process/raw/meta_Beauty.json")
    parser.add_argument("--raw_reviews", default="dataset/process/raw/reviews_Beauty_5.json")
    parser.add_argument("--user_map", default="dataset/Beauty/Beauty_user_id_map.json")
    parser.add_argument("--item_map", default="dataset/Beauty/Beauty_item_id_map.json")
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--inner_size", default=1024, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attn_dropout_prob", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--max_item_list_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--budget", default=100, type=int)
    parser.add_argument("--quotas", default="5,10,25,50")
    parser.add_argument("--max_features", default=50000, type=int)
    parser.add_argument("--min_df", default=2, type=int)
    parser.add_argument("--ngram_max", default=2, type=int)
    parser.add_argument("--max_reviews_per_item", default=20, type=int)
    parser.add_argument("--max_words_per_review", default=80, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    args = parser.parse_args()
    quotas = sorted({int(value) for value in args.quotas.split(",") if value.strip()})
    if any(quota <= 0 or quota >= args.budget for quota in quotas):
        raise ValueError("Every quota must be between 1 and budget-1.")
    align_architecture_to_checkpoint(args)
    from experiments.cross_dataset.analyze_group_metrics import build_model

    config, dataset, train_data, _, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    documents, evidence_meta = build_evidence(args, dataset, train_data, config)
    matrix, _ = make_text_matrix(documents, args.max_features, args.min_df, args.ngram_max)
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    stats = candidates_for_sweep(model, test_data, matrix, args.budget, quotas, args.seed, args.max_batches)
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "budget": args.budget, "quotas": quotas, "tfidf_shape": list(matrix.shape), "num_test_instances": int(len(stats["items"])), "max_batches": args.max_batches, **evidence_meta}
    write_report(Path(args.out_prefix), metric_rows(stats, popularity, args.budget), metadata, args.budget)
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"wrote {args.out_prefix}_summary.csv, .md and _meta.json")


if __name__ == "__main__":
    main()
