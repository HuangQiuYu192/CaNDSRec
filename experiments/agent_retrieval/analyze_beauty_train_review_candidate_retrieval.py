#!/usr/bin/env python3
"""Evaluate split-safe historical-review retrieval as CaNDS candidate evidence.

An Amazon review is included in an item document only when the exact
``(reviewerID, asin, unixReviewTime)`` event occurs as a training target in
the RecBole training split.  Validation and test events are never used.  This
is split-safe evidence; it is deliberately labelled as such rather than
claiming a stricter globally timestamp-causal protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import (
    align_architecture_to_checkpoint,
    clean_text,
    collect_candidates,
    make_text_matrix,
    summarize,
    write_outputs,
)


def raw_id_lookup(map_path: Path) -> dict[int, str]:
    mapping = json.loads(map_path.read_text(encoding="utf-8"))["id2raw"]
    return {int(key): value for key, value in mapping.items()}


def train_review_keys(train_data, config, user_map: dict[int, str], item_map: dict[int, str]) -> set[tuple[str, str, int]]:
    """Return raw review identifiers that occur in the actual train split."""
    inter = train_data.dataset.inter_feat
    uid_field, iid_field, time_field = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"], config["TIME_FIELD"]
    if time_field not in inter:
        raise ValueError(f"Training interactions have no {time_field!r}; cannot create split-safe review documents.")
    user_ids = inter[uid_field].cpu().numpy()
    item_ids = inter[iid_field].cpu().numpy()
    timestamps = inter[time_field].cpu().numpy()
    return {
        (user_map[int(user_id)], item_map[int(item_id)], int(timestamp))
        for user_id, item_id, timestamp in zip(user_ids, item_ids, timestamps)
        if int(user_id) in user_map and int(item_id) in item_map
    }


def review_documents(
    reviews_path: Path,
    allowed_keys: set[tuple[str, str, int]],
    item_to_id: dict[str, int],
    item_num: int,
    max_reviews_per_item: int,
    max_words_per_review: int,
) -> tuple[list[str], dict]:
    """Aggregate only approved raw reviews into bounded per-item documents."""
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    matched = 0
    with reviews_path.open(encoding="utf-8", errors="ignore") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("reviewerID"), row.get("asin"), int(row.get("unixReviewTime", -1)))
            if key not in allowed_keys:
                continue
            item_id = item_to_id.get(row["asin"])
            if item_id is None or item_id >= item_num:
                continue
            text = clean_text(" ".join([str(row.get("summary", "")), str(row.get("reviewText", ""))]))
            if not text:
                continue
            grouped[item_id].append((key[2], " ".join(text.split()[:max_words_per_review])))
            matched += 1
    documents = [""] * item_num
    used_reviews = 0
    for item_id, reviews in grouped.items():
        # Chronological order makes the fixed truncation deterministic and
        # favours the most recent training-period evidence.
        selected = sorted(reviews, key=lambda pair: pair[0])[-max_reviews_per_item:]
        documents[item_id] = " ".join(f"review {text}" for _, text in selected)
        used_reviews += len(selected)
    return documents, {
        "train_target_keys": len(allowed_keys),
        "raw_reviews_matched_to_train": matched,
        "review_document_items": len(grouped),
        "reviews_used_after_per_item_cap": used_reviews,
        "max_reviews_per_item": max_reviews_per_item,
        "max_words_per_review": max_words_per_review,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw_reviews", default="dataset/process/raw/reviews_Beauty_5.json")
    parser.add_argument("--user_map", default="dataset/Beauty/Beauty_user_id_map.json")
    parser.add_argument("--item_map", default="dataset/Beauty/Beauty_item_id_map.json")
    parser.add_argument("--out_prefix", default="analysis_results/agent_retrieval/beauty_train_review_candidate_retrieval/Beauty_h256_len50_temp10")
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
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--cands_quota", default=75, type=int)
    parser.add_argument("--text_quota", default=25, type=int)
    parser.add_argument("--max_features", default=50000, type=int)
    parser.add_argument("--min_df", default=2, type=int)
    parser.add_argument("--ngram_max", default=2, type=int)
    parser.add_argument("--max_reviews_per_item", default=20, type=int)
    parser.add_argument("--max_words_per_review", default=80, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    args = parser.parse_args()
    if args.cands_quota + args.text_quota != args.topk:
        raise ValueError("--cands_quota + --text_quota must equal --topk.")
    align_architecture_to_checkpoint(args)

    from experiments.cross_dataset.analyze_group_metrics import build_model

    config, dataset, train_data, _, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    user_map, item_map = raw_id_lookup(Path(args.user_map)), raw_id_lookup(Path(args.item_map))
    allowed = train_review_keys(train_data, config, user_map, item_map)
    raw_to_item = {raw_id: item_id for item_id, raw_id in item_map.items()}
    documents, audit = review_documents(Path(args.raw_reviews), allowed, raw_to_item, dataset.item_num, args.max_reviews_per_item, args.max_words_per_review)
    matrix, _ = make_text_matrix(documents, args.max_features, args.min_df, args.ngram_max)
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    stats = collect_candidates(model, test_data, matrix, args.topk, args.cands_quota, args.text_quota, args.seed, args.max_batches)
    rows = summarize(stats, popularity, args.topk, args.cands_quota, args.text_quota)
    evidence = {
        "label": "split-safe training-review",
        "reviews_used": True,
        "safety": "Only reviews matched to RecBole training-target interactions are indexed; validation/test interaction reviews are excluded.",
        **audit,
    }
    out_prefix = Path(args.out_prefix)
    write_outputs(out_prefix, rows, stats, args, matrix.shape, evidence)
    print(json.dumps(audit, ensure_ascii=False))
    print(f"wrote {out_prefix}_summary.csv, .md, _samples.csv and _meta.json")


if __name__ == "__main__":
    main()
