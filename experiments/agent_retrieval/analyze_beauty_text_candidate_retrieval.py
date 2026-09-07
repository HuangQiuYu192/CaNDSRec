#!/usr/bin/env python3
"""Validate text retrieval as a complement to CaNDS candidates on Beauty.

This is deliberately a *pre-Agent* experiment.  It uses only static catalog
metadata (title, category paths, and description), represents the current
sequence by the average TF-IDF vector of its history items, and compares:

* CaNDS top-K;
* text-retrieval top-K;
* their union (reported with its actual, possibly larger, candidate size);
* a fixed-budget CaNDS/text fusion; and
* a popularity-matched-size random-fill control.

No review text, rating, sales rank, validation interaction, or test
interaction is used to create item documents.  Review evidence needs a
separate split/time-safe construction step and is intentionally not included
here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def clean_text(value: object) -> str:
    """Keep word-like tokens and make Amazon metadata suitable for TF-IDF."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z]+", " ", value).lower()).strip()


def parse_meta(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def item_documents(meta_path: Path, item_map_path: Path, item_num: int) -> list[str]:
    """Build one static catalog document per internal RecBole item ID."""
    raw2id = {key: int(value) for key, value in json.loads(item_map_path.read_text(encoding="utf-8"))["raw2id"].items()}
    documents = [""] * item_num
    matched = 0
    with meta_path.open(encoding="utf-8", errors="ignore") as source:
        for line in source:
            if not line.strip():
                continue
            row = parse_meta(line)
            item_id = raw2id.get(row.get("asin"))
            if item_id is None or item_id >= item_num:
                continue
            categories = row.get("categories") or []
            category_text = " ".join(" ".join(path) for path in categories if isinstance(path, list))
            # Field labels retain the semantic roles while remaining a simple,
            # reproducible lexical baseline rather than an LLM prompt.
            documents[item_id] = " ".join(
                part for part in (
                    "title " + clean_text(row.get("title")),
                    "categories " + clean_text(category_text),
                    "description " + clean_text(row.get("description")),
                ) if part.strip()
            )
            matched += 1
    if matched == 0:
        raise ValueError(f"No RecBole items matched metadata: {meta_path}")
    return documents


def make_text_matrix(documents: list[str], max_features: int, min_df: int, ngram_max: int):
    # Missing evidence must produce a zero vector, rather than a shared
    # "missing" token that could make unrelated no-evidence items retrieve one
    # another (especially important for split-safe review documents).
    valid = [text for text in documents[1:] if text]
    if not valid:
        raise ValueError("No non-empty evidence documents were built.")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
    )
    vectorizer.fit(valid)
    matrix = vectorizer.transform(documents[1:]).tocsr()
    return sparse.vstack([sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32), matrix], format="csr"), vectorizer


def align_architecture_to_checkpoint(args: argparse.Namespace) -> None:
    """Read architecture dimensions from a checkpoint before RecBole builds.

    ``strict=False`` does not suppress tensor-shape mismatches in PyTorch.
    Beauty checkpoints in this project include both ``inner_size=256`` and
    ``inner_size=1024`` runs, so this small inspection avoids silently binding
    a valid checkpoint to the wrong runner default.
    """
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    ffn_key = "trm_encoder.layer.0.feed_forward.dense_1.weight"
    if ffn_key not in state:
        raise KeyError(f"Cannot infer FFN width: checkpoint lacks {ffn_key}")
    checkpoint_inner = int(state[ffn_key].shape[0])
    if args.inner_size != checkpoint_inner:
        print(f"checkpoint architecture: overriding --inner_size {args.inner_size} -> {checkpoint_inner}")
        args.inner_size = checkpoint_inner


def target_groups(items: np.ndarray, popularity: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-popularity[items], kind="stable")
    return {"all": np.arange(len(items)), **dict(zip(["head", "mid", "tail"], np.array_split(order, 3)))}


def recall_at(targets: np.ndarray, candidates: list[np.ndarray], indices: np.ndarray) -> float:
    if len(indices) == 0:
        return math.nan
    return float(np.mean([targets[i] in candidates[i] for i in indices]))


def mean_candidate_count(candidates: list[np.ndarray], indices: np.ndarray) -> float:
    return float(np.mean([len(candidates[i]) for i in indices])) if len(indices) else math.nan


def top_ids(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top IDs deterministically, excluding index 0 and masked entries."""
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, len(scores) - 1)
    ids = np.argpartition(-scores, kth=k - 1)[:k]
    return ids[np.lexsort((ids, -scores[ids]))].astype(np.int64, copy=False)


def unique_merge(*parts: np.ndarray, limit: int | None = None) -> np.ndarray:
    seen, merged = set(), []
    for part in parts:
        for item in part.tolist():
            if item not in seen:
                seen.add(item)
                merged.append(item)
                if limit is not None and len(merged) >= limit:
                    return np.asarray(merged, dtype=np.int64)
    return np.asarray(merged, dtype=np.int64)


def history_text_scores(item_seq: np.ndarray, matrix: sparse.csr_matrix) -> np.ndarray:
    history = item_seq[item_seq > 0]
    if len(history) == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    query = matrix[history].mean(axis=0)
    return np.asarray(query @ matrix.T).ravel().astype(np.float32, copy=False)


def collect_candidates(model, eval_data, matrix, topk: int, cands_quota: int, text_quota: int, seed: int, max_batches: int | None):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    outputs = {key: [] for key in ["items", "cands", "text", "union", "fusion", "random_control"]}
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
            cands = top_ids(cands_scores[row], topk)
            text_scores = history_text_scores(sequences[row], matrix)
            text_scores[0] = -np.inf
            # Sequential full-sort evaluation excludes every previously seen
            # item. Apply exactly the same mask to text retrieval.
            text_scores[sequences[row][sequences[row] > 0]] = -np.inf
            text = top_ids(text_scores, topk)
            fusion = unique_merge(cands[:cands_quota], text[:text_quota], limit=topk)
            # Same CaNDS quota and final budget as fusion, but the added slots
            # contain random unseen items. It controls for candidate count.
            available = np.flatnonzero(np.isfinite(cands_scores[row]))
            allowed = available[~np.isin(available, cands[:cands_quota])]
            random_fill = rng.choice(allowed, size=min(text_quota, len(allowed)), replace=False) if len(allowed) else np.empty(0, dtype=np.int64)
            random_control = unique_merge(cands[:cands_quota], random_fill, limit=topk)
            outputs["items"].append(target)
            outputs["cands"].append(cands)
            outputs["text"].append(text)
            outputs["union"].append(unique_merge(cands, text))
            outputs["fusion"].append(fusion)
            outputs["random_control"].append(random_control)
    outputs["items"] = np.asarray(outputs["items"], dtype=np.int64)
    return outputs


def summarize(stats: dict, popularity: np.ndarray, topk: int, cands_quota: int, text_quota: int) -> list[dict]:
    variants = {
        f"cands_top{topk}": stats["cands"],
        f"text_top{topk}": stats["text"],
        f"union_up_to_{2 * topk}": stats["union"],
        f"fixed_budget_cands{cands_quota}_text{text_quota}": stats["fusion"],
        f"fixed_budget_cands{cands_quota}_random{text_quota}": stats["random_control"],
    }
    rows = []
    for group, indices in target_groups(stats["items"], popularity).items():
        row_base = {"group": group, "n": int(len(indices))}
        for name, candidates in variants.items():
            row = {**row_base, "variant": name, "candidate_budget": topk if name != f"union_up_to_{2 * topk}" else 2 * topk}
            row[f"candidate_recall@{topk}"] = recall_at(stats["items"], candidates, indices)
            row["mean_candidate_count"] = mean_candidate_count(candidates, indices)
            rows.append(row)
    return rows


def write_outputs(
    out_prefix: Path, rows: list[dict], stats: dict, args, matrix_shape: tuple[int, int], evidence_info: dict | None = None,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "n", "variant", "candidate_budget", f"candidate_recall@{args.topk}", "mean_candidate_count"]
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    evidence_info = evidence_info or {
        "label": "static `title + categories + description`",
        "reviews_used": False,
        "safety": "No reviews are indexed.",
    }
    lines = [f"# Beauty {evidence_info['label']} candidate-retrieval validation", "", evidence_info["safety"], "", "| group | n | variant | candidate budget | Candidate Recall@%d | mean candidates |" % args.topk, "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| {group} | {n} | {variant} | {candidate_budget} | {recall:.4f} | {mean:.2f} |".format(group=row["group"], n=row["n"], variant=row["variant"], candidate_budget=row["candidate_budget"], recall=row[f"candidate_recall@{args.topk}"], mean=row["mean_candidate_count"]))
    lines += ["", "## Interpretation", "", "- `union_up_to_*` has a larger candidate pool and is an upper-bound complementarity diagnostic, not a budget-matched comparison.", "- Compare `fixed_budget_cands*_text*` against `fixed_budget_cands*_random*`: a tail gain here is evidence that textual candidates add value beyond merely replacing some CaNDS candidates.", "- The next stage may add review evidence only after reconstructing a train-only, timestamp-safe review index.", ""]
    Path(f"{out_prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "evidence": evidence_info, "topk": args.topk, "cands_quota": args.cands_quota, "text_quota": args.text_quota, "tfidf_shape": list(matrix_shape), "num_test_instances": int(len(stats["items"])), "max_batches": args.max_batches}
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_fields = ["target_item", "cands_contains_target", "text_contains_target", "union_contains_target", "fusion_contains_target", "random_control_contains_target"]
    with Path(f"{out_prefix}_samples.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=sample_fields)
        writer.writeheader()
        for i, target in enumerate(stats["items"]):
            writer.writerow({"target_item": int(target), **{f"{key}_contains_target": int(target in stats[key][i]) for key in ["cands", "text", "union", "fusion", "random_control"]}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw_meta", default="dataset/process/raw/meta_Beauty.json")
    parser.add_argument("--item_map", default="dataset/Beauty/Beauty_item_id_map.json")
    parser.add_argument("--out_prefix", default="analysis_results/agent_retrieval/beauty_text_candidate_retrieval/Beauty_h256_len50_temp10")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_heads", default=2, type=int)
    parser.add_argument("--inner_size", default=256, type=int)
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
    parser.add_argument("--max_batches", default=None, type=int)
    args = parser.parse_args()
    if args.cands_quota + args.text_quota != args.topk:
        raise ValueError("--cands_quota + --text_quota must equal --topk for a budget-matched fusion.")

    align_architecture_to_checkpoint(args)
    # Importing the project model registry also imports optional model
    # dependencies (for example Faiss).  Delay it until the actual checkpoint
    # evaluation so metadata auditing/index construction remains independently
    # testable on a lightweight local Python installation.
    from experiments.cross_dataset.analyze_group_metrics import build_model

    config, dataset, train_data, _, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    documents = item_documents(Path(args.raw_meta), Path(args.item_map), dataset.item_num)
    matrix, _ = make_text_matrix(documents, args.max_features, args.min_df, args.ngram_max)
    item_field = config["ITEM_ID_FIELD"]
    popularity = np.bincount(train_data.dataset.inter_feat[item_field].cpu().numpy(), minlength=dataset.item_num)
    stats = collect_candidates(model, test_data, matrix, args.topk, args.cands_quota, args.text_quota, args.seed, args.max_batches)
    rows = summarize(stats, popularity, args.topk, args.cands_quota, args.text_quota)
    out_prefix = Path(args.out_prefix)
    write_outputs(out_prefix, rows, stats, args, matrix.shape)
    print(f"wrote {out_prefix}_summary.csv, .md, _samples.csv and _meta.json")


if __name__ == "__main__":
    main()
