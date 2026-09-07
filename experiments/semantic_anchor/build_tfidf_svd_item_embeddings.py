#!/usr/bin/env python3
"""Build lightweight item text embeddings from RecBole .item files.

This script is intentionally small-model only: it uses TF-IDF followed by
TruncatedSVD, so it can test whether item text itself provides useful semantic
anchors before introducing pretrained text encoders or LLM embeddings.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def parse_field_name(header: str) -> str:
    return header.split(":", 1)[0]


def clean_text(value: str) -> str:
    value = value.replace("'", " ")
    value = value.replace('"', " ")
    value = value.replace("&", " and ")
    value = re.sub(r"[^0-9A-Za-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def read_item_texts(item_path: Path, fields: list[str]) -> tuple[list[int], list[str]]:
    with item_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        names = [parse_field_name(col) for col in header]
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        if "item_id" not in name_to_idx:
            raise ValueError(f"{item_path} has no item_id field")

        missing = [field for field in fields if field not in name_to_idx]
        if missing:
            raise ValueError(f"{item_path} missing text fields: {missing}; available={names}")

        item_ids = []
        texts = []
        for row in reader:
            if not row:
                continue
            item_id = int(row[name_to_idx["item_id"]])
            parts = [row[name_to_idx[field]] if name_to_idx[field] < len(row) else "" for field in fields]
            text = clean_text(" ".join(parts))
            item_ids.append(item_id)
            texts.append(text if text else "unknown item")
    return item_ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--data_path", default="dataset")
    parser.add_argument("--fields", default="title,categories")
    parser.add_argument("--dim", default=128, type=int)
    parser.add_argument("--max_features", default=50000, type=int)
    parser.add_argument("--min_df", default=2, type=int)
    parser.add_argument("--ngram_range", default="1,2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    ngram_range = tuple(int(v.strip()) for v in args.ngram_range.split(","))
    if len(ngram_range) != 2:
        raise ValueError("--ngram_range should be like 1,2")

    dataset_dir = Path(args.data_path) / args.dataset
    item_path = dataset_dir / f"{args.dataset}.item"
    out_path = Path(args.out) if args.out else dataset_dir / f"{args.dataset}.tfidf_svd{args.dim}.npy"
    meta_path = out_path.with_suffix(".meta.json")

    item_ids, texts = read_item_texts(item_path, fields)
    max_item_id = max(item_ids)
    if args.dim >= len(texts):
        raise ValueError(f"dim={args.dim} must be smaller than number of items={len(texts)}")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=args.dim, random_state=2025)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    emb = normalize(emb, norm="l2", axis=1).astype(np.float32)

    matrix = np.zeros((max_item_id + 1, args.dim), dtype=np.float32)
    for item_id, vector in zip(item_ids, emb):
        matrix[item_id] = vector

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)
    meta = {
        "dataset": args.dataset,
        "item_file": str(item_path),
        "output": str(out_path),
        "fields": fields,
        "dim": args.dim,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "ngram_range": ngram_range,
        "num_items": len(item_ids),
        "matrix_shape": list(matrix.shape),
        "tfidf_shape": list(tfidf.shape),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
