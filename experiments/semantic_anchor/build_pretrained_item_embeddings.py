#!/usr/bin/env python3
"""Build item text embeddings with a small pretrained sentence encoder."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


def parse_field_name(header: str) -> str:
    return header.split(":", 1)[0]


def clean_text(value: str) -> str:
    value = value.replace("'", " ")
    value = value.replace('"', " ")
    value = value.replace("&", " and ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


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
            text = clean_text(". ".join(part for part in parts if part))
            item_ids.append(item_id)
            texts.append(text if text else "unknown item")
    return item_ids, texts


def safe_model_tag(model_name: str) -> str:
    name = model_name.rstrip("/").split("/")[-1]
    name = re.sub(r"[^0-9A-Za-z]+", "", name)
    return name.lower() or "textencoder"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--data_path", default="dataset")
    parser.add_argument("--fields", default="title,categories")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--normalize", default=True, type=lambda v: str(v).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for pretrained text embeddings. "
            "Install it in the recbole environment or use the TF-IDF/SVD script."
        ) from exc

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    dataset_dir = Path(args.data_path) / args.dataset
    item_path = dataset_dir / f"{args.dataset}.item"
    model_tag = safe_model_tag(args.model_name)
    out_path = Path(args.out) if args.out else dataset_dir / f"{args.dataset}.{model_tag}.npy"
    meta_path = out_path.with_suffix(".meta.json")

    item_ids, texts = read_item_texts(item_path, fields)
    max_item_id = max(item_ids)

    model = SentenceTransformer(args.model_name, device=args.device)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    ).astype(np.float32)

    matrix = np.zeros((max_item_id + 1, emb.shape[1]), dtype=np.float32)
    for item_id, vector in zip(item_ids, emb):
        matrix[item_id] = vector

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)
    meta = {
        "dataset": args.dataset,
        "item_file": str(item_path),
        "output": str(out_path),
        "fields": fields,
        "model_name": args.model_name,
        "model_tag": model_tag,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "batch_size": args.batch_size,
        "device": args.device,
        "normalize": args.normalize,
        "num_items": len(item_ids),
        "matrix_shape": list(matrix.shape),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
