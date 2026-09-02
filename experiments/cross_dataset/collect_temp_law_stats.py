#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def strip_type(name):
    return name.split(":", 1)[0]


def read_inter(path):
    with path.open("r", encoding="utf-8") as f:
        header = [strip_type(x) for x in f.readline().strip().split("\t")]
        user_idx = header.index("user_id")
        item_idx = header.index("item_id")
        time_idx = header.index("timestamp") if "timestamp" in header else None
        user_items = defaultdict(list)
        item_counts = Counter()
        interactions = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            user = parts[user_idx]
            item = parts[item_idx]
            ts = float(parts[time_idx]) if time_idx is not None else interactions
            user_items[user].append((ts, item))
            item_counts[item] += 1
            interactions += 1
    return user_items, item_counts, interactions


def gini(values):
    values = sorted(float(v) for v in values if v > 0)
    if not values:
        return 0.0
    n = len(values)
    total = sum(values)
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def summarize_dataset(dataset_dir, dataset_name):
    inter_path = dataset_dir / dataset_name / f"{dataset_name}.inter"
    user_items, item_counts, interactions = read_inter(inter_path)
    user_num = len(user_items)
    item_num = len(item_counts)
    seq_lens = [len(v) for v in user_items.values()]
    counts = list(item_counts.values())

    probs = [c / interactions for c in counts]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    n_eff = math.exp(entropy)
    density = interactions / max(user_num * item_num, 1)
    sorted_counts = sorted(counts, reverse=True)
    top1 = sorted_counts[0] / interactions if sorted_counts else 0.0
    top10pct_n = max(1, int(math.ceil(0.1 * item_num)))
    top10pct_mass = sum(sorted_counts[:top10pct_n]) / interactions if sorted_counts else 0.0

    return {
        "dataset": dataset_name,
        "users": user_num,
        "items": item_num,
        "interactions": interactions,
        "density": density,
        "avg_seq_len": sum(seq_lens) / max(user_num, 1),
        "median_seq_len": quantile(seq_lens, 0.5),
        "p90_seq_len": quantile(seq_lens, 0.9),
        "max_seq_len": max(seq_lens) if seq_lens else 0,
        "pop_entropy": entropy,
        "effective_items": n_eff,
        "effective_item_ratio": n_eff / max(item_num, 1),
        "pop_gini": gini(counts),
        "top1_item_mass": top1,
        "top10pct_item_mass": top10pct_mass,
        "sqrt_2log_neff": math.sqrt(2.0 * math.log(max(n_eff, 2.0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--datasets", nargs="+", default=["Beauty", "Sports", "Toys", "ML-1M", "Amazon-Books"])
    parser.add_argument("--output", default="experiments/temp_law/dataset_stats.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = [summarize_dataset(data_root, name) for name in args.datasets]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
