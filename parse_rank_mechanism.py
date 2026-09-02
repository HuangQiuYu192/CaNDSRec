import csv
import json
import os


BASE = "analysis_results/rank_distribution_cross_dataset"
PAIRS = [
    ("Beauty", "Beauty_sasrec_h64_len50", "Beauty_cands_h64_len50_temp10"),
    ("Yelp-S3Rec", "YelpS3_sasrec_h64_len50", "YelpS3_cands_h64_len50_temp10"),
    ("ML-1M", "ML1M_sasrec_h64_len50", "ML1M_cands_h64_len50_temp20"),
    ("LastFM-S3Rec", "LastFM_sasrec_h64_len200", "LastFM_cands_h64_len200_temp30"),
]
BUCKETS = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", ">100"]
METRICS = [
    "hit@1",
    "hit@5",
    "hit@10",
    "hit@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr",
    "mean_rank",
    "median_rank",
]


def load_summary(name):
    with open(os.path.join(BASE, name + ".summary.json"), encoding="utf-8") as f:
        return json.load(f)


def load_ranks(name):
    rows = []
    with open(os.path.join(BASE, name + ".ranks.tsv"), encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = row.get("uid") or row.get("user_id") or row.get("user")
            rows.append((uid, int(float(row["rank"]))))
    return rows


def bucket(rank):
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return ">100"


def main():
    for dataset, sasrec_name, cands_name in PAIRS:
        sasrec = load_summary(sasrec_name)
        cands = load_summary(cands_name)
        sasrec_ranks = dict(load_ranks(sasrec_name))
        cands_ranks = dict(load_ranks(cands_name))
        users = [uid for uid in sasrec_ranks if uid in cands_ranks]

        improved = sum(cands_ranks[u] < sasrec_ranks[u] for u in users)
        worse = sum(cands_ranks[u] > sasrec_ranks[u] for u in users)
        same = len(users) - improved - worse
        print(f"\n## {dataset}")
        print(
            f"n={len(users)} improved={improved}({improved / len(users):.4f}) "
            f"worse={worse}({worse / len(users):.4f}) same={same}"
        )
        for k in [1, 5, 10, 20, 50, 100]:
            gain = sum(sasrec_ranks[u] > k and cands_ranks[u] <= k for u in users)
            loss = sum(sasrec_ranks[u] <= k and cands_ranks[u] > k for u in users)
            print(f"top{k}: net={gain - loss:+d} gain={gain} loss={loss}")
        print("metrics:")
        for metric in METRICS:
            left = sasrec.get(metric)
            right = cands.get(metric)
            delta = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
            print(f"  {metric}: sasrec={left} cands={right} delta={delta}")
        print("migration rows=SASRec cols=CAND")
        print("\t" + "\t".join(BUCKETS))
        matrix = {src: {dst: 0 for dst in BUCKETS} for src in BUCKETS}
        for user in users:
            matrix[bucket(sasrec_ranks[user])][bucket(cands_ranks[user])] += 1
        for src in BUCKETS:
            print(src + "\t" + "\t".join(str(matrix[src][dst]) for dst in BUCKETS))


if __name__ == "__main__":
    main()
