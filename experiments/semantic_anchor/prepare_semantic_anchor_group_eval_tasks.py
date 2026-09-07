#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def fmt_float(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def run_name(row: dict) -> str:
    dataset = row["dataset"]
    model = row["model"]
    hidden = str(int(float(row["hidden"])))
    max_len = str(int(float(row["max_len"])))
    temp = fmt_float(row["temp"])
    if model == "CANDSSASRec":
        return f"{dataset}_{model}_h{hidden}_len{max_len}_temp{temp}"
    if row.get("text_model"):
        text_dim = str(int(float(row["text_dim"])))
        source = f"text{row['text_model']}_dim{text_dim}"
    else:
        svd_dim = str(int(float(row["svd_dim"])))
        source = f"svd{svd_dim}"
    mode = row["mode"]
    gate = row["gate"]
    weight = fmt_float(row["weight"])
    return f"{dataset}_{model}_h{hidden}_len{max_len}_temp{temp}_{source}_mode{mode}_gate{gate}_w{weight}"


def latest_checkpoint(ckpt_dir: Path, name: str) -> str:
    checkpoints = sorted((ckpt_dir / name).glob("*.pth"))
    return str(checkpoints[-1]) if checkpoints else ""


def checkpoint_from_log(log_dir: Path, name: str) -> str:
    log_path = log_dir / f"{name}.log"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Loading model structure and parameters from\s+(.+?\.pth)", text)
    if matches:
        return matches[-1].strip().rstrip("\r")
    matches = re.findall(r"Saving current best:\s+(.+?\.pth)", text)
    if matches:
        return matches[-1].strip().rstrip("\r")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", default="log_runs/beauty_tfidf_semantic_anchor_gpu0/semantic_anchor_summary.csv")
    parser.add_argument("--ckpt_dir", default="ckpt/beauty_tfidf_semantic_anchor_gpu0")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--semantic_embedding_path", default="dataset/Beauty/Beauty.tfidf_svd128.npy")
    parser.add_argument("--top_n", default=5, type=int)
    parser.add_argument("--rank_metric", default="ndcg@10")
    parser.add_argument("--out_tsv", default="analysis_results/beauty_tfidf_semantic_anchor_group_eval/tasks.tsv")
    args = parser.parse_args()

    with Path(args.summary_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    baseline = [row for row in rows if row["model"] == "CANDSSASRec"]
    semantic = [row for row in rows if row["model"] == "SemanticCANDSSASRec"]
    semantic.sort(key=lambda row: float(row.get(args.rank_metric) or 0.0), reverse=True)
    selected = baseline[:1] + semantic[: args.top_n]

    tasks = []
    for row in selected:
        name = run_name(row)
        checkpoint = checkpoint_from_log(Path(args.log_dir), name) if args.log_dir else ""
        if not checkpoint:
            checkpoint = latest_checkpoint(Path(args.ckpt_dir), name)
        tasks.append(
            {
                "run_name": name,
                "dataset": row["dataset"],
                "method": row["method"],
                "model": row["model"],
                "hidden": str(int(float(row["hidden"]))),
                "max_len": str(int(float(row["max_len"]))),
                "inner_size": str(int(float(row["hidden"])) * 4),
                "temp": fmt_float(row["temp"]),
                "svd_dim": str(int(float(row["svd_dim"]))) if row.get("svd_dim") else "",
                "text_model": row.get("text_model", ""),
                "text_dim": str(int(float(row["text_dim"]))) if row.get("text_dim") else "",
                "mode": row.get("mode", ""),
                "gate": row.get("gate", ""),
                "weight": fmt_float(row["weight"]) if row.get("weight") else "0",
                "semantic_embedding_path": args.semantic_embedding_path if row["model"] == "SemanticCANDSSASRec" else "",
                "checkpoint": checkpoint,
            }
        )

    headers = [
        "run_name",
        "dataset",
        "method",
        "model",
        "hidden",
        "max_len",
        "inner_size",
        "temp",
        "svd_dim",
        "text_model",
        "text_dim",
        "mode",
        "gate",
        "weight",
        "semantic_embedding_path",
        "checkpoint",
    ]
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)

    print(f"wrote {len(tasks)} tasks to {out_tsv}")
    missing = [task for task in tasks if not task["checkpoint"]]
    if missing:
        print("missing checkpoints:")
        for task in missing:
            print(f"  {task['run_name']}")


if __name__ == "__main__":
    main()
