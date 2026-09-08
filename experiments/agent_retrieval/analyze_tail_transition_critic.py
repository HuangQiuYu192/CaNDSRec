#!/usr/bin/env python3
"""Stage-2 validation for a behaviour-only tail-continuation critic.

The critic never reads an embedding or test/validation target.  It ranks
Stage-1 train-only transition candidates by their observed transition support
and target-side support diversity.  Validation selects a *single* retention
threshold; test is used once as the untouched report.

This is deliberately not a causal click simulator: Beauty has interaction
logs, but no exposed-and-skipped item lists.  The output answers the narrower
question needed before pseudo-training: can a frozen behavioural critic retain
more trustworthy tail continuation proposals than the raw transition rule?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agent_retrieval.analyze_beauty_text_candidate_retrieval import align_architecture_to_checkpoint
from experiments.agent_retrieval.analyze_tail_transition_proposals import sequence_histories
from experiments.agent_retrieval.analyze_tail_transition_support import collect_support
from experiments.cross_dataset.analyze_group_metrics import build_model


def build_transition_evidence(train_data, model, config, eligible_tail: set[int]):
    """Build only train-split direct-edge and target-support evidence."""
    inter = train_data.dataset.inter_feat
    targets = inter[config["ITEM_ID_FIELD"]].cpu().numpy().astype(np.int64)
    sequences = inter[model.ITEM_SEQ].cpu().numpy().astype(np.int64)
    lengths = inter[model.ITEM_SEQ_LEN].cpu().numpy().astype(np.int64)
    edge_counts: dict[int, Counter] = defaultdict(Counter)
    target_contexts = Counter()
    target_predecessors: dict[int, set[int]] = defaultdict(set)
    for target, sequence, length in zip(targets, sequences, lengths):
        length = min(int(length), len(sequence))
        if target not in eligible_tail or length <= 0:
            continue
        predecessor = int(sequence[length - 1])
        if predecessor <= 0:
            continue
        edge_counts[predecessor][int(target)] += 1
        target_contexts[int(target)] += 1
        target_predecessors[int(target)].add(predecessor)
    edge_totals = {predecessor: sum(counts.values()) for predecessor, counts in edge_counts.items()}
    return edge_counts, edge_totals, target_contexts, target_predecessors


def critic_candidates(history, edge_counts, edge_totals, target_contexts, target_predecessors, decay, max_candidates):
    """Rank graph candidates without model scores or angular similarities."""
    aggregate = Counter()
    # This is the Stage-1-style transition rule: each predecessor's outgoing
    # count is normalized before recency-weighted aggregation.
    raw_transition = Counter()
    max_edge = Counter()
    for position, predecessor in enumerate(history):
        recency = float(decay ** (len(history) - 1 - position))
        for target, count in edge_counts.get(int(predecessor), {}).items():
            aggregate[target] += recency * count
            raw_transition[target] += recency * count / edge_totals[int(predecessor)]
            max_edge[target] = max(max_edge[target], count)
    rows = []
    for target, weighted_count in aggregate.items():
        # Each factor is genuine training support.  The logarithms prevent a
        # few less-tail-ish eligible items from dominating solely by frequency.
        edge_strength = math.log1p(weighted_count)
        target_strength = math.log1p(target_contexts[target])
        diversity_strength = math.log1p(len(target_predecessors[target]))
        score = edge_strength + 0.35 * target_strength + 0.35 * diversity_strength
        rows.append((int(target), float(score), float(weighted_count), int(max_edge[target]), int(target_contexts[target]), int(len(target_predecessors[target])), float(raw_transition[target])))
    critic_rows = sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))[:max_candidates]
    raw_rows = sorted(rows, key=lambda row: (-row[6], -row[2], row[0]))[:max_candidates]
    return critic_rows, raw_rows


def collect_split(data, model, edge_counts, edge_totals, target_contexts, target_predecessors, eligible_tail, tail_items, window, decay, max_candidates):
    device = next(model.parameters()).device
    examples = []
    for batched_data in data:
        interaction = batched_data[0].to(device)
        targets = torch.as_tensor(batched_data[3], device=device).long().cpu().numpy().astype(np.int64)
        histories = sequence_histories(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN], window)
        for target, history in zip(targets, histories):
            if int(target) not in tail_items:
                continue
            candidates, raw_candidates = critic_candidates(history, edge_counts, edge_totals, target_contexts, target_predecessors, decay, max_candidates)
            if not candidates:
                continue
            # A target outside the Stage-0 whitelist must never be interpreted
            # as a simulator miss: it was deliberately not eligible to create
            # a pseudo continuation.
            examples.append({"target": int(target), "target_eligible": int(target in eligible_tail), "candidates": candidates, "raw_candidates": raw_candidates})
    return examples


def proposal_rows(examples, threshold=None, candidate_key="candidates"):
    """Evaluate a top proposal, optionally retaining it by Critic threshold."""
    rows = []
    for example in examples:
        target, score, weighted_count, max_edge, contexts, predecessors, raw_score = example[candidate_key][0]
        keep = threshold is None or score >= threshold
        rows.append({
            "target": example["target"], "target_eligible": example["target_eligible"],
            "proposal": target, "critic_score": score, "weighted_edge_count": weighted_count,
            "max_edge_count": max_edge, "target_contexts": contexts,
            "target_unique_predecessors": predecessors, "retained": int(keep),
            "raw_transition_score": raw_score,
            "hit": int(target == example["target"]),
        })
    return rows


def metric_row(split, variant, rows):
    eligible = [row for row in rows if row["target_eligible"]]
    retained = [row for row in eligible if row["retained"]]
    return {
        "split": split, "variant": variant, "eligible_tail_contexts": len(eligible),
        "retained_contexts": len(retained),
        "retention_rate": len(retained) / len(eligible) if eligible else math.nan,
        "proposal_precision": float(np.mean([row["hit"] for row in retained])) if retained else math.nan,
        "proposal_recall": float(sum(row["hit"] for row in retained) / len(eligible)) if eligible else math.nan,
        "mean_critic_score": float(np.mean([row["critic_score"] for row in retained])) if retained else math.nan,
    }


def choose_threshold(valid_examples, quantiles, min_retention):
    scores = np.asarray([example["candidates"][0][1] for example in valid_examples if example["target_eligible"]], dtype=np.float64)
    if not len(scores):
        raise ValueError("No eligible-tail validation proposals. Check Stage-0 whitelist and checkpoint split.")
    thresholds = sorted({float(np.quantile(scores, quantile)) for quantile in quantiles})
    if not any(np.isclose(threshold, scores.min()) for threshold in thresholds):
        thresholds.insert(0, float(scores.min()))
    trials = []
    for threshold in thresholds:
        rows = proposal_rows(valid_examples, threshold)
        metric = metric_row("valid", f"threshold_{threshold:.6f}", rows)
        metric["threshold"] = threshold
        trials.append(metric)
    admissible = [trial for trial in trials if trial["retention_rate"] >= min_retention and not math.isnan(trial["proposal_precision"])]
    if not admissible:
        raise ValueError("No validation threshold satisfies --min_retention; reduce it or inspect Stage-1 support.")
    # Precision is primary: pseudo labels must be trustworthy.  Recall and
    # retention are deterministic tie breakers, never tuned on test.
    best = max(admissible, key=lambda trial: (trial["proposal_precision"], trial["proposal_recall"], trial["retention_rate"]))
    return best, trials


def calibration_rows(split, rows, bins):
    eligible = [row for row in rows if row["target_eligible"]]
    if not eligible:
        return []
    scores = np.asarray([row["critic_score"] for row in eligible])
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    output = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        selected = [row for row in eligible if (row["critic_score"] >= low and (row["critic_score"] < high or index == len(edges) - 2))]
        if selected:
            output.append({"split": split, "bin": index + 1, "score_low": low, "score_high": high, "n": len(selected), "empirical_precision": float(np.mean([row["hit"] for row in selected])), "mean_score": float(np.mean([row["critic_score"] for row in selected]))})
    return output


def write_outputs(out_prefix, selected, trials, test_metrics, calibration, metadata):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = ["split", "variant", "eligible_tail_contexts", "retained_contexts", "retention_rate", "proposal_precision", "proposal_recall", "mean_critic_score", "threshold"]
    summary_rows = trials + test_metrics
    with Path(f"{out_prefix}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with Path(f"{out_prefix}_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "bin", "score_low", "score_high", "n", "empirical_precision", "mean_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(calibration)
    lines = ["# Stage-2 tail-continuation Critic validation", "", "The Critic uses only train-split transition count, target transition contexts, and distinct predecessor support. It does not use item-text, reviews, validation/test labels, or angular similarity.", "", "## Validation threshold selection", "", "| threshold | retained / eligible contexts | retention | proposal precision | proposal recall |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in trials:
        mark = " **selected**" if abs(row["threshold"] - selected["threshold"]) < 1e-12 else ""
        lines.append(f"| {row['threshold']:.6f}{mark} | {row['retained_contexts']} / {row['eligible_tail_contexts']} | {row['retention_rate']:.4f} | {row['proposal_precision']:.4f} | {row['proposal_recall']:.4f} |")
    lines += ["", "## Frozen test result", "", "| variant | eligible contexts | retained contexts | retention | proposal precision | proposal recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in test_metrics:
        lines.append(f"| {row['variant']} | {row['eligible_tail_contexts']} | {row['retained_contexts']} | {row['retention_rate']:.4f} | {row['proposal_precision']:.4f} | {row['proposal_recall']:.4f} |")
    lines += ["", "## Interpretation", "", "- The Critic threshold is selected only on validation. Test rows are a single frozen check.", "- `raw_transition_top1` is the unfiltered Stage-1-style transition ranking. `critic_selected` uses the validation-selected threshold.", "- `proposal_precision` is exact next-item precision of the one retained continuation, so it is conservative: other plausible but unobserved items count as non-hits.", "- Proceed to pseudo-training only if `critic_selected` improves test precision over `raw_transition_top1` at an acceptable retention rate. The accompanying calibration table should be monotonic or nearly so.", ""]
    Path(f"{out_prefix}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
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
    parser.add_argument("--min_contexts", type=int, default=3)
    parser.add_argument("--min_predecessors", type=int, default=2)
    parser.add_argument("--recent_window", type=int, default=5)
    parser.add_argument("--transition_decay", type=float, default=0.7)
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--quantiles", default="0,0.25,0.5,0.75")
    parser.add_argument("--min_retention", type=float, default=0.10)
    parser.add_argument("--calibration_bins", type=int, default=5)
    parser.add_argument("--out_prefix", default="analysis_results/agent_simulator/beauty_tail_transition_critic/Beauty_h256_len50_temp10")
    args = parser.parse_args()
    quantiles = [float(value) for value in args.quantiles.split(",")]
    if not quantiles or any(value < 0 or value > 1 for value in quantiles):
        raise ValueError("--quantiles must be comma-separated values in [0, 1]")
    if not 0 < args.min_retention <= 1:
        raise ValueError("--min_retention must be in (0, 1]")
    align_architecture_to_checkpoint(args)
    config, _, train_data, valid_data, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    _, item_rows = collect_support(train_data, model, config, args.min_contexts, args.min_predecessors)
    tail_items = {row["item_id"] for row in item_rows if row["popularity_group"] == "tail"}
    eligible_tail = {row["item_id"] for row in item_rows if row["popularity_group"] == "tail" and row["eligible_for_simulation"]}
    edge_counts, edge_totals, target_contexts, target_predecessors = build_transition_evidence(train_data, model, config, eligible_tail)
    valid_examples = collect_split(valid_data, model, edge_counts, edge_totals, target_contexts, target_predecessors, eligible_tail, tail_items, args.recent_window, args.transition_decay, args.max_candidates)
    test_examples = collect_split(test_data, model, edge_counts, edge_totals, target_contexts, target_predecessors, eligible_tail, tail_items, args.recent_window, args.transition_decay, args.max_candidates)
    selected, trials = choose_threshold(valid_examples, quantiles, args.min_retention)
    test_rows = proposal_rows(test_examples, selected["threshold"])
    test_metric = metric_row("test", "critic_selected", test_rows)
    test_metric["threshold"] = selected["threshold"]
    raw_test_rows = proposal_rows(test_examples, candidate_key="raw_candidates")
    raw_test_metric = metric_row("test", "raw_transition_top1", raw_test_rows)
    raw_test_metric["threshold"] = math.nan
    calibration = calibration_rows("valid", proposal_rows(valid_examples, selected["threshold"]), args.calibration_bins) + calibration_rows("test", test_rows, args.calibration_bins)
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "eligible_tail_items": len(eligible_tail), "transition_edges": int(sum(len(row) for row in edge_counts.values())), "min_contexts": args.min_contexts, "min_predecessors": args.min_predecessors, "recent_window": args.recent_window, "transition_decay": args.transition_decay, "max_candidates": args.max_candidates, "validation_quantiles": quantiles, "minimum_retention": args.min_retention, "selected_validation_threshold": selected["threshold"]}
    write_outputs(Path(args.out_prefix), selected, trials, [raw_test_metric, test_metric], calibration, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix}_summary.csv, _summary.md, _calibration.csv and _meta.json")


if __name__ == "__main__":
    main()
