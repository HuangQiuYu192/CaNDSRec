#!/usr/bin/env python3
"""Stage-2.5: validate stricter, history-specific tail-continuation gates.

All candidate evidence is derived from the RecBole training split.  Validation
chooses one pre-declared structural rule; test is held out for a single report.
The experiment still creates no pseudo interactions and trains no new model.
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


def parse_values(text, cast):
    values = sorted({cast(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def build_evidence(train_data, model, config, eligible_tail):
    inter = train_data.dataset.inter_feat
    targets = inter[config["ITEM_ID_FIELD"]].cpu().numpy().astype(np.int64)
    sequences = inter[model.ITEM_SEQ].cpu().numpy().astype(np.int64)
    lengths = inter[model.ITEM_SEQ_LEN].cpu().numpy().astype(np.int64)
    edges = defaultdict(Counter)
    target_contexts = Counter()
    target_predecessors = defaultdict(set)
    for target, sequence, length in zip(targets, sequences, lengths):
        length = min(int(length), len(sequence))
        if target not in eligible_tail or length <= 0:
            continue
        predecessor = int(sequence[length - 1])
        if predecessor <= 0:
            continue
        edges[predecessor][int(target)] += 1
        target_contexts[int(target)] += 1
        target_predecessors[int(target)].add(predecessor)
    edge_totals = {predecessor: sum(counts.values()) for predecessor, counts in edges.items()}
    return edges, edge_totals, target_contexts, target_predecessors


def scored_candidates(history, edges, edge_totals, target_contexts, target_predecessors, decay):
    """Return behaviour-only features for every candidate from this history."""
    weighted = Counter()
    raw_transition = Counter()
    support_predecessors = defaultdict(set)
    max_edge_count = Counter()
    recent_two = Counter()
    for position, predecessor in enumerate(history):
        predecessor = int(predecessor)
        recency = float(decay ** (len(history) - 1 - position))
        in_recent_two = position >= max(0, len(history) - 2)
        for target, count in edges.get(predecessor, {}).items():
            contribution = recency * count
            weighted[target] += contribution
            raw_transition[target] += contribution / edge_totals[predecessor]
            support_predecessors[target].add(predecessor)
            max_edge_count[target] = max(max_edge_count[target], count)
            if in_recent_two:
                recent_two[target] += contribution
    rows = []
    for target, weighted_count in weighted.items():
        critic_score = (
            math.log1p(weighted_count)
            + 0.35 * math.log1p(target_contexts[target])
            + 0.35 * math.log1p(len(target_predecessors[target]))
        )
        rows.append({
            "proposal": int(target), "critic_score": float(critic_score),
            "raw_transition_score": float(raw_transition[target]),
            "history_support_count": len(support_predecessors[target]),
            "max_direct_edge_count": int(max_edge_count[target]),
            "recent_concentration": float(recent_two[target] / weighted_count),
        })
    return rows


def collect_examples(data, model, evidence, eligible_tail, tail_items, window, decay):
    edges, edge_totals, target_contexts, target_predecessors = evidence
    device = next(model.parameters()).device
    examples = []
    for batched_data in data:
        interaction = batched_data[0].to(device)
        targets = torch.as_tensor(batched_data[3], device=device).long().cpu().numpy().astype(np.int64)
        histories = sequence_histories(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN], window)
        for target, history in zip(targets, histories):
            if int(target) not in tail_items:
                continue
            candidates = scored_candidates(history, edges, edge_totals, target_contexts, target_predecessors, decay)
            if candidates:
                examples.append({"target": int(target), "target_eligible": int(target in eligible_tail), "candidates": candidates})
    return examples


def choose_top(example, rule):
    candidates = example["candidates"]
    raw = max(candidates, key=lambda row: (row["raw_transition_score"], row["critic_score"], -row["proposal"]))
    structural = [row for row in candidates if row["history_support_count"] >= rule["min_history_support"] and row["max_direct_edge_count"] >= rule["min_max_edge"] and row["recent_concentration"] >= rule["min_recent_concentration"]]
    strict = max(structural, key=lambda row: (row["critic_score"], row["raw_transition_score"], -row["proposal"])) if structural else None
    return raw, strict


def evaluate_rule(examples, rule, threshold=None):
    records = []
    for example in examples:
        if not example["target_eligible"]:
            continue
        raw, strict = choose_top(example, rule)
        keep = strict is not None and (threshold is None or strict["critic_score"] >= threshold)
        records.append({
            "raw_hit": int(raw["proposal"] == example["target"]),
            "strict_hit": int(keep and strict["proposal"] == example["target"]),
            "kept": int(keep),
            "strict_score": strict["critic_score"] if strict else math.nan,
        })
    n = len(records)
    kept = sum(row["kept"] for row in records)
    strict_hits = sum(row["strict_hit"] for row in records)
    return records, {
        "eligible_tail_contexts": n, "retained_contexts": kept,
        "retention_rate": kept / n if n else math.nan,
        "proposal_precision": strict_hits / kept if kept else math.nan,
        "proposal_recall": strict_hits / n if n else math.nan,
        "raw_transition_precision": sum(row["raw_hit"] for row in records) / n if n else math.nan,
    }


def candidate_thresholds(examples, rule, score_quantiles):
    scores = []
    for example in examples:
        if example["target_eligible"]:
            _, strict = choose_top(example, rule)
            if strict is not None:
                scores.append(strict["critic_score"])
    if not scores:
        return []
    values = np.asarray(scores)
    return sorted({float(np.quantile(values, q)) for q in score_quantiles})


def bootstrap(records, seed, draws):
    if not records:
        return {key: [math.nan, math.nan] for key in ["raw_precision", "strict_precision", "precision_gain", "raw_hit_retention"]}
    raw = np.asarray([row["raw_hit"] for row in records], dtype=np.float64)
    strict = np.asarray([row["strict_hit"] for row in records], dtype=np.float64)
    kept = np.asarray([row["kept"] for row in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    outputs = defaultdict(list)
    for _ in range(draws):
        index = rng.integers(0, len(records), len(records))
        raw_precision = raw[index].mean()
        strict_precision = strict[index].sum() / max(kept[index].sum(), 1)
        outputs["raw_precision"].append(raw_precision)
        outputs["strict_precision"].append(strict_precision)
        outputs["precision_gain"].append(strict_precision - raw_precision)
        outputs["raw_hit_retention"].append(strict[index].sum() / max(raw[index].sum(), 1))
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in outputs.items()}


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
    parser.add_argument("--history_support_values", default="1,2")
    parser.add_argument("--max_edge_values", default="1,2")
    parser.add_argument("--recent_concentration_values", default="0,0.5")
    parser.add_argument("--score_quantiles", default="0,0.75")
    parser.add_argument("--min_retention", type=float, default=0.05)
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--out_prefix", default="analysis_results/agent_simulator/beauty_tail_transition_strict_critic/Beauty_h256_len50_temp10")
    args = parser.parse_args()
    if not 0 < args.min_retention <= 1:
        raise ValueError("--min_retention must be in (0, 1]")
    support_values = parse_values(args.history_support_values, int)
    edge_values = parse_values(args.max_edge_values, int)
    concentration_values = parse_values(args.recent_concentration_values, float)
    score_quantiles = parse_values(args.score_quantiles, float)
    if any(value < 0 or value > 1 for value in concentration_values + score_quantiles):
        raise ValueError("Concentrations and quantiles must be in [0, 1]")
    align_architecture_to_checkpoint(args)
    config, _, train_data, valid_data, test_data, model = build_model("CANDSSASRec", args.checkpoint, args)
    _, item_rows = collect_support(train_data, model, config, args.min_contexts, args.min_predecessors)
    tail_items = {row["item_id"] for row in item_rows if row["popularity_group"] == "tail"}
    eligible_tail = {row["item_id"] for row in item_rows if row["popularity_group"] == "tail" and row["eligible_for_simulation"]}
    evidence = build_evidence(train_data, model, config, eligible_tail)
    valid = collect_examples(valid_data, model, evidence, eligible_tail, tail_items, args.recent_window, args.transition_decay)
    test = collect_examples(test_data, model, evidence, eligible_tail, tail_items, args.recent_window, args.transition_decay)
    trials = []
    for support in support_values:
        for max_edge in edge_values:
            for concentration in concentration_values:
                rule = {"min_history_support": support, "min_max_edge": max_edge, "min_recent_concentration": concentration}
                for threshold in candidate_thresholds(valid, rule, score_quantiles):
                    _, metric = evaluate_rule(valid, rule, threshold)
                    trials.append({**rule, "threshold": threshold, **metric})
    valid_trials = [row for row in trials if row["retention_rate"] >= args.min_retention and not math.isnan(row["proposal_precision"])]
    if not valid_trials:
        raise ValueError("No strict rule meets --min_retention; relax the pre-declared grid.")
    selected = max(valid_trials, key=lambda row: (row["proposal_precision"], row["proposal_recall"], row["retention_rate"]))
    rule = {key: selected[key] for key in ["min_history_support", "min_max_edge", "min_recent_concentration"]}
    test_records, test_metric = evaluate_rule(test, rule, selected["threshold"])
    intervals = bootstrap(test_records, args.seed + 17, args.bootstrap_draws)
    raw_hits = sum(row["raw_hit"] for row in test_records)
    strict_hits = sum(row["strict_hit"] for row in test_records)
    test_metric.update({"split": "test", "variant": "strict_critic_selected", **rule, "threshold": selected["threshold"], "raw_hit_retention": strict_hits / raw_hits if raw_hits else math.nan, **{f"{key}_{bound}": value for key, interval in intervals.items() for bound, value in zip(["ci_low", "ci_high"], interval)}})
    output = Path(args.out_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["min_history_support", "min_max_edge", "min_recent_concentration", "threshold", "eligible_tail_contexts", "retained_contexts", "retention_rate", "proposal_precision", "proposal_recall", "raw_transition_precision"]
    with Path(f"{output}_validation_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(trials)
    summary_fields = list(test_metric)
    with Path(f"{output}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader(); writer.writerow(test_metric)
    lines = ["# Stage-2.5 strict tail-continuation Critic", "", "Validation selects among pre-declared history-specific gates. Test is untouched until the final report.", "", "## Selected validation rule", "", "| minimum history supporters | minimum direct-edge count | minimum recent-evidence concentration | Critic threshold | validation retention | validation precision | validation recall |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |", f"| {rule['min_history_support']} | {rule['min_max_edge']} | {rule['min_recent_concentration']:.2f} | {selected['threshold']:.6f} | {selected['retention_rate']:.4f} | {selected['proposal_precision']:.4f} | {selected['proposal_recall']:.4f} |", "", "## Frozen test result", "", "| raw transition precision | strict precision | strict recall | selection rate | raw-hit retention | strict precision 95% CI | precision gain 95% CI |", "| ---: | ---: | ---: | ---: | ---: | --- | --- |", f"| {test_metric['raw_transition_precision']:.4f} | {test_metric['proposal_precision']:.4f} | {test_metric['proposal_recall']:.4f} | {test_metric['retention_rate']:.4f} | {test_metric['raw_hit_retention']:.4f} | [{test_metric['strict_precision_ci_low']:.4f}, {test_metric['strict_precision_ci_high']:.4f}] | [{test_metric['precision_gain_ci_low']:.4f}, {test_metric['precision_gain_ci_high']:.4f}] |", "", "`recent-evidence concentration` is the share of a candidate's weighted graph evidence contributed by the last two real history items. Bootstrap resamples eligible test contexts; it quantifies sampling uncertainty but does not make the result causal.", ""]
    Path(f"{output}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {"dataset": args.dataset, "checkpoint": args.checkpoint, "eligible_tail_items": len(eligible_tail), "transition_edges": int(sum(len(value) for value in evidence[0].values())), "minimum_retention": args.min_retention, "validation_rule_grid": {"history_support_values": support_values, "max_edge_values": edge_values, "recent_concentration_values": concentration_values, "score_quantiles": score_quantiles}, "selected": selected, "bootstrap_draws": args.bootstrap_draws}
    Path(f"{output}_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {output}_validation_grid.csv, _summary.csv, _summary.md and _meta.json")


if __name__ == "__main__":
    main()
