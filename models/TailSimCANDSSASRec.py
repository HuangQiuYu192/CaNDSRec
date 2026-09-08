# -*- coding: utf-8 -*-
"""CaNDS with low-weight, Critic-selected real tail-continuation replay."""

from collections import Counter, defaultdict
import math

import numpy as np
import torch
import torch.nn.functional as F

from .CANDSSASRec import CANDSSASRec


class TailSimCANDSSASRec(CANDSSASRec):
    """Replay only behaviour-supported, *observed* tail targets as auxiliary CE.

    For every training prefix, its own direct edge is removed before candidate
    scoring. A record is selected only when that leave-one-out Critic recovers
    its already observed tail target. This deliberately avoids presenting an
    unverified synthetic candidate as a positive label. Inference is exactly
    CANDSSASRec.
    """

    PSEUDO_FIELD = "tail_sim_target"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.tail_sim_weight = float(config["tail_sim_weight"])
        self.tail_sim_mode = str(config["tail_sim_mode"])
        self.tail_sim_min_contexts = int(config["tail_sim_min_contexts"])
        self.tail_sim_min_predecessors = int(config["tail_sim_min_predecessors"])
        self.tail_sim_min_history_support = int(config["tail_sim_min_history_support"])
        self.tail_sim_min_edge_count = int(config["tail_sim_min_edge_count"])
        self.tail_sim_min_recent_concentration = float(config["tail_sim_min_recent_concentration"])
        self.tail_sim_score_threshold = float(config["tail_sim_score_threshold"])
        self.tail_sim_recent_window = int(config["tail_sim_recent_window"])
        self.tail_sim_decay = float(config["tail_sim_decay"])
        self._prepare_pseudo_targets(dataset, int(config["seed"]))

    @staticmethod
    def _tail_items(popularity):
        active = np.flatnonzero(popularity > 0)
        ordered = active[np.argsort(-popularity[active], kind="stable")]
        return set(np.array_split(ordered, 3)[2].tolist())

    def _prepare_pseudo_targets(self, dataset, seed):
        inter = dataset.inter_feat
        targets = inter[self.POS_ITEM_ID].cpu().numpy().astype(np.int64)
        sequences = inter[self.ITEM_SEQ].cpu().numpy().astype(np.int64)
        lengths = inter[self.ITEM_SEQ_LEN].cpu().numpy().astype(np.int64)
        popularity = np.bincount(targets, minlength=self.n_items)
        tail_items = self._tail_items(popularity)
        edges = defaultdict(Counter)
        target_contexts = Counter()
        target_predecessors = defaultdict(set)
        predecessors = np.zeros(len(targets), dtype=np.int64)
        for index, (target, sequence, length) in enumerate(zip(targets, sequences, lengths)):
            length = min(int(length), len(sequence))
            if length <= 0:
                continue
            predecessor = int(sequence[length - 1])
            predecessors[index] = predecessor
            if predecessor <= 0 or target not in tail_items:
                continue
            edges[predecessor][int(target)] += 1
            target_contexts[int(target)] += 1
            target_predecessors[int(target)].add(predecessor)
        eligible_tail = {
            target for target in tail_items
            if target_contexts[target] >= self.tail_sim_min_contexts
            and len(target_predecessors[target]) >= self.tail_sim_min_predecessors
        }
        # Rebuild the graph after the Stage-0 whitelist is known.
        edges = defaultdict(Counter)
        target_contexts = Counter()
        target_predecessors = defaultdict(set)
        for target, predecessor in zip(targets, predecessors):
            if predecessor > 0 and int(target) in eligible_tail:
                edges[int(predecessor)][int(target)] += 1
                target_contexts[int(target)] += 1
                target_predecessors[int(target)].add(int(predecessor))
        edge_totals = {predecessor: sum(counts.values()) for predecessor, counts in edges.items()}
        strict_indices = []
        raw_indices = []
        eligible_indices = []
        for index, (target, sequence, length, heldout_predecessor) in enumerate(zip(targets, sequences, lengths, predecessors)):
            length = min(int(length), len(sequence))
            if length <= 0 or int(target) not in eligible_tail:
                continue
            eligible_indices.append(index)
            history = sequence[max(0, length - self.tail_sim_recent_window):length]
            history = history[history > 0].astype(np.int64, copy=False)
            strict, raw = self._leave_one_out_candidates(
                history, int(target), int(heldout_predecessor), edges, edge_totals,
                target_contexts, target_predecessors, eligible_tail,
            )
            if raw == int(target):
                raw_indices.append(index)
            if strict == int(target):
                strict_indices.append(index)
        strict_count = len(strict_indices)
        rng = np.random.default_rng(seed)
        if self.tail_sim_mode == "strict":
            selected_indices = np.asarray(strict_indices, dtype=np.int64)
        elif self.tail_sim_mode == "raw":
            # Match strict replay volume so this control tests quality of the
            # selection rule rather than simply applying more auxiliary loss.
            selected_indices = rng.choice(raw_indices, size=min(strict_count, len(raw_indices)), replace=False) if strict_count and raw_indices else np.empty(0, dtype=np.int64)
        elif self.tail_sim_mode == "random":
            selected_indices = rng.choice(eligible_indices, size=min(strict_count, len(eligible_indices)), replace=False) if strict_count and eligible_indices else np.empty(0, dtype=np.int64)
        elif self.tail_sim_mode == "off":
            selected_indices = np.empty(0, dtype=np.int64)
        else:
            raise ValueError("tail_sim_mode must be one of: off, random, raw, strict")
        replay = np.zeros(len(targets), dtype=np.int64)
        replay[selected_indices] = targets[selected_indices]
        inter[self.PSEUDO_FIELD] = torch.as_tensor(replay, dtype=torch.long)
        nonzero = int(np.count_nonzero(replay))
        print(
            "TailSim real-tail replay: mode={} strict_hits={} raw_hits={} replay_targets={} "
            "eligible_tail_prefixes={} eligible_tail_items={}".format(
                self.tail_sim_mode, len(strict_indices), len(raw_indices), nonzero,
                len(eligible_indices), len(eligible_tail)
            )
        )

    def _leave_one_out_candidates(self, history, heldout_target, heldout_predecessor, edges, edge_totals, target_contexts, target_predecessors, eligible_tail):
        """Return (strict_critic_top1, raw_transition_top1), or (None, None)."""
        weighted = Counter()
        raw_scores = Counter()
        support_predecessors = defaultdict(set)
        max_edge = Counter()
        recent_two = Counter()
        removes_edge = heldout_target in eligible_tail and heldout_predecessor > 0
        for position, predecessor in enumerate(history):
            predecessor = int(predecessor)
            denominator = edge_totals.get(predecessor, 0) - int(removes_edge and predecessor == heldout_predecessor)
            if denominator <= 0:
                continue
            recency = self.tail_sim_decay ** (len(history) - 1 - position)
            for candidate, count in edges.get(predecessor, {}).items():
                adjusted_count = count - int(removes_edge and predecessor == heldout_predecessor and candidate == heldout_target)
                if adjusted_count <= 0:
                    continue
                contribution = recency * adjusted_count
                weighted[candidate] += contribution
                raw_scores[candidate] += contribution / denominator
                support_predecessors[candidate].add(predecessor)
                max_edge[candidate] = max(max_edge[candidate], adjusted_count)
                if position >= max(0, len(history) - 2):
                    recent_two[candidate] += contribution
        candidates = []
        for candidate, weighted_count in weighted.items():
            contexts = target_contexts[candidate] - int(removes_edge and candidate == heldout_target)
            predecessor_count = len(target_predecessors[candidate])
            if removes_edge and candidate == heldout_target and edges[heldout_predecessor][candidate] == 1:
                predecessor_count -= 1
            if contexts <= 0 or predecessor_count <= 0:
                continue
            score = math.log1p(weighted_count) + 0.35 * math.log1p(contexts) + 0.35 * math.log1p(predecessor_count)
            candidates.append((candidate, score, raw_scores[candidate], len(support_predecessors[candidate]), max_edge[candidate], recent_two[candidate] / weighted_count))
        if not candidates:
            return None, None
        raw = max(candidates, key=lambda row: (row[2], row[1], -row[0]))[0]
        strict = [
            row for row in candidates
            if row[3] >= self.tail_sim_min_history_support
            and row[4] >= self.tail_sim_min_edge_count
            and row[5] >= self.tail_sim_min_recent_concentration
            and row[1] >= self.tail_sim_score_threshold
        ]
        if not strict:
            return None, raw
        return max(strict, key=lambda row: (row[1], row[2], -row[0]))[0], raw

    def calculate_loss(self, interaction):
        if self.loss_type != "CE" or self.tail_sim_weight <= 0 or self.tail_sim_mode == "off":
            return super().calculate_loss(interaction)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        pseudo_items = interaction[self.PSEUDO_FIELD]
        seq_output = self.forward(item_seq, item_seq_len)
        logits = self._full_scores(seq_output)
        rec_loss = self.loss_fct(logits, pos_items)
        mask = pseudo_items > 0
        if not mask.any():
            return rec_loss
        aux_loss = F.cross_entropy(logits[mask], pseudo_items[mask])
        return rec_loss + self.tail_sim_weight * aux_loss
