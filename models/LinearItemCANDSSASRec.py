# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

from .CANDSSASRec import CANDSSASRec


class LinearItemCANDSSASRec(CANDSSASRec):
    """Angular SASRec with a sparse linear item-item transition score."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.linear_lambda = float(config["linear_lambda"])
        self.linear_topk = int(config["linear_topk"])
        self.linear_seq_decay = float(config["linear_seq_decay"])
        self.linear_edge_decay = float(config["linear_edge_decay"])
        self.linear_normalize = bool(config["linear_normalize"])
        self.linear_use_bias = bool(config["linear_use_bias"])
        self.bias_init_scale = float(config["bias_init_scale"])
        self.bias_reg_weight = float(config["bias_reg_weight"])

        indices, values = self._build_sparse_transition(dataset)
        transition = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items)).coalesce()
        self.register_buffer("transition_indices", transition.indices())
        self.register_buffer("transition_values", transition.values())

        pop_prior = self._build_popularity_prior(dataset)
        self.register_buffer("pop_prior", pop_prior)
        self.item_bias = torch.nn.Parameter(torch.zeros(self.n_items))
        if self.linear_use_bias and self.bias_init_scale != 0:
            with torch.no_grad():
                self.item_bias.copy_(self.bias_init_scale * self.pop_prior)
                self.item_bias[0] = 0.0

    def _build_popularity_prior(self, dataset):
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu().numpy()
        pop = np.bincount(item_ids, minlength=self.n_items).astype(np.float32)
        prior = np.log1p(pop)
        active = pop > 0
        if active.any():
            prior[active] -= prior[active].mean()
            std = prior[active].std()
            if std > 1e-8:
                prior[active] /= std
        prior[0] = 0.0
        return torch.tensor(prior, dtype=torch.float32)

    def _build_sparse_transition(self, dataset):
        item_seq = dataset.inter_feat[self.ITEM_SEQ].cpu().numpy()
        seq_len = dataset.inter_feat[self.ITEM_SEQ_LEN].cpu().numpy()
        target = dataset.inter_feat[self.ITEM_ID].cpu().numpy()
        out_edges = {}

        for seq, length, tgt in zip(item_seq, seq_len, target):
            tgt = int(tgt)
            if tgt <= 0:
                continue
            valid_seq = seq[: int(length)]
            for offset, src in enumerate(valid_seq[::-1]):
                src = int(src)
                if src <= 0:
                    continue
                weight = self.linear_edge_decay ** offset
                bucket = out_edges.setdefault(src, {})
                bucket[tgt] = bucket.get(tgt, 0.0) + weight

        rows, cols, vals = [], [], []
        for src, tgt_counts in out_edges.items():
            pairs = sorted(tgt_counts.items(), key=lambda x: x[1], reverse=True)[: self.linear_topk]
            total = sum(v for _, v in pairs)
            if total <= 0:
                continue
            for tgt, value in pairs:
                rows.append(src)
                cols.append(tgt)
                vals.append(value / total if self.linear_normalize else value)

        if not rows:
            indices = torch.zeros((2, 1), dtype=torch.long)
            values = torch.zeros(1, dtype=torch.float32)
            return indices, values
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        return indices, values

    def _transition_matrix(self):
        return torch.sparse_coo_tensor(
            self.transition_indices,
            self.transition_values,
            (self.n_items, self.n_items),
            device=self.transition_values.device,
        ).coalesce()

    def _linear_scores(self, item_seq):
        batch_size, seq_len = item_seq.size()
        pos = torch.arange(seq_len, device=item_seq.device).view(1, -1)
        valid_len = (item_seq > 0).sum(dim=1, keepdim=True)
        distance = valid_len - 1 - pos
        weights = torch.pow(
            torch.tensor(self.linear_seq_decay, device=item_seq.device, dtype=torch.float32),
            torch.clamp(distance, min=0).float(),
        )
        weights = weights * (item_seq > 0).float()

        history = torch.zeros(batch_size, self.n_items, device=item_seq.device)
        history.scatter_add_(1, item_seq, weights)
        history[:, 0] = 0.0
        if self.linear_normalize:
            history = history / history.sum(dim=1, keepdim=True).clamp_min(1e-8)

        transition = self._transition_matrix()
        scores = torch.sparse.mm(transition.transpose(0, 1), history.transpose(0, 1)).transpose(0, 1)
        return scores

    def _full_scores_with_sequence(self, item_seq, seq_output):
        scores = super()._full_scores(seq_output)
        if self.linear_lambda != 0:
            scores = scores + self.linear_lambda * self._linear_scores(item_seq)
        if self.linear_use_bias:
            scores = scores + self.item_bias.unsqueeze(0)
        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            scores = self._full_scores_with_sequence(item_seq, seq_output)
            pos_score = scores.gather(1, pos_items.unsqueeze(1)).squeeze(1)
            neg_score = scores.gather(1, neg_items.unsqueeze(1)).squeeze(1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = self._full_scores_with_sequence(item_seq, seq_output)
            loss = self.loss_fct(logits, pos_items)

        if self.linear_use_bias and self.bias_reg_weight > 0:
            loss = loss + self.bias_reg_weight * torch.mean(self.item_bias[1:] ** 2)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        scores = self._full_scores_with_sequence(item_seq, seq_output)
        return scores.gather(1, test_item.unsqueeze(1)).squeeze(1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return self._full_scores_with_sequence(item_seq, seq_output)
