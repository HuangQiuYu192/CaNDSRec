# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

from .CalibratedCANDSSASRec import CalibratedCANDSSASRec


class TailCLCalibratedCANDSSASRec(CalibratedCANDSSASRec):
    """Cosine SASRec with item-prior bias and tail-positive item contrast."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.tail_cl_weight = float(config["tail_cl_weight"])
        self.tail_cl_temperature = float(config["tail_cl_temperature"])
        self.tail_cl_quantile = float(config["tail_cl_quantile"])
        self.tail_cl_min_teacher_pop = int(config["tail_cl_min_teacher_pop"])
        self.tail_cl_max_items = int(config["tail_cl_max_items"])

        tail_mask, positive_teacher = self._build_tail_positive_pairs(dataset)
        self.register_buffer("tail_cl_mask", tail_mask)
        self.register_buffer("tail_positive_teacher", positive_teacher)

    def _item_popularity(self, dataset):
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu().numpy()
        return np.bincount(item_ids, minlength=self.n_items).astype(np.float32)

    def _leaf_categories(self, dataset):
        categories = dataset.get_item_feature()["categories"].cpu().numpy()
        leaf = np.zeros(self.n_items, dtype=np.int64)
        for item_id, row in enumerate(categories):
            nonzero = row[row > 0]
            if len(nonzero):
                leaf[item_id] = int(nonzero[-1])
        return leaf

    def _build_tail_positive_pairs(self, dataset):
        pop = self._item_popularity(dataset)
        leaf = self._leaf_categories(dataset)
        active = pop > 0
        threshold = np.quantile(pop[active], self.tail_cl_quantile) if active.any() else 0.0
        tail = active & (pop <= threshold)

        leaf_to_items = {}
        for item_id in np.where(active & (leaf > 0))[0]:
            leaf_to_items.setdefault(int(leaf[item_id]), []).append(int(item_id))

        teacher = np.zeros(self.n_items, dtype=np.int64)
        valid_tail = np.zeros(self.n_items, dtype=np.bool_)
        for item_id in np.where(tail & (leaf > 0))[0]:
            candidates = [
                other
                for other in leaf_to_items.get(int(leaf[item_id]), [])
                if other != item_id and pop[other] >= self.tail_cl_min_teacher_pop and pop[other] > pop[item_id]
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda other: pop[other])
            teacher[item_id] = best
            valid_tail[item_id] = True

        return torch.tensor(valid_tail, dtype=torch.bool), torch.tensor(teacher, dtype=torch.long)

    def _tail_item_cl_loss(self, pos_items):
        if self.tail_cl_weight <= 0:
            return torch.zeros((), device=pos_items.device)

        valid = self.tail_cl_mask[pos_items]
        if not valid.any():
            return torch.zeros((), device=pos_items.device)

        tail_items = pos_items[valid]
        if self.tail_cl_max_items > 0 and tail_items.numel() > self.tail_cl_max_items:
            tail_items = tail_items[: self.tail_cl_max_items]

        teacher_items = self.tail_positive_teacher[tail_items]
        item_dir = self._normalized_item_embedding()
        anchor = item_dir[tail_items]
        logits = torch.matmul(anchor, item_dir.transpose(0, 1)) / self.tail_cl_temperature
        logits[:, 0] = -1e9
        logits.scatter_(1, tail_items.unsqueeze(1), -1e9)
        return F.cross_entropy(logits, teacher_items)

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        return loss + self.tail_cl_weight * self._tail_item_cl_loss(pos_items)
