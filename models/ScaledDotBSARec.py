"""BSARec with an explicit, fixed global logit scale.

This is a calibration control: its within-user ranking is exactly the same as
raw dot-product BSARec for a frozen representation. Retraining changes only
the global full-softmax concentration, allowing a validation-selected test of
whether dimension-dependent dot-logit scale explains the joint-score result.
"""
from __future__ import annotations

import torch

from .BSARec import BSARec


class ScaledDotBSARec(BSARec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temperature = float(config["temperature"])
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def _scores(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.temperature * (sequence @ self.item_embedding.weight.transpose(0, 1))

    def calculate_loss(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos = self.temperature * torch.sum(sequence * self.item_embedding(pos_items), dim=-1)
            neg = self.temperature * torch.sum(sequence * self.item_embedding(neg_items), dim=-1)
            return self.loss_fct(pos, neg)
        return self.loss_fct(self._scores(sequence), pos_items)

    def predict(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        return self.temperature * torch.sum(sequence * self.item_embedding(interaction[self.ITEM_ID]), dim=-1)

    def full_sort_predict(self, interaction):
        return self._scores(self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]))
