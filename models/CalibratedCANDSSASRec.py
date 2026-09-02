# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn

from .CANDSSASRec import CANDSSASRec


class CalibratedCANDSSASRec(CANDSSASRec):
    """Cosine SASRec with a separated, regularized item-prior bias."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.bias_reg_weight = float(config["bias_reg_weight"])
        self.bias_init_scale = float(config["bias_init_scale"])
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        if self.bias_init_scale != 0:
            self._init_bias_from_popularity(dataset)

    def _init_bias_from_popularity(self, dataset):
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu().numpy()
        pop = np.bincount(item_ids, minlength=self.n_items).astype(np.float32)
        prior = np.log1p(pop)
        active = pop > 0
        if active.any():
            prior[active] = prior[active] - prior[active].mean()
        prior[0] = 0.0
        with torch.no_grad():
            self.item_bias.copy_(self.bias_init_scale * torch.tensor(prior))

    def _full_scores(self, seq_output):
        scores = super()._full_scores(seq_output)
        return scores + self.item_bias.unsqueeze(0)

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        if self.bias_reg_weight > 0:
            loss = loss + self.bias_reg_weight * torch.mean(self.item_bias[1:] ** 2)
        return loss

    def predict(self, interaction):
        scores = super().predict(interaction)
        test_item = interaction[self.ITEM_ID]
        return scores + self.item_bias[test_item]
