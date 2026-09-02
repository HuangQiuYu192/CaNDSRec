# -*- coding: utf-8 -*-
import math

import torch
from torch import nn

from .LearnableTempCANDSSASRec import LearnableTempCANDSSASRec


class DataAwareTempCANDSSASRec(LearnableTempCANDSSASRec):
    """Learnable-temperature cosine SASRec with data-aware initialization."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.data_temp_min = float(config["data_temp_min"])
        self.data_temp_max = float(config["data_temp_max"])
        init_temperature = self._estimate_temperature(config, dataset)
        init_log_temperature = math.log(init_temperature)
        with torch.no_grad():
            self.log_temperature.fill_(init_log_temperature)
        self._init_log_temperature = init_log_temperature

    def _estimate_temperature(self, config, dataset):
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu()
        counts = torch.bincount(item_ids, minlength=self.n_items).float()
        counts = counts[1:]
        probs = counts[counts > 0]
        probs = probs / probs.sum().clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        n_eff = max(math.exp(entropy), 2.0)

        scale = float(config["data_temp_scale"])
        min_temp = float(config["data_temp_min"])
        max_temp = float(config["data_temp_max"])
        init_temp = scale * math.sqrt(2.0 * math.log(n_eff))
        return min(max(init_temp, min_temp), max_temp)

    def _temperature(self):
        min_log_temp = math.log(self.data_temp_min)
        max_log_temp = math.log(self.data_temp_max)
        return torch.exp(torch.clamp(self.log_temperature, min_log_temp, max_log_temp))
