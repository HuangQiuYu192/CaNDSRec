# -*- coding: utf-8 -*-
import math

import torch
import torch.nn.functional as F
from torch import nn

from .CANDSSASRec import CANDSSASRec


class LearnableTempCANDSSASRec(CANDSSASRec):
    """Cosine SASRec with a learnable global temperature."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        init_temperature = max(float(config["temperature"]), 1e-4)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.temp_reg_weight = float(config["temp_reg_weight"])
        self._init_log_temperature = math.log(init_temperature)

    def _temperature(self):
        min_log_temp = math.log(1.0)
        max_log_temp = math.log(30.0)
        return torch.exp(torch.clamp(self.log_temperature, min_log_temp, max_log_temp))

    def _full_scores(self, seq_output):
        seq_output = self._normalized_sequence_output(seq_output)
        item_emb = self._normalized_item_embedding()
        return self._temperature() * torch.matmul(seq_output, item_emb.transpose(0, 1))

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            seq_output = self._normalized_sequence_output(seq_output)
            item_emb = self._normalized_item_embedding()
            pos_score = self._temperature() * torch.sum(seq_output * item_emb[pos_items], dim=-1)
            neg_score = self._temperature() * torch.sum(seq_output * item_emb[neg_items], dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = self._full_scores(seq_output)
            loss = self.loss_fct(logits, pos_items)

        if self.temp_reg_weight > 0:
            target = torch.tensor(self._init_log_temperature, device=self.log_temperature.device)
            loss = loss + self.temp_reg_weight * (self.log_temperature - target).pow(2)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self._normalized_sequence_output(seq_output)
        test_item_emb = self._normalized_item_embedding()[test_item]
        return self._temperature() * torch.sum(seq_output * test_item_emb, dim=-1)
