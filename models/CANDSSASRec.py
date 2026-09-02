# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from .SASRec import SASRec


class CANDSSASRec(SASRec):
    """SASRec with cosine-normalized scoring and a learnable-free temperature."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temperature = float(config["temperature"])

    def _normalized_item_embedding(self):
        return F.normalize(self.item_embedding.weight, dim=-1)

    def _normalized_sequence_output(self, seq_output):
        return F.normalize(seq_output, dim=-1)

    def _full_scores(self, seq_output):
        seq_output = self._normalized_sequence_output(seq_output)
        item_emb = self._normalized_item_embedding()
        return self.temperature * torch.matmul(seq_output, item_emb.transpose(0, 1))

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            seq_output = self._normalized_sequence_output(seq_output)
            item_emb = self._normalized_item_embedding()
            pos_score = self.temperature * torch.sum(seq_output * item_emb[pos_items], dim=-1)
            neg_score = self.temperature * torch.sum(seq_output * item_emb[neg_items], dim=-1)
            return self.loss_fct(pos_score, neg_score)

        logits = self._full_scores(seq_output)
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self._normalized_sequence_output(seq_output)
        test_item_emb = self._normalized_item_embedding()[test_item]
        return self.temperature * torch.sum(seq_output * test_item_emb, dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return self._full_scores(seq_output)
