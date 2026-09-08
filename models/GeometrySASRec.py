# -*- coding: utf-8 -*-
"""Score-geometry ablations for SASRec.

This class deliberately changes only the final sequence--item scoring rule.
It supports the two one-sided normalizations needed to test whether a CaNDS
gain arises from removing candidate-side radial factors, sequence-side logit
scale, or their combination.  It is an analysis model, not a new method.
"""

import torch
import torch.nn.functional as F

from .SASRec import SASRec


class GeometrySASRec(SASRec):
    """SASRec with a selectable final-score normalization geometry.

    ``sequence`` preserves item radii but fixes the sequence radius;
    ``item`` removes candidate radii but retains sequence-scale variation;
    ``both`` is algebraically the CaNDS score.  The dot-product baseline uses
    :class:`SASRec` rather than this class.
    """

    VALID_GEOMETRIES = {"sequence", "item", "both"}

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.score_geometry = str(config["score_geometry"])
        if self.score_geometry not in self.VALID_GEOMETRIES:
            raise ValueError(
                f"score_geometry must be one of {sorted(self.VALID_GEOMETRIES)}, "
                f"got {self.score_geometry!r}"
            )
        self.temperature = float(config["temperature"])

    def _score_inputs(self, seq_output):
        item_emb = self.item_embedding.weight
        if self.score_geometry in {"sequence", "both"}:
            seq_output = F.normalize(seq_output, dim=-1)
        if self.score_geometry in {"item", "both"}:
            item_emb = F.normalize(item_emb, dim=-1)
        return seq_output, item_emb

    def _full_scores(self, seq_output):
        seq_output, item_emb = self._score_inputs(seq_output)
        return self.temperature * torch.matmul(seq_output, item_emb.transpose(0, 1))

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            seq_output, item_emb = self._score_inputs(seq_output)
            pos_score = self.temperature * torch.sum(seq_output * item_emb[pos_items], dim=-1)
            neg_score = self.temperature * torch.sum(seq_output * item_emb[neg_items], dim=-1)
            return self.loss_fct(pos_score, neg_score)
        return self.loss_fct(self._full_scores(seq_output), pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output, item_emb = self._score_inputs(seq_output)
        return self.temperature * torch.sum(seq_output * item_emb[test_item], dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        return self._full_scores(self.forward(item_seq, item_seq_len))
