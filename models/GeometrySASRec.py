# -*- coding: utf-8 -*-
"""Score-geometry ablations for SASRec.

This class deliberately changes only the final sequence--item scoring rule.
It supports one-sided, joint, and *partial* normalizations.  The latter is a
controlled radial-power family: ``x / ||x||**gamma`` retains a continuously
adjustable fraction of the original radial degree of freedom.  This is an
analysis model, not a new recommendation method.
"""

import torch

from .SASRec import SASRec


class GeometrySASRec(SASRec):
    """SASRec with a selectable final-score normalization geometry.

    ``sequence`` preserves item radii but fixes the sequence radius;
    ``item`` removes candidate radii but retains sequence-scale variation;
    ``both`` is algebraically the CaNDS score. ``partial`` uses independently
    configured powers for the sequence and item sides. The dot-product
    baseline remains :class:`SASRec` rather than this class.
    """

    VALID_GEOMETRIES = {"sequence", "item", "both", "partial"}

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.score_geometry = str(config["score_geometry"])
        if self.score_geometry not in self.VALID_GEOMETRIES:
            raise ValueError(
                f"score_geometry must be one of {sorted(self.VALID_GEOMETRIES)}, "
                f"got {self.score_geometry!r}"
            )
        self.temperature = float(config["temperature"])
        if self.score_geometry == "sequence":
            self.sequence_norm_power, self.item_norm_power = 1.0, 0.0
        elif self.score_geometry == "item":
            self.sequence_norm_power, self.item_norm_power = 0.0, 1.0
        elif self.score_geometry == "both":
            self.sequence_norm_power, self.item_norm_power = 1.0, 1.0
        else:
            self.sequence_norm_power = float(config["sequence_norm_power"])
            self.item_norm_power = float(config["item_norm_power"])

        for name, power in (
            ("sequence_norm_power", self.sequence_norm_power),
            ("item_norm_power", self.item_norm_power),
        ):
            if not 0.0 <= power <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {power}")

    def _score_inputs(self, seq_output):
        item_emb = self.item_embedding.weight
        if self.sequence_norm_power:
            seq_output = seq_output / (
                torch.linalg.vector_norm(seq_output, dim=-1, keepdim=True).clamp_min(1e-12)
                ** self.sequence_norm_power
            )
        if self.item_norm_power:
            item_emb = item_emb / (
                torch.linalg.vector_norm(item_emb, dim=-1, keepdim=True).clamp_min(1e-12)
                ** self.item_norm_power
            )
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
