"""Score-geometry controls for BSARec.

These variants retain BSARec's encoder and change only its final scoring head.
They are mechanism controls rather than proposed recommendation models.
"""
from __future__ import annotations

import torch

from .BSARec import BSARec


class GeometryBSARec(BSARec):
    """BSARec with sequence-, item-, joint-, or partial normalization."""

    VALID_GEOMETRIES = {"sequence", "item", "both", "partial"}

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temperature = float(config["temperature"])
        self.score_geometry = str(config["score_geometry"])
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.score_geometry not in self.VALID_GEOMETRIES:
            raise ValueError(f"Unknown score_geometry={self.score_geometry!r}")
        if self.score_geometry == "sequence":
            self.sequence_norm_power, self.item_norm_power = 1.0, 0.0
        elif self.score_geometry == "item":
            self.sequence_norm_power, self.item_norm_power = 0.0, 1.0
        elif self.score_geometry == "both":
            self.sequence_norm_power, self.item_norm_power = 1.0, 1.0
        else:
            self.sequence_norm_power = float(config["sequence_norm_power"])
            self.item_norm_power = float(config["item_norm_power"])
        for name, power in (("sequence_norm_power", self.sequence_norm_power), ("item_norm_power", self.item_norm_power)):
            if not 0.0 <= power <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {power}")

    def _norm(self, value: torch.Tensor, power: float, *, detach: bool = False) -> torch.Tensor:
        if not power:
            return value
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)
        if detach:
            norm = norm.detach()
        return value / norm.pow(power)

    def _score_inputs(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._norm(sequence, self.sequence_norm_power),
            self._norm(self.item_embedding.weight, self.item_norm_power),
        )

    def _scores(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence, items = self._score_inputs(sequence)
        return self.temperature * (sequence @ items.transpose(0, 1))

    def calculate_loss(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            sequence, items = self._score_inputs(sequence)
            pos = self.temperature * torch.sum(sequence * items[pos_items], dim=-1)
            neg = self.temperature * torch.sum(sequence * items[neg_items], dim=-1)
            return self.loss_fct(pos, neg)
        return self.loss_fct(self._scores(sequence), pos_items)

    def predict(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        sequence, items = self._score_inputs(sequence)
        return self.temperature * torch.sum(sequence * items[interaction[self.ITEM_ID]], dim=-1)

    def full_sort_predict(self, interaction):
        return self._scores(self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]))


class StopGradSequenceGeometryBSARec(GeometryBSARec):
    """Same forward score as GeometryBSARec, with a detached sequence norm.

    For ``score_geometry=sequence``, this differs only in the backward path of
    the sequence radial denominator, making it a causal gradient control.
    """

    def _score_inputs(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._norm(sequence, self.sequence_norm_power, detach=True),
            self._norm(self.item_embedding.weight, self.item_norm_power),
        )
