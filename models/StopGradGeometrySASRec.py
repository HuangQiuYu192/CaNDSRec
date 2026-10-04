# -*- coding: utf-8 -*-
"""A causal backward-pass control for final-score normalization.

The forward score is exactly the selected :class:`GeometrySASRec` score.  The
only difference is that each radial denominator is detached from autograd.
Consequently, for ``score_geometry=both``, both models rank every candidate
identically at a fixed parameter state, while this control retains the direct
radial gradient that true normalization removes.  It is an analysis control,
not a proposed recommender.
"""

import torch

from .GeometrySASRec import GeometrySASRec


class StopGradGeometrySASRec(GeometrySASRec):
    """GeometrySASRec with an identical forward score and detached norm paths."""

    def _score_inputs(self, seq_output):
        item_emb = self.item_embedding.weight
        if self.sequence_norm_power:
            sequence_norm = torch.linalg.vector_norm(
                seq_output, dim=-1, keepdim=True
            ).clamp_min(1e-12).detach()
            seq_output = seq_output / (sequence_norm ** self.sequence_norm_power)
        if self.item_norm_power:
            item_norm = torch.linalg.vector_norm(
                item_emb, dim=-1, keepdim=True
            ).clamp_min(1e-12).detach()
            item_emb = item_emb / (item_norm ** self.item_norm_power)
        return seq_output, item_emb
