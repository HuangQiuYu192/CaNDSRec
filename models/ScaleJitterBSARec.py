"""Causal sequence-scale perturbation controls for BSARec.

During training only, each sequence representation can be multiplied by a
positive log-normal factor.  This leaves the sequence-normalized score exactly
invariant, while it changes a raw dot-product score's per-example logit scale.
The models are analysis controls, not proposed recommenders.
"""
from __future__ import annotations

import torch

from .BSARec import BSARec
from .GeometryBSARec import GeometryBSARec


class _PositiveScaleJitter:
    def _init_scale_jitter(self, config) -> None:
        self.sequence_scale_jitter_sigma = float(config["sequence_scale_jitter_sigma"])
        if self.sequence_scale_jitter_sigma < 0:
            raise ValueError("sequence_scale_jitter_sigma must be non-negative")
        # Use a private, deterministic generator per optimisation step so the
        # intervention does not shift the global RNG stream used by dropout.
        self._jitter_seed = int(config["seed"]) + 9176
        self.register_buffer("_jitter_step", torch.zeros((), dtype=torch.long), persistent=False)

    def _jitter(self, sequence: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sequence_scale_jitter_sigma == 0:
            return sequence
        generator = torch.Generator(device=sequence.device)
        generator.manual_seed(self._jitter_seed + int(self._jitter_step.item()))
        self._jitter_step.add_(1)
        log_scale = torch.randn(
            sequence.size(0), 1, device=sequence.device, dtype=sequence.dtype, generator=generator
        ) * self.sequence_scale_jitter_sigma
        return sequence * log_scale.exp()


class ScaleJitterBSARec(_PositiveScaleJitter, BSARec):
    """Raw dot-product BSARec with a training-only per-sequence scale jitter."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_scale_jitter(config)

    def calculate_loss(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        sequence = self._jitter(sequence)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos = torch.sum(sequence * self.item_embedding(pos_items), dim=-1)
            neg = torch.sum(sequence * self.item_embedding(neg_items), dim=-1)
            return self.loss_fct(pos, neg)
        return self.loss_fct(sequence @ self.item_embedding.weight.transpose(0, 1), pos_items)


class ScaleJitterSequenceGeometryBSARec(_PositiveScaleJitter, GeometryBSARec):
    """Sequence-normalized BSARec with the same training-only perturbation.

    When ``score_geometry=sequence``, positive scale jitter cancels algebraically
    in the forward score.  Any material change is therefore a numerical or
    implementation issue rather than a score-geometry effect.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_scale_jitter(config)

    def calculate_loss(self, interaction):
        sequence = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        sequence = self._jitter(sequence)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            sequence, items = self._score_inputs(sequence)
            pos = self.temperature * torch.sum(sequence * items[pos_items], dim=-1)
            neg = self.temperature * torch.sum(sequence * items[neg_items], dim=-1)
            return self.loss_fct(pos, neg)
        return self.loss_fct(self._scores(sequence), pos_items)
