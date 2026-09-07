# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .CANDSSASRec import CANDSSASRec


class SemanticCANDSSASRec(CANDSSASRec):
    """CaNDS with lightweight item semantic anchors.

    The base CaNDS score remains cosine-normalized ID matching. This model adds
    an optional semantic branch from precomputed item text embeddings, either as
    score fusion or as an item-embedding anchor.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.semantic_embedding_path = str(config["semantic_embedding_path"])
        self.semantic_fusion_mode = str(config["semantic_fusion_mode"])
        self.semantic_weight = float(config["semantic_weight"])
        self.semantic_gate = str(config["semantic_gate"])
        self.semantic_freeze = bool(config["semantic_freeze"])

        semantic = np.load(self.semantic_embedding_path).astype(np.float32)
        if semantic.shape[0] < self.n_items:
            pad = np.zeros((self.n_items - semantic.shape[0], semantic.shape[1]), dtype=np.float32)
            semantic = np.concatenate([semantic, pad], axis=0)
        elif semantic.shape[0] > self.n_items:
            semantic = semantic[: self.n_items]

        self.semantic_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(semantic),
            freeze=self.semantic_freeze,
            padding_idx=0,
        )
        self.semantic_projector = nn.Linear(semantic.shape[1], self.hidden_size, bias=False)

        pop = np.bincount(
            dataset.inter_feat[self.ITEM_ID].cpu().numpy(),
            minlength=self.n_items,
        ).astype(np.float32)
        self.register_buffer("semantic_gate_values", self._build_gate(pop))

    def _build_gate(self, pop):
        if self.semantic_gate == "constant":
            gate = np.full(self.n_items, self.semantic_weight, dtype=np.float32)
        elif self.semantic_gate == "tail":
            log_pop = np.log1p(pop)
            active = log_pop[1:] if len(log_pop) > 1 else log_pop
            lo = float(active.min()) if active.size else 0.0
            hi = float(active.max()) if active.size else 1.0
            tail_strength = (hi - log_pop) / max(hi - lo, 1e-8)
            gate = self.semantic_weight * tail_strength.astype(np.float32)
        elif self.semantic_gate == "sqrt_tail":
            log_pop = np.log1p(pop)
            active = log_pop[1:] if len(log_pop) > 1 else log_pop
            lo = float(active.min()) if active.size else 0.0
            hi = float(active.max()) if active.size else 1.0
            tail_strength = (hi - log_pop) / max(hi - lo, 1e-8)
            gate = self.semantic_weight * np.sqrt(np.clip(tail_strength, 0.0, 1.0)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported semantic_gate: {self.semantic_gate}")
        gate[0] = 0.0
        return torch.from_numpy(np.clip(gate, 0.0, 1.0))

    def _semantic_item_embedding(self):
        semantic = self.semantic_projector(self.semantic_embedding.weight)
        return F.normalize(semantic, dim=-1)

    def _anchored_item_embedding(self):
        item_emb = self._normalized_item_embedding()
        semantic = self._semantic_item_embedding()
        gate = self.semantic_gate_values.to(item_emb.device).unsqueeze(-1)
        return F.normalize((1.0 - gate) * item_emb + gate * semantic, dim=-1)

    def _full_scores(self, seq_output):
        seq_output = self._normalized_sequence_output(seq_output)
        item_emb = self._normalized_item_embedding()
        id_scores = torch.matmul(seq_output, item_emb.transpose(0, 1))

        if self.semantic_fusion_mode == "score":
            semantic_emb = self._semantic_item_embedding()
            sem_scores = torch.matmul(seq_output, semantic_emb.transpose(0, 1))
            gate = self.semantic_gate_values.to(seq_output.device).unsqueeze(0)
            scores = (1.0 - gate) * id_scores + gate * sem_scores
        elif self.semantic_fusion_mode == "anchor":
            anchored_item_emb = self._anchored_item_embedding()
            scores = torch.matmul(seq_output, anchored_item_emb.transpose(0, 1))
        else:
            raise ValueError(f"Unsupported semantic_fusion_mode: {self.semantic_fusion_mode}")
        return self.temperature * scores

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self._normalized_sequence_output(seq_output)

        if self.semantic_fusion_mode == "score":
            item_emb = self._normalized_item_embedding()[test_item]
            semantic_emb = self._semantic_item_embedding()[test_item]
            id_scores = torch.sum(seq_output * item_emb, dim=-1)
            sem_scores = torch.sum(seq_output * semantic_emb, dim=-1)
            gate = self.semantic_gate_values.to(seq_output.device)[test_item]
            scores = (1.0 - gate) * id_scores + gate * sem_scores
        elif self.semantic_fusion_mode == "anchor":
            item_emb = self._anchored_item_embedding()[test_item]
            scores = torch.sum(seq_output * item_emb, dim=-1)
        else:
            raise ValueError(f"Unsupported semantic_fusion_mode: {self.semantic_fusion_mode}")
        return self.temperature * scores
