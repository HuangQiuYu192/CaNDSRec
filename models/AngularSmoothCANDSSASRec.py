# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

from .CANDSSASRec import CANDSSASRec


class AngularSmoothCANDSSASRec(CANDSSASRec):
    """CaNDS with tail-aware angular-neighborhood positive smoothing.

    Prediction is identical to CANDSSASRec. During training, sparse targets can
    borrow a small amount of supervision from high-confidence angular neighbors
    in the normalized item embedding space.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.angular_smooth_weight = float(config["angular_smooth_weight"])
        self.angular_smooth_k = int(config["angular_smooth_k"])
        self.angular_smooth_temperature = float(config["angular_smooth_temperature"])
        self.angular_smooth_pop_quantile = float(config["angular_smooth_pop_quantile"])
        self.angular_smooth_sim_threshold = float(config["angular_smooth_sim_threshold"])

        item_pop, smooth_mask = self._build_smoothing_buffers(dataset)
        self.register_buffer("angular_item_popularity", item_pop)
        self.register_buffer("angular_smooth_mask", smooth_mask)

    def _build_smoothing_buffers(self, dataset):
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu().numpy()
        pop = np.bincount(item_ids, minlength=self.n_items).astype(np.float32)
        active = pop > 0
        threshold = np.quantile(pop[active], self.angular_smooth_pop_quantile) if active.any() else 0.0
        smooth_mask = active & (pop <= threshold)

        return (
            torch.tensor(pop, dtype=torch.float32),
            torch.tensor(smooth_mask, dtype=torch.bool),
        )

    def _angular_neighbor_loss(self, logits, pos_items):
        if self.angular_smooth_weight <= 0 or self.angular_smooth_k <= 0:
            return torch.zeros((), device=logits.device)
        if self.loss_type != "CE":
            return torch.zeros((), device=logits.device)

        valid = self.angular_smooth_mask[pos_items]
        if not valid.any():
            return torch.zeros((), device=logits.device)

        selected_logits = logits[valid]
        selected_items = pos_items[valid]

        item_dir = self._normalized_item_embedding()
        with torch.no_grad():
            detached_item_dir = item_dir.detach()
            anchor = detached_item_dir[selected_items]
            sim = torch.matmul(anchor, detached_item_dir.transpose(0, 1))
            sim[:, 0] = -float("inf")
            sim.scatter_(1, selected_items.unsqueeze(1), -float("inf"))
            topk = min(self.angular_smooth_k, self.n_items - 2)
            values, indices = torch.topk(sim, k=topk, dim=1)
            keep = values >= self.angular_smooth_sim_threshold
            weights = torch.where(
                keep,
                torch.softmax(values / self.angular_smooth_temperature, dim=1),
                torch.zeros_like(values),
            )
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        log_prob = F.log_softmax(selected_logits, dim=1)
        neighbor_log_prob = log_prob.gather(1, indices)
        per_sample_loss = -(weights * neighbor_log_prob).sum(dim=1)
        nonempty = weights.sum(dim=1) > 0
        if not nonempty.any():
            return torch.zeros((), device=logits.device)

        return per_sample_loss[nonempty].mean()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            return super().calculate_loss(interaction)

        logits = self._full_scores(seq_output)
        rec_loss = self.loss_fct(logits, pos_items)
        smooth_loss = self._angular_neighbor_loss(logits, pos_items)
        return rec_loss + self.angular_smooth_weight * smooth_loss
