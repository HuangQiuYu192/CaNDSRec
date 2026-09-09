import torch
import torch.nn.functional as F

from .BSARec import BSARec


class CANDSBSARec(BSARec):
    """BSARec with a temperature-scaled cosine recommendation head."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temperature = float(config["temperature"])

    def _items(self):
        return F.normalize(self.item_embedding.weight, dim=-1)

    def _scores(self, seq_output):
        return self.temperature * torch.matmul(
            F.normalize(seq_output, dim=-1), self._items().transpose(0, 1)
        )

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = F.normalize(self.forward(item_seq, item_seq_len), dim=-1)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            items = self._items()
            return self.loss_fct(
                self.temperature * torch.sum(seq_output * items[pos_items], dim=-1),
                self.temperature * torch.sum(seq_output * items[neg_items], dim=-1),
            )
        return self.loss_fct(self._scores(seq_output), pos_items)

    def predict(self, interaction):
        seq_output = F.normalize(self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]), dim=-1)
        return self.temperature * torch.sum(seq_output * self._items()[interaction[self.ITEM_ID]], dim=-1)

    def full_sort_predict(self, interaction):
        return self._scores(self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]))
