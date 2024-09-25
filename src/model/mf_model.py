from typing import Optional

import torch
from torch import nn


class MatrixFactorizaiton(nn.Module):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        emb_dim: int,
        dropout: float = 0.2,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(user_size, emb_dim)
        self.user_bias = nn.Embedding(user_size, 1)
        self.item_emb = nn.Embedding(item_size, emb_dim)
        self.item_bias = nn.Embedding(item_size, 1)
        self.dropout = nn.Dropout(dropout)
        if weights_path:
            self.load_state_dict(torch.load(weights_path, weights_only=True))
            self.eval()

    def batch_predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        B, L = items.size()
        user_emb = self.dropout(self.user_emb(users)).view(B, 1, -1)
        item_emb = self.dropout(self.item_emb(items))
        user_bias = self.dropout(self.user_bias(users))
        item_bias = self.dropout(self.item_bias(items))
        out = (
            torch.einsum("bid,bjd->bij", user_emb, item_emb).view(B, L)
            + user_bias
            + item_bias.view(B, L)
        )
        return out

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        user_emb = self.dropout(self.user_emb(users))
        item_emb = self.dropout(self.item_emb.weight)
        user_bias = self.dropout(self.user_bias(users))
        item_bias = self.dropout(self.item_bias.weight)
        out = (
            torch.einsum("id,jd->ij", user_emb, item_emb)
            + user_bias
            + item_bias.transpose(-2, 1)
        )
        return out
