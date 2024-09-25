import math
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn


class RelativePosEmb(nn.Module):
    def __init__(
        self,
        relative_bias_slope: float = 8.0,
        relative_bias_inter: int = 128,
        num_heads: int = 8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.value = relative_bias_slope
        self.mlp = nn.Sequential(
            nn.Linear(1, relative_bias_inter, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(relative_bias_inter, num_heads, bias=False),
        )
        self.seq_len = 0
        self.rel_coords_table = None

    def forward(
        self,
        seq_len: int,
    ) -> torch.Tensor:
        if seq_len != self.seq_len:
            self.rel_coords_table = torch.zeros(seq_len + 1, seq_len + 1, 1)
            self.rel_coords_table[1:, 1:] = self._compute_rel_coords(seq_len)
            self.rel_coords_table = self.rel_coords_table.to(self.mlp[0].weight)
            self.seq_len = seq_len
        rel_bias = rearrange(self.mlp(self.rel_coords_table), "i j n -> 1 n i j")

        return rel_bias

    @torch.no_grad()
    def _compute_rel_coords(self, seq_len: int) -> torch.Tensor:
        coords = torch.linspace(-self.value / 2, self.value / 2, seq_len)
        rel_coords = coords[None, :] - coords[:, None]
        rel_coords_table = (
            torch.sign(rel_coords)
            * torch.log2(torch.abs(rel_coords) + 1.0)
            / math.log2(self.value + 1)
        )
        return rel_coords_table.unsqueeze(-1)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hid_dim: int = 512,
        bias: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w1 = nn.Linear(dim, int(2 / 3 * hid_dim), bias=bias)
        self.w2 = nn.Linear(dim, int(2 / 3 * hid_dim), bias=bias)
        self.w3 = nn.Linear(int(2 / 3 * hid_dim), dim, bias=bias)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swiglu = self.silu(self.w1(x)) * self.w2(x)
        return self.w3(swiglu)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.2,
        out_dropout: float = 0.2,
        attn_bias: bool = False,
        relative_bias: bool = True,
        relative_bias_inter: int = 64,
        relative_bias_slope: float = 8.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.relative_bias = relative_bias
        self.qkv = nn.Linear(dim, 3 * dim, bias=attn_bias)
        self.attn_dropout = attn_dropout
        self.out = nn.Linear(dim, dim, bias=attn_bias)
        self.out_dropout = nn.Dropout(out_dropout)

        if self.relative_bias:
            self.pos_emb = RelativePosEmb(
                relative_bias_slope=relative_bias_slope,
                relative_bias_inter=relative_bias_inter,
                num_heads=num_heads,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.size()
        q, k, v = rearrange(
            self.qkv(x), "b l (qkv h d) -> qkv b h l d", h=self.num_heads, qkv=3
        )
        attn_bias = torch.zeros(B, self.num_heads, L, L, device=x.device)
        if self.relative_bias:
            attn_bias += repeat(self.pos_emb(L - 1), "1 h i j -> k h i j", k=B)
        if attn_mask is not None:
            attn_mask = torch.cat(
                (torch.ones(B, 1, dtype=torch.bool, device=x.device), attn_mask), dim=1
            )
            attn_mask = repeat(attn_mask, "b l -> b h s l", h=self.num_heads, s=L)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_bias, self.attn_dropout if self.training else 0
            )
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.out_dropout(self.out(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hid_dim: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.2,
        out_dropout: float = 0.2,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        relative_bias: bool = True,
        relative_bias_inter: int = 64,
        relative_bias_slope: float = 8.0,
        norm_layer: nn.Module = nn.LayerNorm,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout,
            attn_bias=attn_bias,
            relative_bias=relative_bias,
            relative_bias_inter=relative_bias_inter,
            relative_bias_slope=relative_bias_slope,
        )
        self.mlp = MLP(dim=dim, hid_dim=hid_dim, bias=mlp_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))

        return x


class RankingModel(nn.Module):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        num_layers: int = 4,
        dim: int = 128,
        hid_dim: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.3,
        out_dropout: float = 0.3,
        emb_dropout: float = 0.3,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        relative_bias: bool = True,
        relative_bias_inter: int = 64,
        relative_bias_slope: float = 8.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(user_size, dim)
        self.item_emb = nn.Embedding(item_size, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    dim=dim,
                    hid_dim=hid_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    out_dropout=out_dropout,
                    attn_bias=attn_bias,
                    mlp_bias=mlp_bias,
                    relative_bias=relative_bias,
                    relative_bias_inter=relative_bias_inter,
                    relative_bias_slope=relative_bias_slope,
                    norm_layer=norm_layer,
                )
            )
        self.out = nn.Linear(dim, 1, bias=mlp_bias)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_emb = self.dropout(self.user_emb(users))
        item_emb = self.dropout(self.item_emb(items))
        x = torch.cat((user_emb.unsqueeze(1), item_emb), dim=-2)
        for block in self.transformer_blocks:
            x = block(x, attn_mask)
        out = self.out(x)
        return out.squeeze(-1)[:, 1:]
