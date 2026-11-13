"""Temporal Transformer encoder over a customer's basket history."""

from typing import Any

import torch
from torch import nn

from basketmind.models.batch import pad_flat


class TemporalEncoder(nn.Module):
    """Basket vectors -> causal Transformer -> state after the last real basket.

    Input per basket: mean item embedding (computed by the caller) concatenated with the numeric basket
    features, projected to `embedding_dim`, plus a learned position embedding. Attention is causal and
    padding is masked, so a position never sees later baskets or padding.
    """

    def __init__(self, cfg: dict[str, Any], basket_feature_dim: int, item_dim: int):
        super().__init__()
        d = cfg["embedding_dim"]
        self.max_len = cfg["max_sequence_length"]
        self.causal = cfg["causal"]
        self.input_proj = nn.Linear(item_dim + basket_feature_dim, d)
        self.position = nn.Embedding(self.max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg["num_heads"], cfg["feedforward_dim"], cfg["dropout"], activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg["num_layers"], norm=nn.LayerNorm(d), enable_nested_tensor=False
        )
        self.drop = nn.Dropout(cfg["dropout"])

    def encode_sequence(
        self, basket_vecs: torch.Tensor, basket_feats: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position states (B, L, D) and the validity mask (B, L)."""
        x, mask = pad_flat(self.input_proj(torch.cat([basket_vecs, basket_feats], dim=-1)), counts)
        length = x.shape[1]
        if length > self.max_len:
            raise ValueError(f"history length {length} exceeds max_sequence_length {self.max_len}")
        x = self.drop(x + self.position(torch.arange(length, device=x.device)))
        # bool masks (True = blocked) to match the bool padding mask; mixing types is deprecated in PyTorch
        causal = (
            torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), 1)
            if self.causal
            else None
        )
        return self.encoder(x, mask=causal, src_key_padding_mask=~mask), mask

    def forward(
        self, basket_vecs: torch.Tensor, basket_feats: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """(B, D): the state at each example's last real basket."""
        states, _ = self.encode_sequence(basket_vecs, basket_feats, counts)
        return states[torch.arange(len(counts), device=states.device), counts - 1]
