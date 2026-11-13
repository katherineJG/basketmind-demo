"""Customer context encoder and fusion network."""

from typing import Any

import torch
from torch import nn


class CustomerContextEncoder(nn.Module):
    """Id embedding + country embedding + normalised numeric features -> MLP.

    Id 0 is the 'unseen customer' embedding, so customers without training history still work. The
    numeric features (frequency, basket size, quantity, spend, recency, interval) carry customer
    information on their own: the identity embedding is never the only signal.
    """

    def __init__(self, cfg: dict[str, Any], n_customers: int, n_countries: int):
        super().__init__()
        self.customer = nn.Embedding(n_customers + 1, cfg["id_dim"])
        self.country = nn.Embedding(n_countries + 1, cfg["country_dim"])
        h1, h2 = cfg["hidden_dims"]
        self.mlp = nn.Sequential(
            nn.Linear(cfg["id_dim"] + cfg["country_dim"] + cfg["numeric_dim"], h1),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(h1, h2),
            nn.GELU(),
        )
        self.out_dim = h2

    def forward(
        self, customer_idx: torch.Tensor, country_idx: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([self.customer(customer_idx), self.country(country_idx), numeric], dim=-1))


class FusionNetwork(nn.Module):
    """Concatenation -> Linear -> GELU -> LayerNorm -> Dropout -> Linear (blueprint 5.4)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(parts, dim=-1))
