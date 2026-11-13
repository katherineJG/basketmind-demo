"""Model input format. Histories are stored flat (no padding) and padded inside the model."""

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    """B examples; example b has `basket_counts[b]` history baskets (>= 1, oldest first, at most 50).

    Baskets of all examples are concatenated in order, N = basket_counts.sum() in total.
    """

    item_ids: torch.Tensor  # (T,) long: catalogue index of every item of every basket, flat
    item_offsets: torch.Tensor  # (N,) long: start of each basket inside item_ids (an empty basket has size 0)
    basket_counts: torch.Tensor  # (B,) long: history baskets per example
    basket_feats: torch.Tensor  # (N, basket_feature_dim) float, normalised
    customer_idx: torch.Tensor  # (B,) long: 0 = unseen customer, else 1..n_customers
    country_idx: torch.Tensor  # (B,) long: 0 = unknown country, else 1..n_countries
    customer_feats: torch.Tensor  # (B, numeric_dim) float, normalised with train statistics

    @property
    def size(self) -> int:
        return int(self.basket_counts.numel())

    def to(self, device: torch.device) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def validate(self, basket_feature_dim: int, numeric_dim: int) -> None:
        n = int(self.basket_counts.sum())
        assert self.basket_counts.min() >= 1, "every example needs at least one history basket"
        assert self.item_offsets.shape == (n,) and self.basket_feats.shape == (n, basket_feature_dim)
        assert self.customer_idx.shape == self.country_idx.shape == (self.size,)
        assert self.customer_feats.shape == (self.size, numeric_dim)


def pad_flat(flat: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, D) flat rows + (B,) counts -> padded (B, L, D) and a bool mask (B, L), True where real."""
    b, length = counts.numel(), int(counts.max())
    mask = torch.arange(length, device=counts.device)[None, :] < counts[:, None]
    out = flat.new_zeros(b, length, flat.shape[-1])
    out[mask] = flat  # row-major order of `mask` matches the flat layout
    return out, mask
