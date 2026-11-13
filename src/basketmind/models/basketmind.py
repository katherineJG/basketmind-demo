"""BasketMind: temporal Transformer + product graph + customer context -> fusion -> multi-task heads."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from basketmind.graph import GraphSnapshot
from basketmind.models.batch import Batch, pad_flat
from basketmind.models.customer import CustomerContextEncoder, FusionNetwork
from basketmind.models.graph_encoder import ProductGraphEncoder
from basketmind.models.temporal import TemporalEncoder


class PairHead(nn.Module):
    """Scores (customer state, item) pairs: MLP over [fused ; item vector] -> one number."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, fused: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """fused (B, D), items (B, K, D) -> (B, K)."""
        pair = torch.cat([fused[:, None, :].expand(-1, items.shape[1], -1), items], dim=-1)
        return self.net(pair).squeeze(-1)


class BasketMindModel(nn.Module):
    """forward(batch, graph, quantity_items) -> dict with

    basket_logits (B, n_items): relevance of every catalogue item (Head A; use with weighted BCE),
    fused (B, D): the shared customer state,
    quantity (B, K): predicted log1p quantity for `quantity_items` (Head B), only if items are given,
    reversal (B, K): reversal logit for those pairs (Head C), only if the head is enabled.

    `branches` in the config switch the temporal, graph and customer encoders on or off for ablations.
    Item scores are fused . item_vector + bias; the item vector is the graph encoder's output when the graph
    branch is on, otherwise the raw shared item embedding.
    """

    def __init__(self, cfg: dict[str, Any], n_items: int, n_customers: int, n_countries: int):
        super().__init__()
        self.init_args = {
            "cfg": cfg,
            "n_items": n_items,
            "n_customers": n_customers,
            "n_countries": n_countries,
        }
        self.cfg = cfg
        self.branches = cfg["branches"]
        if not any(self.branches.values()):
            raise ValueError("at least one branch must be enabled")
        d = cfg["item_embedding_dim"]
        if cfg["fusion"]["output_dim"] != d or (
            self.branches["graph"]
            and (cfg["graph_encoder"]["input_dim"] != d or cfg["graph_encoder"]["hidden_dim"] != d)
        ):
            raise ValueError(
                "item embedding, graph and fusion dimensions must all match (item scores are dot products)"
            )
        self.item_emb = nn.Embedding(n_items, d)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        in_dim = 0
        if self.branches["temporal"]:
            self.temporal = TemporalEncoder(cfg["temporal_encoder"], cfg["basket_feature_dim"], d)
            in_dim += cfg["temporal_encoder"]["embedding_dim"]
        if self.branches["graph"]:
            self.graph_encoder = ProductGraphEncoder(cfg["graph_encoder"])
            in_dim += cfg["graph_encoder"]["hidden_dim"]
        if self.branches["customer"]:
            self.customer = CustomerContextEncoder(cfg["customer_encoder"], n_customers, n_countries)
            in_dim += self.customer.out_dim
        self.fusion = FusionNetwork(in_dim, d, cfg["fusion"]["dropout"])
        self.quantity_head = PairHead(d, cfg["heads"]["quantity_hidden"])
        self.reversal_head = (
            PairHead(d, cfg["heads"]["quantity_hidden"]) if cfg["heads"]["reversal_enabled"] else None
        )

    def forward(
        self,
        batch: Batch,
        graph: GraphSnapshot | None = None,
        quantity_items: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        counts = batch.basket_counts
        parts: list[torch.Tensor] = []
        item_vectors = self.item_emb.weight
        if self.branches["temporal"]:
            basket_vecs = F.embedding_bag(
                batch.item_ids, self.item_emb.weight, batch.item_offsets, mode="mean"
            )
            parts.append(self.temporal(basket_vecs, batch.basket_feats, counts))
        if self.branches["graph"]:
            if graph is None:
                raise ValueError("the graph branch is enabled, so a GraphSnapshot is required")
            item_vectors = self.graph_encoder(self.item_emb.weight, graph)
            graph_baskets = F.embedding_bag(batch.item_ids, item_vectors, batch.item_offsets, mode="mean")
            padded, _ = pad_flat(graph_baskets, counts)
            parts.append(padded.sum(1) / counts[:, None])  # mean over real baskets (padding rows are zero)
        if self.branches["customer"]:
            parts.append(self.customer(batch.customer_idx, batch.country_idx, batch.customer_feats))

        fused = self.fusion(parts)
        out = {"fused": fused, "basket_logits": fused @ item_vectors.T + self.item_bias}
        if quantity_items is not None:
            chosen = item_vectors[quantity_items]
            out["quantity"] = self.quantity_head(fused, chosen)
            if self.reversal_head is not None:
                out["reversal"] = self.reversal_head(fused, chosen)
        return out
