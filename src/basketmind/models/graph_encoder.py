"""GraphSAGE encoder over the product graph with PPMI edge weights."""

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from basketmind.graph import GraphSnapshot


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE mean aggregator with edge weights: h_i = W_l * (sum_j w_ij x_j / sum_j w_ij) + W_r * x_i.

    PyG's stock SAGEConv ignores edge weights, so the weighted mean is written here on top of PyG's
    MessagePassing. Nodes without edges aggregate zeros and keep only their own (W_r) term.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # edge_index[0] = source, edge_index[1] = target; normalise weights over each target's incoming edges
        total = torch.zeros(x.shape[0], device=x.device).index_add(0, edge_index[1], edge_weight)
        norm = edge_weight / total[edge_index[1]].clamp(min=1e-12)
        agg = self.propagate(edge_index, x=x, norm=norm, size=(x.shape[0], x.shape[0]))
        return self.lin_neigh(agg) + self.lin_self(x)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return x_j * norm[:, None]


class ProductGraphEncoder(nn.Module):
    """Stack of weighted SAGE layers: (n_items, input_dim) node features -> (n_items, hidden_dim)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        if cfg["aggregation"] != "mean":
            raise ValueError("only mean aggregation is implemented")
        dims = [cfg["input_dim"]] + [cfg["hidden_dim"]] * cfg["num_layers"]
        self.convs = nn.ModuleList(WeightedSAGEConv(a, b) for a, b in zip(dims, dims[1:], strict=False))
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor, graph: GraphSnapshot) -> torch.Tensor:
        if graph.n_nodes != x.shape[0]:
            raise ValueError(f"graph has {graph.n_nodes} nodes but {x.shape[0]} node features were given")
        for i, conv in enumerate(self.convs):
            x = conv(x, graph.edge_index, graph.edge_weight)
            if i < len(self.convs) - 1:
                x = self.drop(self.act(x))
        return x
