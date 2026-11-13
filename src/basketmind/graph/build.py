"""Product co-occurrence graphs with PPMI weights. Snapshots only ever see baskets before their cutoff."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class GraphSnapshot:
    """Sparse symmetric product graph (no self loops) built from baskets strictly before `cutoff_time`."""

    edge_index: torch.Tensor  # (2, E) long, both directions
    edge_weight: torch.Tensor  # (E,) float, PPMI > 0
    cutoff_time: pd.Timestamp
    n_nodes: int
    n_baskets: int  # baskets used

    def to(self, device: torch.device) -> "GraphSnapshot":
        return GraphSnapshot(
            self.edge_index.to(device),
            self.edge_weight.to(device),
            self.cutoff_time,
            self.n_nodes,
            self.n_baskets,
        )


def calculate_ppmi(
    cooc: sp.spmatrix,
    item_counts: np.ndarray,
    total: float,
    min_cooccurrence: float = 0.0,
    support: sp.spmatrix | None = None,
) -> sp.coo_matrix:
    """Positive PMI of a co-occurrence matrix: max(0, log(c_ij * N / (c_i * c_j))).

    `cooc[i, j]` = (weighted) number of baskets holding both items, `item_counts[i]` = baskets holding i,
    `total` = N (all baskets). The diagonal is dropped. Pairs seen in fewer than `min_cooccurrence` baskets
    are removed, judged on `support` (raw basket counts) when given, so temporal decay reweights edges
    without deleting them; otherwise on `cooc` itself.
    """
    c = sp.coo_matrix(cooc)
    seen = c.data if support is None else np.asarray(sp.csr_matrix(support)[c.row, c.col]).ravel()
    keep = (c.row != c.col) & (seen >= max(min_cooccurrence, np.finfo(float).tiny))
    row, col, val = c.row[keep], c.col[keep], c.data[keep]
    pmi = np.log(val * total / (item_counts[row] * item_counts[col]))
    pos = pmi > 0
    return sp.coo_matrix((pmi[pos], (row[pos], col[pos])), shape=c.shape)


def build_product_graph(
    baskets: pd.DataFrame, cutoff_time: pd.Timestamp, items: list[str], config: dict[str, Any]
) -> GraphSnapshot:
    """Graph over `items` from baskets with timestamp < cutoff_time. Later baskets cannot influence it.

    `baskets` needs `items` (list of stock codes) and `timestamp`. With temporal decay enabled each
    basket is weighted 0.5 ** (age_days / half_life_days), age measured back from the cutoff.
    """
    cutoff_time = pd.Timestamp(cutoff_time)
    past = baskets[baskets["timestamp"] < cutoff_time]
    vocab = {s: i for i, s in enumerate(items)}
    rows, cols = [], []
    for r, its in enumerate(past["items"]):
        idx = sorted({vocab[s] for s in its if s in vocab})
        rows.extend([r] * len(idx))
        cols.extend(idx)
    b = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(past), len(items)))
    decay = config["temporal_decay"]
    if decay["enabled"]:
        age = (cutoff_time - past["timestamp"]).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age / decay["half_life_days"])
    else:
        w = np.ones(len(past))
    bw = sp.diags(w) @ b
    cooc = (b.T @ bw).tocsr()
    counts = np.asarray(bw.sum(axis=0)).ravel()
    ppmi = calculate_ppmi(
        cooc,
        np.maximum(counts, 1e-12),
        float(w.sum()),
        config["min_cooccurrence"],
        support=(b.T @ b).tocsr(),
    )
    return GraphSnapshot(
        edge_index=torch.as_tensor(np.vstack([ppmi.row, ppmi.col]), dtype=torch.long),
        edge_weight=torch.as_tensor(ppmi.data, dtype=torch.float32),
        cutoff_time=cutoff_time,
        n_nodes=len(items),
        n_baskets=len(past),
    )


def create_graph_snapshots(
    baskets: pd.DataFrame, boundaries: list[pd.Timestamp], items: list[str], config: dict[str, Any]
) -> dict[pd.Timestamp, GraphSnapshot]:
    """One snapshot per boundary. Examples inside [boundary_k, boundary_k+1) should use snapshot k."""
    return {pd.Timestamp(b): build_product_graph(baskets, b, items, config) for b in boundaries}
