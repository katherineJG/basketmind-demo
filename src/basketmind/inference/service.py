"""Inference service: turns a customer's history into ranked product recommendations.

It reuses the training code for everything that could drift (features, batch assembly, the product graph),
so a customer scored here gets exactly the inputs the model saw in training. The service answers questions
about a single point in time (`as_of`, by default just after the last basket in the data).
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from basketmind.baselines.task import Task, load_task
from basketmind.features.build import (
    FeatureStore,
    build_feature_store,
    customer_feature_rows,
    customer_history_index,
    normalise_customer_feats,
)
from basketmind.graph import GraphSnapshot, build_product_graph
from basketmind.models import BasketMindModel, Batch
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import build_batch
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

MAX_HISTORY = 50  # the model's maximum sequence length


class NotFound(KeyError):
    """The customer or product is not in the data (or the customer has no basket before `as_of`)."""


@dataclass
class Recommendation:
    rank: int
    product_id: str
    description: str
    score: float  # raw model logit: ranks products for this customer; it is NOT a calibrated probability
    in_history: bool
    times_bought: int
    last_bought: str | None
    predicted_quantity: (
        float  # estimated units if the customer buys it (log1p quantity head, back-transformed)
    )


def load_descriptions(processed_dir: Path) -> dict[str, str]:
    """The most common product text for each stock code."""
    events = pd.read_parquet(processed_dir / "purchase_events.parquet", columns=["StockCode", "Description"])
    return (
        events.dropna().groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0]).to_dict()
    )


class BasketMindService:
    def __init__(
        self,
        model: BasketMindModel,
        task: Task,
        store: FeatureStore,
        baskets: pd.DataFrame,
        graph: GraphSnapshot,
        descriptions: dict[str, str],
        feat_cfg: dict[str, Any],
        device: torch.device,
        as_of: pd.Timestamp | None = None,
        metadata: dict[str, Any] | None = None,
        analytics: dict[str, Any] | None = None,
    ):
        self.model, self.task, self.store, self.graph = model.eval(), task, store, graph
        self.feat_cfg, self.device, self.metadata = feat_cfg, device, metadata or {}
        self.analytics = analytics or {}  # precomputed by scripts/export_analytics.py
        self.descriptions = descriptions
        self.baskets = baskets
        self.as_of = (
            pd.Timestamp(as_of) if as_of is not None else baskets["timestamp"].max() + pd.Timedelta(minutes=1)
        )
        self.by_customer = {
            c: g.sort_values(["timestamp", "InvoiceNo"]) for c, g in baskets.groupby("CustomerID")
        }
        self.hist_index = customer_history_index(baskets)
        self.item_index = {s: i for i, s in enumerate(task.items)}
        self._graph_gpu = graph.to(device)
        # The graph encoder is the costly part of a forward pass (it touches every edge) and does not
        # depend on the customer, so it runs once here instead of on every request.
        self._graph_vectors = (
            self.model.encode_graph(self._graph_gpu) if self.model.branches["graph"] else None
        )
        n = task.n_items
        ei, w = graph.edge_index.numpy(), graph.edge_weight.numpy()
        self.adjacency = sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n))
        counts = Counter(s for items in baskets["items"] for s in items)
        self.product_counts = counts
        self.last_sold = {}
        for items, ts in zip(baskets["items"], baskets["timestamp"], strict=True):
            for s in items:
                if self.last_sold.get(s, ts) <= ts:
                    self.last_sold[s] = ts

    # ------------------------------------------------------------------------------------ building
    @classmethod
    def from_artifacts(cls, root: Path, device: str = "auto") -> "BasketMindService":
        """Load the processed data and the exported checkpoint (see scripts/export_model.py)."""
        cfg = load_config("inference", root / "configs")
        dev = get_device(device)
        proc = root / "data" / "processed"
        task = load_task(proc)
        baskets = pd.read_parquet(proc / "baskets.parquet")
        feat_cfg = load_config("features", root / "configs")
        store = build_feature_store(task, baskets, feat_cfg)
        ckpt = root / cfg["checkpoint"]
        model, payload = load_checkpoint(ckpt, dev)
        meta_path = root / "artifacts" / "model_metadata" / "model.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metadata |= {"checkpoint": cfg["checkpoint"], "checkpoint_metadata": payload["metadata"]}
        as_of = baskets["timestamp"].max() + pd.Timedelta(minutes=1)
        graph = build_product_graph(baskets, as_of, task.items, load_config("graph", root / "configs"))
        analytics = {
            p.stem: json.loads(p.read_text()) for p in (root / "artifacts" / "analytics").glob("*.json")
        }
        return cls(
            model,
            task,
            store,
            baskets,
            graph,
            load_descriptions(proc),
            feat_cfg,
            dev,
            as_of,
            metadata,
            analytics,
        )

    # ------------------------------------------------------------------------------------ helpers
    def _describe(self, product_id: str) -> str:
        return self.descriptions.get(product_id, "")

    def _history(self, customer_id: str, as_of: pd.Timestamp) -> pd.DataFrame:
        if customer_id not in self.by_customer:
            raise NotFound(f"customer {customer_id!r} not found")
        past = self.by_customer[customer_id]
        past = past[past["timestamp"] < as_of]
        if past.empty:
            raise NotFound(f"customer {customer_id!r} has no basket before {as_of}")
        return past

    def _prepare(self, customer_id: str, as_of: pd.Timestamp) -> tuple[Batch, pd.DataFrame]:
        """Model input for one customer at `as_of` (built by the training code) and all earlier baskets."""
        past = self._history(customer_id, as_of)
        recent = past.tail(MAX_HISTORY)
        raw, _ = customer_feature_rows(
            pd.DataFrame({"CustomerID": [customer_id], "target_timestamp": [as_of]}),
            self.baskets,
            self.feat_cfg["recency_reference"],
            index=self.hist_index,
        )
        batch = build_batch(
            self.task,
            self.store,
            [recent["InvoiceNo"].tolist()],
            np.array([self.store.customer_index.get(customer_id, 0)]),
            np.array([self.store.country_index.get(recent["Country"].iat[-1], 0)]),
            normalise_customer_feats(raw, self.store.stats["customer"]),
        )
        return batch.to(self.device), past

    def _forward(
        self, batch: Batch, graph: GraphSnapshot | None = None, items: np.ndarray | None = None
    ) -> dict:
        # A graph passed in (another point in time) is encoded on the spot; the default one is cached.
        g = graph.to(self.device) if graph is not None else self._graph_gpu
        cached = self._graph_vectors if graph is None else None
        quantity_items = None if items is None else torch.as_tensor(items, device=self.device)[None, :]
        with torch.no_grad():
            return self.model(batch, g, quantity_items, graph_vectors=cached)

    def score_customer(
        self, customer_id: str, as_of: pd.Timestamp | None = None, graph: GraphSnapshot | None = None
    ) -> tuple[torch.Tensor, pd.DataFrame]:
        """Logits over the whole catalogue for one customer, and the baskets that were used as history."""
        batch, past = self._prepare(customer_id, pd.Timestamp(as_of) if as_of is not None else self.as_of)
        return self._forward(batch, graph)["basket_logits"][0], past

    def _bought(self, customer_id: str, past: pd.DataFrame) -> dict[str, tuple[int, pd.Timestamp]]:
        seen: dict[str, tuple[int, pd.Timestamp]] = {}
        for items, ts in zip(past["items"], past["timestamp"], strict=True):
            for s in items:
                n, _ = seen.get(s, (0, ts))
                seen[s] = (n + 1, ts)
        return seen

    # --------------------------------------------------------------------------------------- queries
    def recommend(
        self, customer_id: str, k: int = 10, exclude_history: bool = False, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        batch, past = self._prepare(customer_id, pd.Timestamp(as_of) if as_of is not None else self.as_of)
        logits = self._forward(batch)["basket_logits"][0]
        bought = self._bought(customer_id, past)
        if exclude_history:
            owned = [self.item_index[s] for s in bought if s in self.item_index]
            logits = logits.clone()
            logits[torch.as_tensor(owned, device=logits.device, dtype=torch.long)] = float("-inf")
        top = logits.topk(min(k, self.task.n_items)).indices
        top = top[torch.isfinite(logits[top])].cpu().numpy()  # excluded (-inf) items are never returned
        if len(top) == 0:
            return []
        quantities = np.maximum(np.expm1(self._forward(batch, items=top)["quantity"][0].cpu().numpy()), 0.0)
        out = []
        for rank, (item, qty) in enumerate(zip(top, quantities, strict=True), 1):
            sid = self.task.items[int(item)]
            n_bought, last = bought.get(sid, (0, None))
            out.append(
                Recommendation(
                    rank,
                    sid,
                    self._describe(sid),
                    float(logits[int(item)]),
                    n_bought > 0,
                    n_bought,
                    None if last is None else str(last.date()),
                    float(qty),
                )
            )
        return out

    def customers(self, limit: int = 50, offset: int = 0, prefix: str = "") -> list[dict[str, Any]]:
        rows = [
            {
                "customer_id": c,
                "n_baskets": len(g),
                "last_basket": str(g["timestamp"].iat[-1].date()),
                "country": g["Country"].iat[-1],
            }
            for c, g in self.by_customer.items()
            if c.startswith(prefix)
        ]
        rows.sort(key=lambda r: (-r["n_baskets"], r["customer_id"]))
        return rows[offset : offset + limit]

    def n_customers(self, prefix: str = "") -> int:
        return sum(1 for c in self.by_customer if c.startswith(prefix))

    def customer(self, customer_id: str) -> dict[str, Any]:
        past = self._history(customer_id, self.as_of)
        return {
            "customer_id": customer_id,
            "n_baskets": len(past),
            "first_basket": str(past["timestamp"].iat[0].date()),
            "last_basket": str(past["timestamp"].iat[-1].date()),
            "country": past["Country"].iat[-1],
            "total_spend": float(past["total_value"].sum()),
            "avg_basket_size": float(past["n_items"].mean()),
            "seen_in_training": customer_id in self.store.customer_index,
        }

    def history(self, customer_id: str, limit: int = 20) -> list[dict[str, Any]]:
        past = self._history(customer_id, self.as_of).tail(limit)[::-1]
        return [
            {
                "invoice": r.InvoiceNo,
                "date": str(r.timestamp.date()),
                "n_items": int(r.n_items),
                "total_value": float(r.total_value),
                "items": [
                    {"product_id": s, "description": self._describe(s), "quantity": int(q)}
                    for s, q in zip(r.items, r.quantities, strict=True)
                ],
            }
            for r in past.itertuples()
        ]

    def product(self, product_id: str) -> dict[str, Any]:
        if product_id not in self.product_counts:
            raise NotFound(f"product {product_id!r} not found")
        in_catalogue = product_id in self.item_index
        return {
            "product_id": product_id,
            "description": self._describe(product_id),
            "baskets_containing": int(self.product_counts[product_id]),
            "last_sold": str(self.last_sold[product_id].date()),
            "in_catalogue": in_catalogue,
            "n_related": int(self.adjacency[self.item_index[product_id]].nnz) if in_catalogue else 0,
        }

    def related(self, product_id: str, k: int = 10) -> list[dict[str, Any]]:
        if product_id not in self.item_index:
            raise NotFound(f"product {product_id!r} is not in the catalogue")
        row = self.adjacency[self.item_index[product_id]]
        order = np.argsort(-row.data)[:k]
        return [
            {
                "product_id": self.task.items[int(row.indices[i])],
                "description": self._describe(self.task.items[int(row.indices[i])]),
                "association": float(row.data[i]),
            }
            for i in order
        ]

    def explain(self, customer_id: str, product_id: str, top: int = 5) -> dict[str, Any]:
        """Evidence for one recommendation from the data (not from inside the network)."""
        if product_id not in self.item_index:
            raise NotFound(f"product {product_id!r} is not in the catalogue")
        logits, past = self.score_customer(customer_id)
        bought = self._bought(customer_id, past)
        idx = self.item_index[product_id]
        rank = int((logits > logits[idx]).sum().item()) + 1
        row = self.adjacency[idx]
        weights = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
        links = sorted(
            (
                (weights[self.item_index[s]], s)
                for s in bought
                if s in self.item_index and self.item_index[s] in weights
            ),
            reverse=True,
        )[:top]
        n_bought, last = bought.get(product_id, (0, None))
        return {
            "customer_id": customer_id,
            "product_id": product_id,
            "description": self._describe(product_id),
            "model_rank": rank,
            "model_score": float(logits[idx]),
            "bought_before": n_bought > 0,
            "times_bought": n_bought,
            "last_bought": None if last is None else str(last.date()),
            "linked_to_history": [
                {"product_id": s, "description": self._describe(s), "association": float(w)} for w, s in links
            ],
            "popularity_rank": int((self.task.popularity > self.task.popularity[idx]).sum()) + 1,
        }

    def analytic(self, name: str) -> dict[str, Any]:
        """A precomputed analysis ('cancellations' or 'performance'); NotFound if it has not been exported."""
        if name not in self.analytics:
            raise NotFound(f"analytics {name!r} not found; run scripts/export_analytics.py")
        return self.analytics[name]

    def overview(self) -> dict[str, Any]:
        b = self.baskets
        monthly = b.groupby(b["timestamp"].dt.strftime("%Y-%m")).agg(
            baskets=("InvoiceNo", "size"), value=("total_value", "sum")
        )
        top = self.product_counts.most_common(10)
        return {
            "customers": len(self.by_customer),
            "baskets": len(b),
            "products": len(self.product_counts),
            "first_basket": str(b["timestamp"].min().date()),
            "last_basket": str(b["timestamp"].max().date()),
            "total_value": float(b["total_value"].sum()),
            "monthly": [
                {"month": m, "baskets": int(r.baskets), "value": float(r.value)}
                for m, r in monthly.iterrows()
            ],
            "top_products": [
                {"product_id": s, "description": self._describe(s), "baskets": int(n)} for s, n in top
            ],
            "top_countries": b["Country"]
            .value_counts()
            .head(5)
            .rename_axis("country")
            .reset_index(name="baskets")
            .to_dict("records"),
        }
