"""Inference service: turns a customer's history into ranked product recommendations.

It reuses the training code for everything that could drift (features, batch assembly, the product graph),
so a customer scored here gets exactly the inputs the model saw in training. The service answers questions
about a single point in time (`as_of`, by default just after the last basket in the data). Everything that
needs no model lives in `data_service.DataService`.
"""

import json
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
from basketmind.inference.data_service import DataService, NotFound, Recommendation, load_descriptions
from basketmind.models import BasketMindModel, Batch
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import build_batch
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

__all__ = ["BasketMindService", "NotFound", "Recommendation", "load_descriptions"]
MAX_HISTORY = 50  # the model's maximum sequence length


class BasketMindService(DataService):
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
        n = task.n_items
        ei, w = graph.edge_index.numpy(), graph.edge_weight.numpy()
        super().__init__(
            baskets,
            task.items,
            task.popularity,
            store.customer_index,
            sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n)),
            descriptions,
            pd.Timestamp(as_of)
            if as_of is not None
            else baskets["timestamp"].max() + pd.Timedelta(minutes=1),
            metadata,
            analytics,
        )
        self.model, self.task, self.store, self.graph = model.eval(), task, store, graph
        self.feat_cfg, self.device = feat_cfg, device
        self.hist_index = customer_history_index(baskets)
        self._graph_gpu = graph.to(device)
        # The graph encoder is the costly part of a forward pass (it touches every edge) and does not
        # depend on the customer, so it runs once here instead of on every request.
        self._graph_vectors = (
            self.model.encode_graph(self._graph_gpu) if self.model.branches["graph"] else None
        )

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

    # ------------------------------------------------------------------------------------ model
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

    # --------------------------------------------------------------------------------------- queries
    def recommend(
        self, customer_id: str, k: int = 10, exclude_history: bool = False, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        batch, past = self._prepare(customer_id, pd.Timestamp(as_of) if as_of is not None else self.as_of)
        logits = self._forward(batch)["basket_logits"][0]
        if exclude_history:
            bought = self._bought(customer_id, past)
            owned = [self.item_index[s] for s in bought if s in self.item_index]
            logits = logits.clone()
            logits[torch.as_tensor(owned, device=logits.device, dtype=torch.long)] = float("-inf")
        top = logits.topk(min(k, self.task.n_items)).indices
        top = top[torch.isfinite(logits[top])].cpu().numpy()  # excluded (-inf) items are never returned
        if len(top) == 0:
            return []
        quantities = np.maximum(np.expm1(self._forward(batch, items=top)["quantity"][0].cpu().numpy()), 0.0)
        picks = [(int(i), float(logits[int(i)]), float(q)) for i, q in zip(top, quantities, strict=True)]
        return self._recommendations(customer_id, past, picks)

    def explain(self, customer_id: str, product_id: str, top: int = 5) -> dict[str, Any]:
        """Data evidence for one recommendation (not from inside the network), plus the model's rank."""
        out = self._evidence(customer_id, product_id, top)  # also checks the product and the customer
        logits, _ = self.score_customer(customer_id)
        idx = self.item_index[product_id]
        out["model_rank"] = int((logits > logits[idx]).sum().item()) + 1
        out["model_score"] = float(logits[idx])
        return out
