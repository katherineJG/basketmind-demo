"""The parts of the inference service that need no model, and a service that answers from precomputed results.

`DataService` answers every question about the data (customers, history, products, related products,
overview, evidence for a recommendation). `BasketMindService` (service.py) adds the model;
`PrecomputedService` (below) reads the model's answers from a bundle written by `scripts/export_demo.py`.
Nothing in this module imports torch, so a small host can serve the full API without installing it.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

BUNDLE_BASKET_COLUMNS = [
    "InvoiceNo",
    "CustomerID",
    "timestamp",
    "Country",
    "items",
    "quantities",
    "n_items",
    "total_value",
]


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


class DataService:
    """Everything that can be answered from the data alone. Subclasses supply the recommendations."""

    def __init__(
        self,
        baskets: pd.DataFrame,
        items: list[str],
        popularity: np.ndarray,
        train_customers,
        adjacency: sp.csr_matrix,
        descriptions: dict[str, str],
        as_of: pd.Timestamp,
        metadata: dict[str, Any] | None = None,
        analytics: dict[str, Any] | None = None,
    ):
        self.baskets, self.items, self.popularity = baskets, list(items), np.asarray(popularity)
        self.train_customers = set(train_customers)  # customers with a basket in the training period
        self.adjacency, self.descriptions = adjacency, descriptions
        self.as_of = pd.Timestamp(as_of)
        self.metadata = metadata or {}
        self.analytics = analytics or {}  # precomputed by scripts/export_analytics.py
        self.device = "none"
        self.by_customer = {
            c: g.sort_values(["timestamp", "InvoiceNo"]) for c, g in baskets.groupby("CustomerID")
        }
        self.item_index = {s: i for i, s in enumerate(self.items)}
        self.product_counts = Counter(s for items in baskets["items"] for s in items)
        self.last_sold: dict[str, pd.Timestamp] = {}
        for basket_items, ts in zip(baskets["items"], baskets["timestamp"], strict=True):
            for s in basket_items:
                if self.last_sold.get(s, ts) <= ts:
                    self.last_sold[s] = ts

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

    def _bought(self, customer_id: str, past: pd.DataFrame) -> dict[str, tuple[int, pd.Timestamp]]:
        seen: dict[str, tuple[int, pd.Timestamp]] = {}
        for items, ts in zip(past["items"], past["timestamp"], strict=True):
            for s in items:
                n, _ = seen.get(s, (0, ts))
                seen[s] = (n + 1, ts)
        return seen

    def _recommendations(
        self, customer_id: str, past: pd.DataFrame, picks: list[tuple[int, float, float]]
    ) -> list[Recommendation]:
        """Turn (catalogue index, score, predicted quantity) picks, best first, into Recommendation rows."""
        bought = self._bought(customer_id, past)
        out = []
        for rank, (item, score, qty) in enumerate(picks, 1):
            sid = self.items[int(item)]
            n_bought, last = bought.get(sid, (0, None))
            out.append(
                Recommendation(
                    rank,
                    sid,
                    self._describe(sid),
                    float(score),
                    n_bought > 0,
                    n_bought,
                    None if last is None else str(last.date()),
                    float(qty),
                )
            )
        return out

    # --------------------------------------------------------------------------------------- queries
    def recommend(
        self, customer_id: str, k: int = 10, exclude_history: bool = False, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        raise NotImplementedError

    def explain(self, customer_id: str, product_id: str, top: int = 5) -> dict[str, Any]:
        raise NotImplementedError

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
            "seen_in_training": customer_id in self.train_customers,
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
                "product_id": self.items[int(row.indices[i])],
                "description": self._describe(self.items[int(row.indices[i])]),
                "association": float(row.data[i]),
            }
            for i in order
        ]

    def _evidence(self, customer_id: str, product_id: str, top: int) -> dict[str, Any]:
        """Data-side evidence for one recommendation: purchase history, co-purchase links, popularity rank."""
        if product_id not in self.item_index:
            raise NotFound(f"product {product_id!r} is not in the catalogue")
        past = self._history(customer_id, self.as_of)
        bought = self._bought(customer_id, past)
        idx = self.item_index[product_id]
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
            "bought_before": n_bought > 0,
            "times_bought": n_bought,
            "last_bought": None if last is None else str(last.date()),
            "linked_to_history": [
                {"product_id": s, "description": self._describe(s), "association": float(w)} for w, s in links
            ],
            "popularity_rank": int((self.popularity > self.popularity[idx]).sum()) + 1,
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


class PrecomputedService(DataService):
    """Answers recommendations from lists the real model computed ahead of time (see scripts/export_demo.py).

    For every customer the bundle holds the model's top `k` products under both policies (repeats allowed,
    and only products new to the customer), with scores, predicted quantities and each product's rank over
    the whole catalogue. Everything else comes from the data exactly as in the live service. The model
    itself is not run.
    """

    def __init__(self, *args, customer_ids, lists: dict[str, np.ndarray], **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "precomputed"
        self._row = {c: i for i, c in enumerate(customer_ids)}
        self._lists = lists  # items (n, 2, k) int16 with -1 padding, score, quantity, rank

    @classmethod
    def load(cls, directory: Path) -> "PrecomputedService":
        d = Path(directory)
        meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
        baskets = pd.read_parquet(d / "baskets.parquet")
        recs = np.load(d / "recs.npz", allow_pickle=False)
        return cls(
            baskets,
            meta["items"],
            np.asarray(meta["popularity"]),
            meta["train_customers"],
            sp.load_npz(d / "adjacency.npz").tocsr(),
            meta["descriptions"],
            pd.Timestamp(meta["as_of"]),
            meta["metadata"],
            meta["analytics"],
            customer_ids=recs["customers"].tolist(),
            lists={k: recs[k] for k in ("items", "score", "quantity", "rank")},
        )

    def _slot(self, customer_id: str, exclude_history: bool) -> tuple[int, int]:
        if customer_id not in self._row:
            raise NotFound(f"customer {customer_id!r} has no precomputed recommendations")
        return self._row[customer_id], int(exclude_history)

    def recommend(
        self, customer_id: str, k: int = 10, exclude_history: bool = False, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        if as_of is not None:
            raise ValueError("the precomputed demo only answers for the default point in time")
        past = self._history(customer_id, self.as_of)
        row, policy = self._slot(customer_id, exclude_history)
        items = self._lists["items"][row, policy]
        n = min(k, int((items >= 0).sum()))
        picks = [
            (items[j], self._lists["score"][row, policy, j], self._lists["quantity"][row, policy, j])
            for j in range(n)
        ]
        return self._recommendations(customer_id, past, picks)

    def explain(self, customer_id: str, product_id: str, top: int = 5) -> dict[str, Any]:
        out = self._evidence(customer_id, product_id, top)  # also checks the product and the customer
        idx = self.item_index[product_id]
        for policy in (0, 1):
            row = self._slot(customer_id, bool(policy))[0]
            hit = np.flatnonzero(self._lists["items"][row, policy] == idx)
            if len(hit):
                out["model_rank"] = int(self._lists["rank"][row, policy, hit[0]])
                out["model_score"] = float(self._lists["score"][row, policy, hit[0]])
                return out
        raise NotFound(
            f"the demo only explains products in a customer's precomputed top {self._lists['items'].shape[2]}"
        )
