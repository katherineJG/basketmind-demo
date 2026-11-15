"""Which kind of customer or target is each example? Labels used to break results down."""

from typing import Any

import numpy as np
import pandas as pd

HEAD_N = 100  # a product is "head" if it is among the 100 most frequent in the training baskets
BUCKETS = [("1", 1, 1), ("2-3", 2, 3), ("4-10", 4, 10), ("11+", 11, 10**9)]
HEAD_LABELS = ["no head items", "some head items", "mostly head items"]


def history_bucket(n: int) -> str:
    return next(label for label, lo, hi in BUCKETS if lo <= n <= hi)


def assign_slices(task: Any, basket_country: dict[str, str], split: str) -> pd.DataFrame:
    """One row per example of `split` (index = example_id, the row position in task.examples[split]).

    cold_start      the customer has no basket in the training split (no learned customer vector)
    history_bucket  number of earlier baskets in the example's history: 1, 2-3, 4-10, 11+
    country_group   'United Kingdom' or 'Other', from the customer's most recent earlier basket
    month           the target basket's month (temporal drift)
    target_bucket   how much of the target basket is made of the HEAD_N most popular training products:
                    none / under half / half or more
    """
    ex = task.examples[split]
    head = set(np.argsort(-task.popularity)[:HEAD_N].tolist())
    head_share = np.array([np.mean([i in head for i in idx]) for idx in ex["target_idx"]])
    hist_len = ex["history_invoices"].map(len).to_numpy()
    last_country = [basket_country[h[-1]] for h in ex["history_invoices"]]
    return pd.DataFrame(
        {
            "customer": ex["CustomerID"].to_numpy(),
            "cold_start": ~ex["CustomerID"].isin(task.train_customers).to_numpy(),
            "history_len": hist_len,
            "history_bucket": [history_bucket(n) for n in hist_len],
            "country_group": np.where(np.array(last_country) == "United Kingdom", "United Kingdom", "Other"),
            "month": ex["target_timestamp"].dt.strftime("%Y-%m").to_numpy(),
            "head_share": head_share,
            "target_bucket": pd.cut(head_share, [-0.1, 0.0, 0.4999, 1.0], labels=HEAD_LABELS).astype(str),
        },
        index=pd.RangeIndex(len(ex), name="example_id"),
    )


def popularity_bias(
    tops: np.ndarray, popularity: np.ndarray, n_items: int, head_n: int = HEAD_N
) -> dict[str, float]:
    """How concentrated a model's top-10 lists are, from an (examples, 10) array of recommended item indices.

    coverage          share of the catalogue that appears in at least one top-10 list
    head_share        share of all recommendations that are head products
    mean_percentile   average popularity percentile of recommended products (100 = the most popular)
    """
    head = np.zeros(n_items, dtype=bool)
    head[np.argsort(-popularity)[:head_n]] = True
    rank = np.argsort(np.argsort(popularity))  # 0 = least popular
    flat = tops.ravel()
    return {
        "coverage": len(np.unique(flat)) / n_items,
        "head_share": float(head[flat].mean()),
        "mean_percentile": float(100 * (rank[flat] / (n_items - 1)).mean()),
    }
