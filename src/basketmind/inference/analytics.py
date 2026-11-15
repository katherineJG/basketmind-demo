"""Precomputed analytics served by the API for the dashboard (cancellations and model performance).

Both are built offline from the real data and experiment outputs by `scripts/export_analytics.py`, so the
API only reads small JSON files and never recomputes anything per request.
"""

from typing import Any

import numpy as np
import pandas as pd

from basketmind.data import cancellation_labels as cl
from basketmind.evaluation import rigour

GROUPS = {"gru", "abl_graph_customer", "abl_full"}  # deep learning models; the rest are reference methods
SELECTED = "abl_graph_customer"  # chosen on validation among the BasketMind variants
# Added after the course report (docs/repeat_purchases.md); not in rigour.NAMES, which older reports use.
EXTRA_NAMES = {"personal_frequency": "Personal frequency"}
EXTRA_PAIRS = [  # (A, B): A minus B
    ("abl_graph_customer", "personal_frequency"),
    ("item_item_cf", "personal_frequency"),
    ("gru", "personal_frequency"),
]
NAMES = rigour.NAMES | EXTRA_NAMES


def cancellation_payload(
    purchases: pd.DataFrame, cancellations: pd.DataFrame, product_regex: str, descriptions: dict[str, str]
) -> dict[str, Any]:
    """Label feasibility (data/cancellation_labels.py), monthly activity and the most cancelled products."""
    p, c = cl.prepare(purchases, cancellations, product_regex)
    out = cl.analyse(purchases, cancellations, product_regex)
    month = lambda s: s.dt.strftime("%Y-%m")  # noqa: E731
    sales, canc = p.groupby(month(p["InvoiceDate"])).size(), c.groupby(month(c["InvoiceDate"])).size()
    out["monthly"] = [
        {
            "month": m,
            "purchase_lines": int(sales.get(m, 0)),
            "cancellation_lines": int(canc.get(m, 0)),
            "cancellation_rate": float(canc.get(m, 0) / max(sales.get(m, 0) + canc.get(m, 0), 1)),
        }
        for m in sorted(set(sales.index) | set(canc.index))
    ]
    per_product = c.groupby("StockCode").size().sort_values(ascending=False).head(10)
    bought = p.groupby("StockCode").size()
    out["top_cancelled"] = [
        {
            "product_id": s,
            "description": descriptions.get(s, ""),
            "cancellation_lines": int(n),
            "share_of_lines": float(n / (bought.get(s, 0) + n)),
        }
        for s, n in per_product.items()
    ]
    out["horizons"] = {str(h): v for h, v in out["horizons"].items()}  # JSON keys must be strings
    return out


def _ci(row: pd.Series) -> dict[str, float]:
    return {"mean": float(row["mean"]), "lo": float(row["lo"]), "hi": float(row["hi"])}


def performance_payload(
    rows: pd.DataFrame,
    slices: pd.DataFrame,
    validation: dict[str, dict[str, float]],
    popularity: np.ndarray,
    n_items: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Test results with customer-level intervals, paired comparisons, slice results and popularity bias.

    `validation` maps a model id to {"recall@10": mean, "std": std over seeds} from the evaluation runs.
    """
    avg = rigour.prepare(rows)
    ov = {m: rigour.overall(avg, slices, "full", m, n_boot) for m in ("recall@10", "ndcg@10", "mrr")}
    ov_new = rigour.overall(avg, slices, "exclude_history", "recall@10", n_boot)
    sd = rigour.seed_std(rows, "full", "recall@10")
    bias = rigour.bias_table(rows, popularity, n_items)
    models = []
    for m in NAMES:
        if m not in ov["mrr"].index:
            continue
        models.append(
            {
                "id": m,
                "name": NAMES[m],
                "group": "deep" if m in GROUPS else "other",
                "selected": m == SELECTED,
                "test": {k: _ci(ov[k].loc[m]) for k in ov},
                "test_new_items": {"recall@10": _ci(ov_new.loc[m])},
                "validation": validation.get(m),
                "seed_std_recall10": float(sd[m]),
                "coverage": float(bias.loc[m, "coverage"]),
                "head_share": float(bias.loc[m, "head_share"]),
            }
        )
    paired = []
    for policy, metrics in (("full", ["recall@10", "ndcg@10", "mrr"]), ("exclude_history", ["recall@10"])):
        for metric in metrics:
            for _, r in rigour.paired(
                avg, slices, policy, metric, n_boot, rigour.PAIRS + EXTRA_PAIRS
            ).iterrows():
                paired.append(
                    {
                        "a": r["a"],
                        "b": r["b"],
                        "metric": metric,
                        "new_items": policy != "full",
                        "mean_diff": float(r["mean_diff"]),
                        "lo": float(r["lo"]),
                        "hi": float(r["hi"]),
                        "p": float(r["p"]),
                    }
                )
    slice_out: dict[str, list[dict[str, Any]]] = {}
    for column in ("history_bucket", "cold_start"):
        t = rigour.by_slice(avg, slices, column, "full", "recall@10", min(n_boot, 500))
        slice_out[column] = [
            {
                "slice": str(r["slice"]),
                "model": r["model"],
                "mean": float(r["mean"]),
                "half": float(r["half"]),
                "n": int(r["n"]),
            }
            for _, r in t.iterrows()
        ]
    first = ov["mrr"].iloc[0]
    return {
        "models": models,
        "paired": paired,
        "slices": slice_out,
        "meta": {
            "n_test_examples": int(first["n"]),
            "n_test_customers": int(first["n_customers"]),
            "split": "test",
        },
    }
