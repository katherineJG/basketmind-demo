"""Basket aggregation and chronological splitting."""

from typing import Any

import pandas as pd


def aggregate_baskets(events: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice from purchase events that have a CustomerID.

    Repeated stock codes inside an invoice are summed. The basket time is the earliest line time
    (64 invoices span several minutes). Anonymous invoices are excluded here (they cannot be placed in
    a customer sequence) and counted in the quality report.
    """
    p = events[(events["event_type"] == "purchase") & events["CustomerID"].notna()].copy()
    p["line_value"] = p["Quantity"] * p["UnitPrice"]
    lines = p.groupby(["InvoiceNo", "StockCode"], as_index=False).agg(
        Quantity=("Quantity", "sum"), line_value=("line_value", "sum")
    )
    lines = lines.sort_values(["InvoiceNo", "StockCode"])
    items = lines.groupby("InvoiceNo").agg(
        items=("StockCode", list),
        quantities=("Quantity", list),
        n_items=("StockCode", "size"),
        total_quantity=("Quantity", "sum"),
        total_value=("line_value", "sum"),
    )
    head = p.groupby("InvoiceNo").agg(
        CustomerID=("CustomerID", "first"),
        timestamp=("InvoiceDate", "min"),
        Country=("Country", lambda s: s.mode().iat[0]),
    )
    out = head.join(items).reset_index()
    return out.sort_values(["CustomerID", "timestamp", "InvoiceNo"]).reset_index(drop=True)


def create_temporal_splits(
    baskets: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign each basket to train/validation/test by time. Never random.

    Returns (baskets with a `split` column, metadata). Baskets outside every window get
    split='unassigned' and are reported, not dropped.
    """
    cfg = config["splits"]
    bounds = {
        k: (pd.Timestamp(v["start"]), pd.Timestamp(v["end"]) + pd.Timedelta(days=1)) for k, v in cfg.items()
    }
    ordered = sorted(bounds.items(), key=lambda kv: kv[1][0])
    for (_, (_, end_a)), (name_b, (start_b, _)) in zip(ordered, ordered[1:], strict=False):
        if end_a > start_b:
            raise ValueError(f"split windows overlap at {name_b}")
    out = baskets.copy()
    out["split"] = "unassigned"
    for name, (start, end) in bounds.items():
        out.loc[(out["timestamp"] >= start) & (out["timestamp"] < end), "split"] = name
    meta = {
        "boundaries": {k: dict(v) for k, v in cfg.items()},
        "baskets": out["split"].value_counts().to_dict(),
        "customers": out.groupby("split")["CustomerID"].nunique().to_dict(),
        "observed": {
            k: [str(g["timestamp"].min()), str(g["timestamp"].max())] for k, g in out.groupby("split")
        },
    }
    return out, meta
