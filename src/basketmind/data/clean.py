"""Cleaning: label every row with an event type. Nothing is silently discarded."""

from typing import Any

import pandas as pd

EVENT_TYPES = [
    "purchase",
    "cancellation",
    "adjustment",
    "non_product",
    "invalid_price",
    "duplicate_overlap",
]


def clean_transactions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add `event_type` to every row (first matching rule wins) and normalise StockCode.

    Rules, in priority order:
      duplicate_overlap  invoice already present in an earlier sheet (the sheets overlap in Dec 2010)
      cancellation       InvoiceNo starts with the cancellation prefix ("C")
      adjustment         any other non-positive quantity (stock write-offs, "A" bad-debt invoices)
      non_product        StockCode is not a product code (POST, DOT, M, BANK CHARGES, gift_*, ...)
      invalid_price      positive quantity but price <= 0
      purchase           everything else
    Rows with a missing CustomerID stay in their event type (flag `has_customer`); they are never
    merged into a fictitious customer.
    """
    out = df.copy()
    out["StockCode"] = out["StockCode"].str.strip().str.upper()
    is_prod = out["StockCode"].str.match(config["product_code_regex"])

    # Invoices that already appear in an earlier sheet: later copies are duplicates.
    sheet_order = {s: i for i, s in enumerate(out["source_sheet"].drop_duplicates())}
    rank = out["source_sheet"].map(sheet_order)
    first_rank = rank.groupby(out["InvoiceNo"]).transform("min")

    event = pd.Series("purchase", index=out.index, dtype="object")
    event[(out["Quantity"] > 0) & (out["UnitPrice"] <= 0)] = "invalid_price"
    event[~is_prod] = "non_product"
    event[out["Quantity"] <= 0] = "adjustment"
    event[out["InvoiceNo"].str.startswith(config["cancellation_prefix"])] = "cancellation"
    event[rank > first_rank] = "duplicate_overlap"

    out["event_type"] = event
    out["has_customer"] = out["CustomerID"].notna()
    return out
