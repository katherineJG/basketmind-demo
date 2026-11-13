"""Customer sequences and next-basket examples, built without temporal leakage."""

from bisect import bisect_left
from typing import Any

import pandas as pd


def build_customer_sequences(baskets: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: chronologically ordered invoices, timestamps and splits."""
    ordered = baskets.sort_values(["CustomerID", "timestamp", "InvoiceNo"])
    seq = ordered.groupby("CustomerID").agg(
        invoices=("InvoiceNo", list),
        timestamps=("timestamp", list),
        splits=("split", list),
    )
    seq["n_baskets"] = seq["invoices"].str.len()
    return seq.reset_index()


def build_next_basket_examples(sequences: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """One example per basket that has history: history -> target basket.

    History = the customer's baskets with a STRICTLY earlier timestamp (ties are excluded because 178
    same-customer baskets share a timestamp), truncated to the last `max_sequence_length`. Examples are
    labelled with the split of the TARGET basket; history may come from earlier splits (it is past data).
    First baskets (no history) are not examples; they are counted as cold-start in the quality report.
    """
    max_len = config["max_sequence_length"]
    rows = []
    for cust, invoices, stamps, splits in zip(
        sequences["CustomerID"],
        sequences["invoices"],
        sequences["timestamps"],
        sequences["splits"],
        strict=True,
    ):
        for t, ts in enumerate(stamps):
            n_prior = bisect_left(stamps, ts)  # stamps are sorted: count of strictly earlier baskets
            if n_prior == 0:
                continue
            hist = invoices[max(0, n_prior - max_len) : n_prior]
            rows.append((cust, invoices[t], ts, splits[t], hist))
    return pd.DataFrame(
        rows,
        columns=["CustomerID", "target_invoice", "target_timestamp", "split", "history_invoices"],
    )


def verify_temporal_integrity(examples: pd.DataFrame, baskets: pd.DataFrame) -> dict[str, int]:
    """Assert no future information in any example. Raises AssertionError on the first violation type.

    Checks: history baskets belong to the same customer, are strictly earlier than the target, the
    target is not in its own history, and history is chronologically ordered.
    """
    info = baskets.set_index("InvoiceNo")[["CustomerID", "timestamp"]]
    cust, ts = info["CustomerID"].to_dict(), info["timestamp"].to_dict()
    for c, target, t_ts, hist in zip(
        examples["CustomerID"],
        examples["target_invoice"],
        examples["target_timestamp"],
        examples["history_invoices"],
        strict=True,
    ):
        assert target not in hist, f"target {target} in its own history"
        assert all(cust[h] == c for h in hist), f"history of {target} contains another customer"
        stamps = [ts[h] for h in hist]
        assert all(s < t_ts for s in stamps), f"future/tied basket in history of {target}"
        assert stamps == sorted(stamps), f"history of {target} not chronological"
    return {"examples_checked": len(examples)}
