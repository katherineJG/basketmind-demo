"""Can cancellations be turned into trustworthy 'this purchase was reversed' labels?

The source data has no link from a cancellation line to the purchase it reverses, so any label has to be inferred.
This module measures how well a simple inference works, so that a reversal head is only trained if the labels
can bear it.

Linking rule: a cancellation line (customer, product, time) is attributed to the same customer's MOST RECENT
EARLIER purchase line of that product. Label rule under test: a purchase line is "reversed within H days" if the
same customer cancels that product within H days after it.
"""

import numpy as np
import pandas as pd

HORIZONS = (30, 90)


def prepare(
    purchases: pd.DataFrame, cancellations: pd.DataFrame, product_regex: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep product lines that have a customer; give every purchase line an id and the cumulative quantity bought."""
    p = purchases[purchases["has_customer"] & purchases["StockCode"].str.match(product_regex)].copy()
    c = cancellations[
        cancellations["has_customer"] & cancellations["StockCode"].str.match(product_regex)
    ].copy()
    p = p.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    p["pid"] = np.arange(len(p))
    p["cum_qty"] = p.groupby(["CustomerID", "StockCode"])["Quantity"].cumsum()
    c["cancelled"] = c["Quantity"].abs()
    return p, c.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)


def link_cancellations(p: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """For each cancellation: the most recent earlier purchase of the same customer and product (if any)."""
    cols = ["CustomerID", "StockCode", "InvoiceDate", "pid", "cum_qty"]
    linked = pd.merge_asof(
        c,
        p[cols].rename(columns={"InvoiceDate": "bought_at"}).assign(InvoiceDate=lambda d: d["bought_at"]),
        on="InvoiceDate",
        by=["CustomerID", "StockCode"],
        direction="backward",
        allow_exact_matches=False,
    )
    linked["has_prior"] = linked["pid"].notna()
    linked["qty_covered"] = linked["has_prior"] & (linked["cum_qty"] >= linked["cancelled"])
    linked["lag_days"] = (linked["InvoiceDate"] - linked["bought_at"]).dt.total_seconds() / 86400
    return linked


def label_purchases(p: pd.DataFrame, c: pd.DataFrame, linked: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Window label per purchase line, and whether the cancellation it matched is really attributed to that line."""
    nxt = pd.merge_asof(
        p[["pid", "CustomerID", "StockCode", "InvoiceDate"]],
        c[["CustomerID", "StockCode", "InvoiceDate"]]
        .rename(columns={"InvoiceDate": "cancelled_at"})
        .assign(InvoiceDate=lambda d: d["cancelled_at"]),
        on="InvoiceDate",
        by=["CustomerID", "StockCode"],
        direction="forward",
        allow_exact_matches=False,
    )
    nxt["reversed"] = (nxt["cancelled_at"] - nxt["InvoiceDate"]).dt.total_seconds() / 86400 <= horizon
    origin = set(linked.loc[linked["lag_days"] <= horizon, "pid"].dropna().astype(int))
    nxt["attributed"] = nxt["reversed"] & nxt["pid"].isin(origin)
    return nxt


def analyse(purchases: pd.DataFrame, cancellations: pd.DataFrame, product_regex: str) -> dict:
    p, c = prepare(purchases, cancellations, product_regex)
    linked = link_cancellations(p, c)
    data_end = p["InvoiceDate"].max()
    first_months = linked["InvoiceDate"] < linked["InvoiceDate"].min() + pd.DateOffset(months=3)
    out = {
        "purchase_lines": len(p),
        "cancellation_lines": len(c),
        "cancellation_share_of_lines": len(c) / (len(p) + len(c)),
        "linked": {
            "has_prior_purchase": float(linked["has_prior"].mean()),
            "quantity_covered": float(linked["qty_covered"].mean()),
            "no_prior_in_first_3_months": float((~linked.loc[first_months, "has_prior"]).mean()),
            "no_prior_after_3_months": float((~linked.loc[~first_months, "has_prior"]).mean()),
            "lag_days_median": float(linked["lag_days"].median()),
            "lag_days_p90": float(linked["lag_days"].quantile(0.9)),
            "within_30_days": float((linked["lag_days"] <= 30).mean()),
            "within_90_days": float((linked["lag_days"] <= 90).mean()),
        },
        "horizons": {},
    }
    for h in HORIZONS:
        lab = label_purchases(p, c, linked, h)
        complete = lab["InvoiceDate"] + pd.Timedelta(days=h) <= data_end  # the full window is observable
        lab = lab[complete]
        out["horizons"][h] = {
            "purchase_lines_with_complete_window": len(lab),
            "base_rate": float(lab["reversed"].mean()),
            "attribution_precision": float(lab["attributed"].sum() / max(lab["reversed"].sum(), 1)),
        }
    return out


def render(r: dict, product_regex: str) -> str:
    L, H = r["linked"], r["horizons"]
    rows = "\n".join(
        f"| {h} days | {v['purchase_lines_with_complete_window']:,} | {v['base_rate']:.2%} | {v['attribution_precision']:.0%} |"
        for h, v in H.items()
    )
    return f"""# Cancellation labels: can they support a reversal head?

Generated by `scripts/analyse_cancellations.py`; do not edit by hand.

The data has no link from a cancellation line to the purchase it reverses, so a label has to be inferred. This report
measures how well a simple rule works on product lines (stock codes matching `{product_regex}`) that have a customer id.

**Rule.** A cancellation line is attributed to the same customer's most recent earlier purchase line of the same product.
A purchase line is labelled *reversed within H days* if the same customer cancels that product within H days after it.

## Can cancellations be linked to a purchase at all?

{r["cancellation_lines"]:,} cancellation lines against {r["purchase_lines"]:,} purchase lines ({r["cancellation_share_of_lines"]:.1%} of all product lines).

| Check | Share of cancellation lines |
|---|---:|
| Customer bought that product earlier (in this data) | {L["has_prior_purchase"]:.1%} |
| ...and earlier quantity bought covers the quantity cancelled | {L["quantity_covered"]:.1%} |
| No earlier purchase, cancelled in the first 3 months of the data | {L["no_prior_in_first_3_months"]:.1%} |
| No earlier purchase, cancelled after the first 3 months | {L["no_prior_after_3_months"]:.1%} |
| Cancelled within 30 days of that earlier purchase | {L["within_30_days"]:.1%} |
| Cancelled within 90 days of that earlier purchase | {L["within_90_days"]:.1%} |

Median gap between purchase and cancellation: {L["lag_days_median"]:.1f} days; 90th percentile: {L["lag_days_p90"]:.0f} days.
Cancelled quantity is compared with the cumulative quantity bought, ignoring earlier returns, so coverage is an upper bound.

## What would the window label look like?

Only purchase lines whose whole window lies inside the data are counted. *Attribution precision* is the share of
labelled purchase lines that really are the most recent earlier purchase for a cancellation (the rest are earlier
purchases of the same product that the window label also flags, so they are probably wrong).

| Window | Purchase lines | Base rate (reversed) | Attribution precision |
|---|---:|---:|---:|
{rows}

## Leakage constraint

A purchase's label depends on events up to H days after it. For training and validation the label must be computed
only from cancellations that happened before the split boundary, and purchases whose window crosses the boundary must
be dropped or labelled from what was known at the boundary. Otherwise the label leaks information across splits.
"""
