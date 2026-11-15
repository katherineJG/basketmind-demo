import pandas as pd
import pytest

from basketmind.data import cancellation_labels as cl

REGEX = r"^\d{5}[A-Z]*$"
DAY = pd.Timestamp("2011-01-01")


def lines(rows, qty_sign=1):
    df = pd.DataFrame(rows, columns=["CustomerID", "StockCode", "Quantity", "day"])
    df["InvoiceDate"] = DAY + pd.to_timedelta(df.pop("day"), unit="D")
    df["Quantity"] *= qty_sign
    df["has_customer"] = df["CustomerID"].notna()
    df["InvoiceNo"] = "x"
    return df


def scenario():
    purchases = lines(
        [("A", "11111", 5, 0), ("A", "11111", 3, 10), ("B", "22222", 2, 0), ("C", "POST", 1, 0)]
    )
    cancellations = lines(
        [("A", "11111", 4, 12), ("A", "11111", 20, 40), ("B", "33333", 1, 5), (None, "11111", 1, 3)],
        qty_sign=-1,
    )
    return purchases, cancellations


def test_prepare_drops_non_products_and_anonymous_lines_and_adds_cumulative_quantity():
    p, c = cl.prepare(*scenario(), REGEX)
    assert set(p["StockCode"]) == {"11111", "22222"}  # POST is not a product
    assert len(c) == 3  # the anonymous cancellation is gone
    assert p.loc[p["StockCode"] == "11111", "cum_qty"].tolist() == [5, 8]


def test_links_go_to_the_most_recent_earlier_purchase():
    p, c = cl.prepare(*scenario(), REGEX)
    linked = cl.link_cancellations(p, c).set_index(["CustomerID", "cancelled"])
    first = linked.loc[("A", 4)]  # day 12 -> the day-10 purchase, 2 days earlier, cumulative 8 >= 4
    assert first["has_prior"] and first["qty_covered"] and first["lag_days"] == pytest.approx(2.0)
    second = linked.loc[("A", 20)]  # day 40 -> still the day-10 purchase, but 8 < 20
    assert second["has_prior"] and not second["qty_covered"] and second["lag_days"] == pytest.approx(30.0)
    orphan = linked.loc[("B", 1)]  # B never bought product 33333
    assert not orphan["has_prior"] and not orphan["qty_covered"]


def test_window_labels_and_attribution_precision_hand_computed():
    p, c = cl.prepare(*scenario(), REGEX)
    linked = cl.link_cancellations(p, c)
    lab = cl.label_purchases(p, c, linked, 30).set_index(["CustomerID", "InvoiceDate"])
    day0, day10 = lab.loc[("A", DAY)], lab.loc[("A", DAY + pd.Timedelta(days=10))]
    assert (
        day0["reversed"] and not day0["attributed"]
    )  # flagged by the window, but the day-10 purchase is the origin
    assert day10["reversed"] and day10["attributed"]
    assert not lab.loc[("B", DAY)]["reversed"]  # B's cancelled product was a different one


def test_analyse_reports_shares_and_censors_incomplete_windows():
    purchases, cancellations = scenario()
    r = cl.analyse(purchases, cancellations, REGEX)
    assert r["purchase_lines"] == 3 and r["cancellation_lines"] == 3
    assert r["linked"]["has_prior_purchase"] == pytest.approx(2 / 3)
    assert r["linked"]["quantity_covered"] == pytest.approx(1 / 3)
    h30 = r["horizons"][30]
    # the data ends on day 10 (the last purchase), so no purchase has a full 30-day window inside the data
    assert h30["purchase_lines_with_complete_window"] == 0
    text = cl.render(r, REGEX)
    assert "Leakage constraint" in text and "Attribution precision" in text
