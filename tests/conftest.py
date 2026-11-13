"""Tiny SYNTHETIC fixtures for tests only. Never used for reported results."""

import pandas as pd
import pytest

from basketmind.utils.config import load_config


@pytest.fixture
def data_cfg():
    return load_config("data")


def make_raw() -> pd.DataFrame:
    """Raw-shaped rows (canonical column names) covering every event type."""
    rows = [
        # invoice, stock, desc, qty, date, price, customer, country, sheet
        ("100", "85123a", "A", 2, "2010-03-01 10:00", 2.0, "1", "UK", "s1"),  # lower-case code
        (
            "100",
            "85123A",
            "A",
            3,
            "2010-03-01 10:00",
            2.0,
            "1",
            "UK",
            "s1",
        ),  # repeated code, summed
        ("100", "22000", "B", 1, "2010-03-01 10:01", 5.0, "1", "UK", "s1"),
        (
            "101",
            "22000",
            "B",
            1,
            "2010-03-01 10:00",
            5.0,
            "1",
            "UK",
            "s1",
        ),  # tie with 100's timestamp
        ("102", "22000", "B", 1, "2010-04-01 09:00", 5.0, "1", "UK", "s1"),
        ("103", "85123A", "A", 4, "2011-02-01 09:00", 2.0, "1", "UK", "s1"),
        ("104", "22000", "B", 1, "2011-08-01 09:00", 5.0, "1", "UK", "s1"),
        ("105", "22000", "B", 1, "2010-05-01 09:00", 5.0, "2", "DE", "s1"),
        ("106", "22000", "B", 1, "2010-05-02 09:00", 5.0, None, "UK", "s1"),  # anonymous
        ("C107", "22000", "B", -1, "2010-05-03 09:00", 5.0, "2", "DE", "s1"),  # cancellation
        ("108", "POST", "postage", 1, "2010-05-03 09:00", 18.0, "2", "DE", "s1"),  # non-product
        ("109", "22000", "B", -5, "2010-05-04 09:00", 0.0, None, "UK", "s1"),  # adjustment
        ("110", "22000", "B", 2, "2010-05-05 09:00", 0.0, "2", "DE", "s1"),  # invalid price
        ("105", "22000", "B", 1, "2010-05-01 09:00", 5.0, "2", "DE", "s2"),  # overlap copy of 105
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
            "source_sheet",
        ],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("string")
    return df
