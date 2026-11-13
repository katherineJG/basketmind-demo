import pandas as pd
import pytest
from tests.conftest import make_raw

from basketmind.data.baskets import aggregate_baskets, create_temporal_splits
from basketmind.data.clean import clean_transactions
from basketmind.data.ingest import validate_raw_data
from basketmind.sequences.build import (
    build_customer_sequences,
    build_next_basket_examples,
    verify_temporal_integrity,
)


def test_validate_rejects_missing_column(data_cfg):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(make_raw().drop(columns=["UnitPrice"]), data_cfg)


def test_validate_reports_counts(data_cfg):
    rep = validate_raw_data(make_raw(), data_cfg)
    assert rep["rows"] == 14 and rep["null_counts"]["CustomerID"] == 2


def test_clean_labels_every_row_and_drops_none(data_cfg):
    raw = make_raw()
    out = clean_transactions(raw, data_cfg)
    assert len(out) == len(raw)
    by_inv = out.drop_duplicates("InvoiceNo").set_index("InvoiceNo")["event_type"]
    assert by_inv["C107"] == "cancellation"
    assert by_inv["108"] == "non_product"
    assert by_inv["109"] == "adjustment"
    assert by_inv["110"] == "invalid_price"
    assert by_inv["106"] == "purchase"  # anonymous rows remain purchases, flagged not merged
    assert not out.loc[out["InvoiceNo"] == "106", "has_customer"].any()


def test_overlap_copy_is_labelled_duplicate(data_cfg):
    out = clean_transactions(make_raw(), data_cfg)
    inv105 = out[out["InvoiceNo"] == "105"]
    assert sorted(inv105["event_type"]) == ["duplicate_overlap", "purchase"]


def test_stockcode_normalised_to_upper(data_cfg):
    out = clean_transactions(make_raw(), data_cfg)
    assert (out["StockCode"] == out["StockCode"].str.upper()).all()


def test_aggregate_sums_repeated_codes_and_skips_anonymous(data_cfg):
    b = aggregate_baskets(clean_transactions(make_raw(), data_cfg))
    assert "106" not in set(b["InvoiceNo"])
    row = b[b["InvoiceNo"] == "100"].iloc[0]
    assert row["items"] == ["22000", "85123A"] and row["quantities"] == [1, 5]
    assert row["total_value"] == pytest.approx(5.0 + 10.0) and row["n_items"] == 2
    assert row["timestamp"] == pd.Timestamp("2010-03-01 10:00")  # earliest line
    assert not b.duplicated("InvoiceNo").any()


def _baskets(data_cfg):
    b = aggregate_baskets(clean_transactions(make_raw(), data_cfg))
    return create_temporal_splits(b, data_cfg)


def test_splits_are_chronological_and_complete(data_cfg):
    b, meta = _baskets(data_cfg)
    assert (b["split"] != "unassigned").all()
    last = lambda s: b.loc[b["split"] == s, "timestamp"].max()  # noqa: E731
    first = lambda s: b.loc[b["split"] == s, "timestamp"].min()  # noqa: E731
    assert last("train") < first("validation") <= last("validation") < first("test")
    assert meta["baskets"]["train"] == 4  # 100, 101, 102 (customer 1) + 105 (customer 2)


def test_overlapping_split_windows_rejected(data_cfg):
    bad = {
        **data_cfg,
        "splits": {
            **data_cfg["splits"],
            "validation": {"start": "2010-06-01", "end": "2011-06-30"},
        },
    }
    with pytest.raises(ValueError, match="overlap"):
        create_temporal_splits(_baskets(data_cfg)[0], bad)


def test_examples_use_strictly_earlier_history_only(data_cfg):
    b, _ = _baskets(data_cfg)
    ex = build_next_basket_examples(build_customer_sequences(b), {"max_sequence_length": 50})
    by = ex.set_index("target_invoice")["history_invoices"]
    assert by["102"] == ["100", "101"]  # both strictly earlier than 2010-04-01
    assert by["103"] == ["100", "101", "102"]
    assert "100" not in by.index and "101" not in by.index  # tied first baskets: no history
    assert "104" in by.index and by["104"][-1] == "103"
    verify_temporal_integrity(ex, b)


def test_history_truncated_to_max_length(data_cfg):
    b, _ = _baskets(data_cfg)
    ex = build_next_basket_examples(build_customer_sequences(b), {"max_sequence_length": 2})
    assert ex.set_index("target_invoice")["history_invoices"]["104"] == ["102", "103"]


def test_integrity_check_catches_future_leak(data_cfg):
    b, _ = _baskets(data_cfg)
    ex = build_next_basket_examples(build_customer_sequences(b), {"max_sequence_length": 50})
    leaked = ex.copy()
    i = leaked.index[leaked["target_invoice"] == "102"][0]
    leaked.at[i, "history_invoices"] = ["103"]  # future basket
    with pytest.raises(AssertionError):
        verify_temporal_integrity(leaked, b)
    tied = ex.copy()
    j = tied.index[tied["target_invoice"] == "102"][0]
    tied.at[j, "history_invoices"] = ["102"]  # target inside its own history
    with pytest.raises(AssertionError):
        verify_temporal_integrity(tied, b)
