import pandas as pd

from basketmind.data.pipeline import run_m1


def test_pipeline_end_to_end_on_synthetic_xlsx(synthetic_root):
    root = synthetic_root
    report = run_m1(root=root)

    out = root / "data/processed"
    for f in [
        "purchase_events",
        "cancellation_events",
        "other_events",
        "baskets",
        "customer_sequences",
        "next_basket_examples",
    ]:
        assert (out / f"{f}.parquet").exists(), f
    assert (out / "split_metadata.json").exists() and (root / "docs/data_quality_report.md").exists()
    assert report["event_types"]["cancellation"]["rows"] == 1
    assert report["temporal_integrity"]["examples_checked"] == report["examples"]["total"]
    # raw workbook untouched and still two sheets
    assert list(pd.read_excel(root / "data/raw/online_retail_II.xlsx", sheet_name=None)) == ["s1", "s2"]
