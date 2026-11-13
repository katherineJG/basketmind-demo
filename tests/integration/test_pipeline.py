import shutil

import pandas as pd
from tests.conftest import make_raw

from basketmind.data.pipeline import run_m1
from basketmind.utils.config import project_root


def test_pipeline_end_to_end_on_synthetic_xlsx(tmp_path):
    """Writes a tiny SYNTHETIC two-sheet xlsx in the UCI layout and runs the full pipeline on it."""
    root = tmp_path
    shutil.copytree(project_root() / "configs", root / "configs")
    (root / "docs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    raw = make_raw().rename(
        columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}
    )
    raw["Customer ID"] = pd.to_numeric(raw["Customer ID"], errors="coerce")
    with pd.ExcelWriter(root / "data/raw/online_retail_II.xlsx") as xw:
        for name, g in raw.groupby("source_sheet"):
            g.drop(columns="source_sheet").to_excel(xw, sheet_name=name, index=False)

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
    assert list(pd.read_excel(root / "data/raw/online_retail_II.xlsx", sheet_name=None)) == [
        "s1",
        "s2",
    ]
