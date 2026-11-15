"""Write docs/cancellation_labels.md: how well can cancellations be turned into reversal labels?"""

import pandas as pd

from basketmind.data.cancellation_labels import analyse, render
from basketmind.utils.config import load_config, project_root

if __name__ == "__main__":
    root = project_root()
    proc = root / "data" / "processed"
    regex = load_config("data")["product_code_regex"]
    cols = ["CustomerID", "StockCode", "Quantity", "InvoiceDate", "InvoiceNo", "has_customer"]
    result = analyse(
        pd.read_parquet(proc / "purchase_events.parquet", columns=cols),
        pd.read_parquet(proc / "cancellation_events.parquet", columns=cols),
        regex,
    )
    (root / "docs" / "cancellation_labels.md").write_text(render(result, regex), encoding="utf-8")
    print("wrote docs/cancellation_labels.md")
