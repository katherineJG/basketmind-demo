"""Raw ingestion and validation. The raw file is only read, never modified."""

from pathlib import Path
from typing import Any

import pandas as pd

from basketmind.utils.logging import get_logger

log = get_logger("data.ingest")
_TEXT = {"Invoice": str, "StockCode": str, "Description": str, "Country": str}


def load_raw_data(path: str | Path, config: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    """Load every sheet of the source workbook into one frame with canonical column names.

    Only dtype normalisation happens here: no row is dropped. `source_sheet` records the origin.
    Reading the xlsx takes ~45 s, so a parquet copy is cached at config['raw_cache_path'].
    """
    cache = Path(config["raw_cache_path"])
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    log.info("reading %s (slow)", path)
    sheets = pd.read_excel(path, sheet_name=None, dtype=_TEXT)
    frames = [df.assign(source_sheet=name) for name, df in sheets.items()]
    df = pd.concat(frames, ignore_index=True).rename(columns=config["column_map"])
    # CustomerID arrives as float (13085.0); keep it as a string id, with <NA> for missing.
    df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")
    df["Description"] = df["Description"].astype("string")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def validate_raw_data(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Check schema and types (raises ValueError on hard failures) and return a quality summary."""
    missing = [c for c in config["required_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        raise ValueError("InvoiceDate is not a datetime column")
    for col in ("Quantity", "UnitPrice"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} is not numeric")
    if df["InvoiceNo"].isna().any() or df["InvoiceDate"].isna().any():
        raise ValueError("InvoiceNo and InvoiceDate must not contain nulls")
    return {
        "rows": len(df),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
        "null_counts": {c: int(df[c].isna().sum()) for c in config["required_columns"]},
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "non_positive_price_rows": int((df["UnitPrice"] <= 0).sum()),
    }
