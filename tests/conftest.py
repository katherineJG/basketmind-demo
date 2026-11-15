"""Tiny SYNTHETIC fixtures for tests only. Never used for reported results."""

import shutil

import pandas as pd
import pytest

from basketmind.utils.config import load_config, project_root


@pytest.fixture(autouse=True)
def _release_gpu_after_test():
    """Avoid the ROCm-on-Windows exit deadlock after tests that leave GPU tensors alive (see release_gpu)."""
    yield
    from basketmind.utils.device import release_gpu

    release_gpu()


@pytest.fixture(autouse=True)
def _mlflow_in_tmp(tmp_path, monkeypatch):
    """Keep MLflow's store out of the project root during tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")


@pytest.fixture
def training_data(synthetic_root):
    """SYNTHETIC training data (features, snapshots, windows) built through the real pipeline code."""
    from basketmind.data.pipeline import run_m1
    from basketmind.training.data import build_training_data

    run_m1(root=synthetic_root)
    return build_training_data(
        synthetic_root / "data/processed", load_config("features"), load_config("graph"), quantity_samples=4
    )


@pytest.fixture
def service_parts(training_data, synthetic_root):
    """(service, training_data, baskets): an inference service around a random model and a hand-made graph."""
    import pandas as pd
    import torch

    from basketmind.graph import GraphSnapshot
    from basketmind.inference import BasketMindService
    from basketmind.models import BasketMindModel
    from basketmind.utils.device import get_device
    from basketmind.utils.seed import seed_everything

    dev = get_device()
    baskets = pd.read_parquet(synthetic_root / "data/processed/baskets.parquet")
    task, store = training_data.task, training_data.store
    seed_everything(0)
    model = BasketMindModel(load_config("model"), task.n_items, store.n_customers, store.n_countries).to(dev)
    graph = GraphSnapshot(  # the two synthetic products are linked, PPMI weight 0.7
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0.7, 0.7]),
        baskets["timestamp"].max(),
        task.n_items,
        len(baskets),
    )
    desc = {s: f"product {s}" for s in task.items}
    service = BasketMindService(
        model, task, store, baskets, graph, desc, load_config("features"), dev, metadata={"variant": "test"}
    )
    return service, training_data, baskets


@pytest.fixture
def data_cfg():
    return load_config("data")


@pytest.fixture
def synthetic_root(tmp_path):
    """Temp project root with configs and a tiny SYNTHETIC two-sheet xlsx (UCI layout), not yet processed."""
    shutil.copytree(project_root() / "configs", tmp_path / "configs")
    (tmp_path / "docs").mkdir()
    (tmp_path / "data/raw").mkdir(parents=True)
    raw = make_raw().rename(
        columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}
    )
    raw["Customer ID"] = pd.to_numeric(raw["Customer ID"], errors="coerce")
    with pd.ExcelWriter(tmp_path / "data/raw/online_retail_II.xlsx") as xw:
        for name, g in raw.groupby("source_sheet"):
            g.drop(columns="source_sheet").to_excel(xw, sheet_name=name, index=False)
    return tmp_path


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
