"""Run every dashboard page headlessly against the real API code on the synthetic fixture data."""

from pathlib import Path

import pytest
from dashboard.utils import api_client
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from basketmind_api.main import create_app

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["overview", "customers", "predictions", "products", "cancellations", "performance"]


@pytest.fixture(autouse=True)
def _fresh_cache():
    import streamlit as st

    st.cache_data.clear()


@pytest.fixture
def served(service_parts, monkeypatch):
    service, _, _ = service_parts
    service.analytics = {}  # the analytics files are exported from real data, not part of the fixture
    with TestClient(create_app(service=service)) as c:
        monkeypatch.setattr(api_client, "make_client", lambda url: c)
        yield c


def run(page: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "dashboard" / "pages" / f"{page}.py"), default_timeout=60).run()


@pytest.mark.parametrize("page", ["overview", "customers", "predictions", "products"])
def test_pages_with_data_render_without_errors(served, page):
    at = run(page)
    assert not at.exception
    assert not at.error
    assert at.title


def test_prediction_page_shows_a_ranked_table_and_the_prediction_banner(served):
    at = run("predictions")
    assert any("Model predictions" in w.value for w in at.warning)
    assert at.dataframe  # ranked list and explanation links
    assert at.subheader[0].value == "Why this product?"


def test_historical_pages_say_they_hold_no_predictions(served):
    assert any("no model predictions" in i.value for i in run("overview").info)


def test_missing_analytics_are_reported_not_crashed(served):
    for page in ("cancellations", "performance"):
        at = run(page)
        assert not at.exception
        assert any("export_analytics" in w.value for w in at.warning)


def test_unreachable_api_shows_a_message_and_stops(monkeypatch):
    class Down:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            import httpx

            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client, "make_client", lambda url: Down())
    for page in PAGES:
        at = run(page)
        assert not at.exception
        assert any("not reachable" in e.value for e in at.error)


def test_degraded_api_says_the_model_is_not_loaded(monkeypatch):
    def broken():
        raise FileNotFoundError("no checkpoint")

    with TestClient(create_app(factory=broken)) as c:
        monkeypatch.setattr(api_client, "make_client", lambda url: c)
        at = run("overview")
    assert not at.exception
    assert any("model is not loaded" in e.value for e in at.error)
