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


@pytest.fixture
def demo(service_parts, tmp_path, monkeypatch):
    """Demo mode on a bundle built from the synthetic service: the dashboard calls the API in-process."""
    import streamlit as st

    from basketmind.inference.demo_export import write_bundle

    live, _, _ = service_parts
    live.analytics = {}
    write_bundle(live, tmp_path, k=2)
    monkeypatch.setattr(api_client, "DEMO_DIR", tmp_path)
    monkeypatch.setenv("BASKETMIND_MODE", "demo")
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


@pytest.mark.parametrize("page", ["overview", "customers", "predictions", "products"])
def test_demo_mode_pages_run_on_precomputed_results_without_a_server(demo, page):
    at = run(page)
    assert not at.exception and not at.error and at.title


def test_demo_mode_says_what_it_is_and_locks_the_api_address(demo):
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Precomputed demo" in i.value for i in at.info)
    assert not at.sidebar.text_input  # a visitor cannot point the server at another address
    assert any("CC BY 4.0" in m.value for m in at.sidebar.markdown)


def test_the_live_dashboard_keeps_the_api_address_box(monkeypatch, tmp_path):
    monkeypatch.delenv("BASKETMIND_MODE", raising=False)
    monkeypatch.setattr(api_client, "DEMO_DIR", tmp_path)  # no PUBLIC_DEMO marker here
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert [t.label for t in at.sidebar.text_input] == ["API address"]
    assert not any("Precomputed demo" in i.value for i in at.info)


def test_demo_mode_without_its_data_fails_politely_and_shows_no_path(tmp_path, monkeypatch):
    import streamlit as st

    monkeypatch.setattr(api_client, "DEMO_DIR", tmp_path / "missing")
    monkeypatch.setenv("BASKETMIND_MODE", "demo")
    st.cache_resource.clear()
    at = run("overview")
    assert not at.exception
    messages = " ".join(e.value for e in at.error)
    assert "demo data is missing" in messages or "not reachable" in messages
    assert str(tmp_path) not in messages


def test_a_public_demo_marker_file_switches_demo_mode_on(tmp_path, monkeypatch):
    monkeypatch.delenv("BASKETMIND_MODE", raising=False)
    monkeypatch.setattr(api_client, "DEMO_DIR", tmp_path)
    assert not api_client.demo_mode()
    (tmp_path / "PUBLIC_DEMO").write_text("")
    assert api_client.demo_mode()


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
