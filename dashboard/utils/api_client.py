"""The dashboard's only way to get data: the BasketMind REST API. Failures become `ApiError`, never a stack trace.

In demo mode (the public deployment) the same API code runs inside the Streamlit process on precomputed results
(artifacts/demo, see scripts/export_demo.py), so no server, model or torch is needed. Everything still goes through
HTTP-style calls to the API app; only the transport differs.
"""

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

DEFAULT_URL = os.environ.get("BASKETMIND_API_URL", "http://127.0.0.1:8000")
DEMO_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demo"


class ApiError(Exception):
    """An API call failed. `status` is the HTTP status, or 0 if the API could not be reached."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def base_url() -> str:
    return st.session_state.get("api_url", DEFAULT_URL)


def demo_mode() -> bool:
    """True for the public deployment: set BASKETMIND_MODE=demo, or ship artifacts/demo/PUBLIC_DEMO."""
    return os.environ.get("BASKETMIND_MODE") == "demo" or (DEMO_DIR / "PUBLIC_DEMO").exists()


class _Shared:
    """One long-lived in-process client, handed out through the same `with` interface as an httpx client."""

    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *exc):
        return False


@st.cache_resource(show_spinner="Loading the demo data...")
def _embedded_client() -> _Shared:
    # imported here so the live dashboard does not need the API packages on its path
    from fastapi.testclient import TestClient

    from basketmind.inference import PrecomputedService
    from basketmind_api.main import create_app

    client = TestClient(create_app(service=PrecomputedService.load(DEMO_DIR)))
    client.__enter__()  # starts the app once; it stays up for the life of the server
    return _Shared(client)


def make_client(url: str):
    if demo_mode():
        return _embedded_client()
    return httpx.Client(base_url=url, timeout=30.0)


@st.cache_data(ttl=120, show_spinner=False)
def _get(url: str, path: str, params: tuple) -> Any:
    try:
        with make_client(url) as client:
            response = client.get(path, params=dict(params))
    except FileNotFoundError as e:  # demo mode without its data; do not show a server path to a visitor
        raise ApiError(0, "the demo data is missing from this deployment") from e
    except httpx.HTTPError as e:
        raise ApiError(0, f"cannot reach the API at {url} ({type(e).__name__})") from e
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, str(detail))
    return response.json()


def get(path: str, **params: Any) -> Any:
    """GET `path` from the API (cached for two minutes). Raises ApiError on any failure."""
    clean = tuple(sorted((k, v) for k, v in params.items() if v is not None))
    return _get(base_url(), path, clean)
