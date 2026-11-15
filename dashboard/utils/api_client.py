"""The dashboard's only way to get data: the BasketMind REST API. Failures become `ApiError`, never a stack trace."""

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_URL = os.environ.get("BASKETMIND_API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    """An API call failed. `status` is the HTTP status, or 0 if the API could not be reached."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def base_url() -> str:
    return st.session_state.get("api_url", DEFAULT_URL)


def make_client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=30.0)


@st.cache_data(ttl=120, show_spinner=False)
def _get(url: str, path: str, params: tuple) -> Any:
    try:
        with make_client(url) as client:
            response = client.get(path, params=dict(params))
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
