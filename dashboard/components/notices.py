"""Banners and guards that every page uses: what kind of numbers these are, and what to do if the API is down."""

import streamlit as st

from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError


def historical_banner() -> None:
    st.info("**Historical data.** This page describes what customers actually did. It contains no model predictions.")


def prediction_banner() -> None:
    st.warning(
        "**Model predictions.** These come from a research prototype. Scores rank products for a customer; "
        "they are not probabilities, and nothing here has been tested on real business outcomes."
    )


def require_api() -> dict:
    """Return the API health, or show a clear message and stop the page if the API or model is unavailable."""
    try:
        health = api_client._get(api_client.base_url(), "/health", ())
    except ApiError as e:
        st.error(
            f"The BasketMind API is not reachable: {e.detail}. "
            "Start it with `python -m uvicorn basketmind_api.main:app`, then reload this page."
        )
        st.stop()
    if not health.get("model_loaded"):
        st.error(f"The API is running but its model is not loaded: {health.get('detail', 'unknown reason')}.")
        st.stop()
    return health


def show_error(e: ApiError, what: str = "that") -> None:
    """A friendly message for a failed call: 404 means 'not found', anything else is shown plainly."""
    if e.status == 404:
        st.warning(f"Could not find {what}: {e.detail}")
    else:
        st.error(f"The API returned an error ({e.status}): {e.detail}")
