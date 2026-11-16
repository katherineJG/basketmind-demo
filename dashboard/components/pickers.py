"""Customer picker shared by the customer and prediction pages."""

import streamlit as st

from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError


def pick_customer(key: str) -> str | None:
    """A search box plus a list of matching customers. Returns the chosen customer id, or None."""
    query = st.text_input("Search customer ID (prefix)", key=f"{key}_q")
    try:
        page = api_client.get("/customers", limit=50, prefix=query.strip() or None)
    except ApiError as e:
        st.error(f"Could not list customers: {e.detail}")
        return None
    if not page["items"]:
        st.warning("No customer matches that search.")
        return None
    labels = {
        c["customer_id"]: f"{c['customer_id']} ({c['n_baskets']} baskets, {c['country']})"
        for c in page["items"]
    }
    st.caption(f"Showing {len(labels)} of {page['total']:,} matching customers.")
    return st.selectbox("Customer", list(labels), format_func=labels.get, key=f"{key}_pick")
