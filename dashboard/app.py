"""BasketMind dashboard entry point: `streamlit run dashboard/app.py`. Pages live in dashboard/pages/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets pages import `dashboard.*`

import streamlit as st  # noqa: E402

from dashboard.utils import api_client  # noqa: E402
from dashboard.utils.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="BasketMind", layout="wide")
apply_theme()
with st.sidebar:
    st.text_input("API address", value=api_client.DEFAULT_URL, key="api_url")

pages = [
    st.Page("pages/overview.py", title="Executive overview", default=True),
    st.Page("pages/customers.py", title="Customer intelligence"),
    st.Page("pages/predictions.py", title="Next basket predictions"),
    st.Page("pages/products.py", title="Product intelligence"),
    st.Page("pages/cancellations.py", title="Cancellation intelligence"),
    st.Page("pages/performance.py", title="Model performance"),
]
st.navigation(pages).run()
