"""BasketMind dashboard entry point: `streamlit run dashboard/app.py`. Pages live in dashboard/pages/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets pages import `dashboard.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # the demo imports the API in-process
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import streamlit as st  # noqa: E402

from dashboard.utils import api_client  # noqa: E402
from dashboard.utils.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="BasketMind", layout="wide")
apply_theme()
DEMO = api_client.demo_mode()
with st.sidebar:
    if DEMO:
        st.markdown(
            "**About.** BasketMind is a portfolio project: a hybrid deep-learning next-basket recommender, "
            "compared honestly with simpler baselines.\n\n"
            "**Data.** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), "
            "CC BY 4.0. Customer ids are the dataset's own anonymous numbers."
        )
    else:  # a visitor to the public demo must not be able to point the server at another address
        st.text_input("API address", value=api_client.DEFAULT_URL, key="api_url")
if DEMO:
    st.info(
        "**Precomputed demo.** The model's recommendations for every customer were computed ahead of time and "
        "are served from files, so this app runs without the model. Everything else is computed from the data. "
        "It is a research prototype for demonstration, not a production system."
    )

pages = [
    st.Page("pages/overview.py", title="Executive overview", default=True),
    st.Page("pages/customers.py", title="Customer intelligence"),
    st.Page("pages/predictions.py", title="Next basket predictions"),
    st.Page("pages/products.py", title="Product intelligence"),
    st.Page("pages/cancellations.py", title="Cancellation intelligence"),
    st.Page("pages/performance.py", title="Model performance"),
]
st.navigation(pages).run()
