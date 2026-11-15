import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.notices import historical_banner, require_api, show_error
from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError
from dashboard.utils.theme import BLUE, style_fig

st.title("Product intelligence")
historical_banner()
require_api()

top = api_client.get("/analytics/overview")["top_products"]
choices = {t["product_id"]: f"{t['product_id']}  {t['description']}" for t in top}
picked = st.selectbox("Pick a popular product", list(choices), format_func=choices.get)
typed = st.text_input("...or type a product code (for example 85123A)").strip()
pid = typed or picked
try:
    p = api_client.get(f"/products/{pid}")
    related = api_client.get(f"/products/{pid}/related", k=10)
except ApiError as e:
    show_error(e, f"product {pid}")
    st.stop()

st.subheader(p["description"])
c = st.columns(3)
c[0].metric("Baskets containing it", f"{p['baskets_containing']:,}")
c[1].metric("Last sold", p["last_sold"][:10])
c[2].metric("Linked products", p["n_related"])
if not p["in_catalogue"]:
    st.info("This product is not in the model's training catalogue, so it has no co-purchase links.")
if not related:
    st.warning("No related products found.")
    st.stop()
st.subheader("Often bought together")
st.caption("Association is PPMI: how much more often two products share a basket than chance predicts. From history.")
frame = pd.DataFrame(related)
fig = px.bar(frame.iloc[::-1], x="association", y="description", orientation="h", color_discrete_sequence=[BLUE])
st.plotly_chart(style_fig(fig, height=380), width="stretch")
with st.expander("Table view"):
    st.dataframe(frame, hide_index=True, width="stretch")
