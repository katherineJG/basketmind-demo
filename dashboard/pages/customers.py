import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.notices import historical_banner, require_api, show_error
from dashboard.components.pickers import pick_customer
from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError
from dashboard.utils.theme import BLUE, style_fig

st.title("Customer intelligence")
historical_banner()
require_api()

cid = pick_customer("cust")
if cid:
    try:
        p = api_client.get(f"/customers/{cid}")
        history = api_client.get(f"/customers/{cid}/history", limit=50)
    except ApiError as e:
        show_error(e, f"customer {cid}")
        st.stop()
    c = st.columns(4)
    c[0].metric("Baskets", p["n_baskets"])
    c[1].metric("Total spend", f"£{p['total_spend']:,.0f}")
    c[2].metric("Average basket size", f"{p['avg_basket_size']:.1f} items")
    c[3].metric("Country", p["country"])
    st.caption(f"First basket {p['first_basket'][:10]}, last basket {p['last_basket'][:10]}.")
    if not p["seen_in_training"]:
        st.info(
            "This customer has no basket in the training period, so the model treats them as a cold start."
        )
    if not history:
        st.warning("This customer has no baskets on record.")
        st.stop()
    frame = pd.DataFrame(history)
    st.subheader("Basket value over time")
    fig = px.bar(frame, x="date", y="total_value", color_discrete_sequence=[BLUE])
    st.plotly_chart(style_fig(fig), width="stretch")
    st.subheader("Baskets")
    for b in history:
        title = f"{b['date'][:10]}  |  invoice {b['invoice']}  |  {b['n_items']} items  |  £{b['total_value']:,.2f}"
        with st.expander(title):
            st.dataframe(pd.DataFrame(b["items"]), hide_index=True, width="stretch")
