import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.notices import historical_banner, require_api
from dashboard.utils import api_client
from dashboard.utils.theme import BLUE, style_fig

st.title("Executive overview")
historical_banner()
require_api()
o = api_client.get("/analytics/overview")

c = st.columns(4)
c[0].metric("Customers", f"{o['customers']:,}")
c[1].metric("Baskets", f"{o['baskets']:,}")
c[2].metric("Products", f"{o['products']:,}")
c[3].metric("Total sales", f"£{o['total_value']:,.0f}")
st.caption(f"Data from {o['first_basket'][:10]} to {o['last_basket'][:10]}. Source: UCI Online Retail II.")

monthly = pd.DataFrame(o["monthly"])
st.subheader("Baskets per month")
fig = px.bar(monthly, x="month", y="baskets", color_discrete_sequence=[BLUE])
st.plotly_chart(style_fig(fig), width="stretch")
with st.expander("Table view"):
    st.dataframe(monthly, hide_index=True, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Most-bought products")
    top = pd.DataFrame(o["top_products"])
    fig = px.bar(top.iloc[::-1], x="baskets", y="description", orientation="h", color_discrete_sequence=[BLUE])
    st.plotly_chart(style_fig(fig, height=380), width="stretch")
    with st.expander("Table view"):
        st.dataframe(top, hide_index=True, width="stretch")
with right:
    st.subheader("Baskets by country")
    ctry = pd.DataFrame(o["top_countries"])
    fig = px.bar(ctry.iloc[::-1], x="baskets", y="country", orientation="h", color_discrete_sequence=[BLUE])
    st.plotly_chart(style_fig(fig, height=380), width="stretch")
    with st.expander("Table view"):
        st.dataframe(ctry, hide_index=True, width="stretch")
