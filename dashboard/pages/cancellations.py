import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.notices import historical_banner, require_api, show_error
from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError
from dashboard.utils.theme import BLUE, style_fig

st.title("Cancellation intelligence")
historical_banner()
require_api()
try:
    a = api_client.get("/analytics/cancellations")
except ApiError as e:
    show_error(e, "the cancellation analysis")
    st.stop()

c = st.columns(3)
c[0].metric("Purchase lines", f"{a['purchase_lines']:,}")
c[1].metric("Cancellation lines", f"{a['cancellation_lines']:,}")
c[2].metric("Cancellations as share of lines", f"{a['cancellation_share_of_lines']:.1%}")

st.subheader("Cancellation rate by month")
monthly = pd.DataFrame(a["monthly"])
fig = px.line(monthly, x="month", y="cancellation_rate", markers=True, color_discrete_sequence=[BLUE])
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(style_fig(fig), width="stretch")
with st.expander("Table view"):
    st.dataframe(monthly, hide_index=True, width="stretch")

st.subheader("Most-cancelled products")
top = pd.DataFrame(a["top_cancelled"])
fig = px.bar(
    top.iloc[::-1], x="cancellation_lines", y="description", orientation="h", color_discrete_sequence=[BLUE]
)
st.plotly_chart(style_fig(fig, height=380), width="stretch")
with st.expander("Table view"):
    st.dataframe(top, hide_index=True, width="stretch")

st.subheader("Could a reversal model be trained?")
st.write(
    "A cancellation row says a product came back, not which purchase it undoes. These are the labels you get by "
    "assuming a purchase is reversed if the same customer cancels the same product within a window."
)
horizons = pd.DataFrame([{"window (days)": k, **v} for k, v in a["horizons"].items()]).rename(
    columns={
        "purchase_lines_with_complete_window": "purchase lines with a full window",
        "base_rate": "share labelled reversed",
        "attribution_precision": "attribution precision",
    }
)
st.dataframe(horizons, hide_index=True, width="stretch")
st.caption("No reversal model is served. See docs/cancellation_labels.md for the full feasibility analysis.")
