import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.notices import prediction_banner, require_api, show_error
from dashboard.components.pickers import pick_customer
from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError
from dashboard.utils.theme import BLUE, GRAY, style_fig

st.title("Next basket predictions")
prediction_banner()
require_api()

cid = pick_customer("pred")
c1, c2 = st.columns(2)
k = c1.slider("Number of products", 1, 30, 10)
new_only = c2.toggle("Only products not bought before", value=False)
if not cid:
    st.stop()
try:
    rec = api_client.get(f"/recommendations/{cid}", k=k, exclude_history=new_only)
except ApiError as e:
    show_error(e, f"customer {cid}")
    st.stop()
st.caption(f"Predictions as of {rec['as_of']}. A higher score ranks the product higher for this customer.")
if not rec["items"]:
    st.warning("The model returned no products for this customer with these settings.")
    st.stop()

frame = pd.DataFrame(rec["items"])
frame["status"] = frame["in_history"].map({True: "Bought before", False: "New to this customer"})
fig = go.Figure()
for status, colour in (("New to this customer", BLUE), ("Bought before", GRAY)):
    part = frame[frame["status"] == status]
    fig.add_bar(x=part["score"], y=part["description"], orientation="h", name=status, marker_color=colour)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, height=max(300, 28 * len(frame) + 80), legend=True), width="stretch")
st.caption(
    "Scores are raw model outputs used for ranking, not probabilities. Quantities are indicative only."
)
with st.expander("Table view", expanded=True):
    cols = ["rank", "product_id", "description", "score", "status", "times_bought", "predicted_quantity"]
    st.dataframe(frame[cols], hide_index=True, width="stretch")

st.subheader("Why this product?")
names = dict(zip(frame["product_id"], frame["description"], strict=True))
pid = st.selectbox("Product", list(names), format_func=lambda x: f"{x}  {names[x]}")
try:
    e = api_client.get(f"/recommendations/{cid}/explain", product_id=pid)
except ApiError as err:
    show_error(err, f"an explanation for {pid}")
    st.stop()
history = (
    f"This customer bought it {e['times_bought']} time(s), last on {e['last_bought']}."
    if e["bought_before"]
    else "This customer has not bought it before."
)
st.write(
    f"Ranked **#{e['model_rank']}** by the model and **#{e['popularity_rank']}** by overall popularity. {history}"
)
if e["linked_to_history"]:
    st.write("Products in this customer's history that are often bought together with it:")
    st.dataframe(pd.DataFrame(e["linked_to_history"]), hide_index=True, width="stretch")
else:
    st.write("None of this customer's past products are linked to it in the co-purchase graph.")
st.caption(e["note"])
