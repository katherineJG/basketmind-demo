import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.notices import require_api, show_error
from dashboard.utils import api_client
from dashboard.utils.api_client import ApiError
from dashboard.utils.theme import model_color, style_fig

st.title("Model performance")
st.info(
    "**Test-set results from finished experiments**, with 95% confidence intervals resampled by customer. "
    "BasketMind did not beat the best baselines; the numbers below show by how much."
)
require_api()
try:
    perf = api_client.get("/models/performance")
    meta = api_client.get("/models/metadata")["metadata"]
except ApiError as e:
    show_error(e, "the performance results")
    st.stop()
st.caption(
    f"{perf['meta']['n_test_examples']:,} test baskets from {perf['meta']['n_test_customers']:,} customers."
)
with st.expander("Model being served"):
    st.json(meta)

names = {m["id"]: m["name"] for m in perf["models"]}
metric = st.radio("Metric", ["recall@10", "ndcg@10", "mrr"], horizontal=True)
new_items = st.toggle("Only products the customer had not bought before", value=False)
key = "test_new_items" if new_items else "test"
rows = sorted((m for m in perf["models"] if metric in m[key]), key=lambda m: m[key][metric]["mean"])
if not rows:
    st.warning("This metric is not available for the selected setting.")
    st.stop()
val = [m[key][metric] for m in rows]
fig = go.Figure(
    go.Bar(
        x=[v["mean"] for v in val],
        y=[m["name"] for m in rows],
        orientation="h",
        marker_color=[model_color(m["id"]) for m in rows],
        error_x={
            "type": "data",
            "symmetric": False,
            "array": [v["hi"] - v["mean"] for v in val],
            "arrayminus": [v["mean"] - v["lo"] for v in val],
        },
        text=[f"{v['mean']:.3f}" for v in val],
        textposition="outside",
    )
)
st.plotly_chart(style_fig(fig, height=360), width="stretch")
if "personal_frequency" in names:
    st.caption(
        "Personal frequency ranks a customer's own past products by how often and how recently they bought them. "
        "It needs no learning, and it was added after the course report. Most of the next basket is repeat purchases, "
        "which is why it is hard to beat. It cannot suggest anything the customer has not bought before, so in the "
        "new-items view it falls back to popularity and scores exactly the same."
    )
with st.expander("Table view"):
    table = pd.DataFrame(
        {
            "model": [m["name"] for m in rows],
            "mean": [v["mean"] for v in val],
            "lower": [v["lo"] for v in val],
            "upper": [v["hi"] for v in val],
        }
    ).iloc[::-1]
    st.dataframe(table, hide_index=True, width="stretch")

st.subheader("Paired comparisons")
st.caption(
    "A negative difference means the first model scored lower. An interval that excludes zero is a reliable gap."
)
paired = pd.DataFrame(perf["paired"])
if not paired.empty:
    paired = paired[(paired["metric"] == metric) & (paired["new_items"] == new_items)]
if paired.empty:
    st.write("No paired comparisons for this selection.")
else:
    paired = paired.assign(a=paired["a"].map(names), b=paired["b"].map(names))
    st.dataframe(paired[["a", "b", "mean_diff", "lo", "hi", "p"]], hide_index=True, width="stretch")

st.subheader("Recall@10 by customer history length")
cells = pd.DataFrame(perf["slices"]["history_bucket"])
order = list(dict.fromkeys(cells["slice"]))
fig = go.Figure()
for mid in ("item_item_cf", "abl_graph_customer", "gru"):
    part = cells[cells["model"] == mid].set_index("slice").reindex(order).reset_index()
    if part["mean"].notna().any():
        fig.add_bar(
            x=part["slice"],
            y=part["mean"],
            name=names.get(mid, mid),
            marker_color=model_color(mid),
            error_y={"type": "data", "array": part["half"]},
        )
fig.update_layout(barmode="group")
fig.update_xaxes(title="Baskets before the target basket")
st.plotly_chart(style_fig(fig, legend=True), width="stretch")
with st.expander("Table view"):
    st.dataframe(cells.assign(model=cells["model"].map(names)), hide_index=True, width="stretch")
