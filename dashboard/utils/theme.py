"""Colours and chart styling from the project design system (design.md).

Chart colours were checked with the data-viz palette validator: blue / light blue / grey keep neighbouring marks
apart under colour-vision deficiency. The same entity keeps the same colour on every chart.
"""

import plotly.graph_objects as go
import streamlit as st

BLUE, NAVY, GRAY, TINT = "#174bc4", "#0a1d88", "#898781", "#6f93e6"
LIGHT, RULE, INK2 = "#FAF9F6", "#e1e0d9", "#52514e"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
MODEL_COLORS = {"abl_graph_customer": BLUE, "gru": TINT}  # every other method is grey

_CSS = f"""
<style>
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
h1, h2, h3 {{ color: {NAVY}; font-weight: 600; }}
[data-testid="stMetric"] {{ background: {LIGHT}; border-radius: 4px; padding: 16px 24px; }}
[data-testid="stMetricLabel"] {{ color: {INK2}; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 600; }}
a {{ color: {BLUE}; text-decoration: none; }} a:hover {{ color: {NAVY}; text-decoration: underline; }}
</style>
"""


def model_color(model_id: str) -> str:
    return MODEL_COLORS.get(model_id, GRAY)


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    """White surface, hairline grid, system font, thin marks: the chart conventions of the design system."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 16, "t": 24, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": FONT, "size": 13, "color": "#000000"},
        showlegend=legend,
        legend={"orientation": "h", "y": -0.18, "x": 0},
        bargap=0.45,
    )
    fig.update_xaxes(gridcolor=RULE, linecolor=RULE, zeroline=False, ticks="")
    fig.update_yaxes(gridcolor=RULE, linecolor=RULE, zeroline=False, ticks="")
    return fig
