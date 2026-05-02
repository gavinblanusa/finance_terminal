"""
Plotly helpers for Macro Dashboard: time-axis focus, log-y for spike-heavy series, light mode bar.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

try:
    from app.macro_indicators import MetricSpec
except ModuleNotFoundError:
    from macro_indicators import MetricSpec

# Hover-only mode bar (DESIGN.md: reduce Plotly chrome on dense pages)
PLOTLY_MACRO_CONFIG: dict[str, Any] = {
    "displayModeBar": "hover",
    "doubleClick": "reset",
}


def build_macro_line_figure(df: pd.DataFrame, spec: MetricSpec) -> go.Figure:
    """
    Line chart: default x window ~last 10y, range selector, optional log y
    (initial claims, M2) to keep recent history readable next to pandemic spikes.
    """
    fig = go.Figure()
    if df is not None and "value" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["value"], mode="lines"))
    color = spec.line_color
    fig.update_traces(line=dict(color=color, width=1.7), connectgaps=True)
    y_title = spec.yaxis_hint
    use_log = spec.y_log and (df is not None and not df.empty) and (df["value"] > 0).all()
    if use_log:
        y_title = f"{y_title} (log scale)"
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    if df is None or df.empty or len(df) < 2:
        if use_log:
            fig.update_yaxes(type="log", showexponent="none", exponentformat="none", tickformat=".3s")
        return fig

    end = df.index.max()
    if pd.isna(end):
        if use_log:
            fig.update_yaxes(type="log", showexponent="none", exponentformat="none", tickformat=".3s")
        return fig
    start_10y = end - pd.DateOffset(years=10)
    fig.update_xaxes(
        range=(start_10y, end),
        rangeslider=dict(visible=False),
        rangeselector=dict(
            activecolor="#E8A838",
            bgcolor="rgba(38,39,48,0.9)",
            font=dict(size=10, color="white"),
            bordercolor="rgba(148,163,184,0.5)",
            borderwidth=1,
            x=0, y=1, xanchor="left", yanchor="top",
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(count=20, label="20Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )
    if use_log:
        fig.update_yaxes(type="log", showexponent="none", exponentformat="none", tickformat=".3s")
    return fig
