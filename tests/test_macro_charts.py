"""Smoke tests for macro time-series figure helper (no network)."""

import pandas as pd
import numpy as np

from macro_indicators import macro_metric_by_id
from macro_charts import PLOTLY_MACRO_CONFIG, build_macro_line_figure


def test_plotly_config_has_hover_modebar() -> None:
    assert PLOTLY_MACRO_CONFIG.get("displayModeBar") in ("hover", False)


def test_build_macro_line_figure_log_y_runs() -> None:
    spec = macro_metric_by_id("init_claims")
    assert spec is not None
    assert spec.y_log
    idx = pd.date_range("2015-01-01", periods=20, freq="ME")
    vals = np.full(20, 2e5) + np.linspace(0, 1e3, 20)
    df = pd.DataFrame({"value": vals}, index=idx)
    fig = build_macro_line_figure(df, spec)
    assert fig.to_dict()["layout"]["yaxis"].get("type") == "log"


def test_build_macro_line_figure_linear_runs() -> None:
    spec = macro_metric_by_id("unemployment")
    assert spec is not None
    assert not spec.y_log
    idx = pd.date_range("2010-01-01", periods=20, freq="ME")
    df = pd.DataFrame({"value": np.linspace(3.0, 5.0, 20)}, index=idx)
    fig = build_macro_line_figure(df, spec)
    d = fig.to_dict()
    yax = d.get("layout", {}).get("yaxis", {}) or {}
    assert yax.get("type", "linear") in (None, "linear", "category", "-")
