"""Unit tests for macro Indicators: Sahm series and health bands (no network)."""

import numpy as np
import pandas as pd

from macro_indicators import (
    compute_sahm_series,
    evaluate_macro_metric,
    macro_metric_by_id,
    RULE_SET_VERSION,
)


def test_rule_set_version_is_string() -> None:
    assert len(RULE_SET_VERSION) >= 1


def test_sahm_series_shape() -> None:
    idx = pd.date_range("2018-01-01", periods=20, freq="MS")
    u = 3.8 * np.ones(20)
    u[17:20] = [4.0, 4.4, 5.0]  # rising unemployment
    dfn = pd.DataFrame({"value": u}, index=idx)
    sa = compute_sahm_series(dfn)
    assert sa is not None
    assert sa["value"].notna().any()


def test_sahm_evaluate_uses_metric_id() -> None:
    dfn = pd.DataFrame(
        {
            "value": [0.0, 0.1, 0.2, 0.45, 0.3, 0.55],
        },
        index=pd.date_range("2020-01-01", periods=6, freq="MS"),
    )
    s, v, p, a = evaluate_macro_metric(dfn, "sahm")
    assert s in ("Healthy", "Teetering", "Unhealthy", "No Data")
    assert "pp" in v
    # latest 0.55 is Stress band (>=0.5)
    assert s == "Unhealthy"


def test_cpi_band() -> None:
    idx = pd.date_range("2019-01-01", periods=24, freq="MS")
    a = 200.0
    b = 205.0  # ~2.5% over 1y if aligned
    vals = np.linspace(a, b, 24)
    df = pd.DataFrame({"value": vals}, index=idx)
    st, v, p, a0 = evaluate_macro_metric(df, "cpi")
    assert st in ("Healthy", "Teetering", "Unhealthy")
    assert a0 is not None


def test_registry_has_sahm_and_nfci() -> None:
    assert macro_metric_by_id("sahm") is not None
    assert macro_metric_by_id("nfci") is not None


def test_init_claims_bands_use_persons_not_thousands() -> None:
    """FRED IC4WSA is in persons; bands match ~280k/350k thresholds."""
    idx = pd.date_range("2024-01-01", periods=3, freq="W")
    for latest, want in [
        (250_000.0, "Healthy"),
        (300_000.0, "Teetering"),
        (400_000.0, "Unhealthy"),
    ]:
        df = pd.DataFrame({"value": [200_000.0, 220_000.0, latest]}, index=idx)
        st, v, _p, _a = evaluate_macro_metric(df, "init_claims")
        assert st == want
        assert "k" in v
