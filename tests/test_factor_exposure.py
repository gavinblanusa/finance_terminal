"""Unit tests for Fama–French factor exposure and attribution (synthetic data)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from factor_exposure import (
    ATTR_MIN_DAYS,
    FACTOR_COLS,
    MIN_OVERLAP_DAYS,
    build_factor_attribution,
    resolve_attribution_window,
    _regress_one_ticker,
)


def _synthetic_ff_calendar(
    start: date,
    n_days: int,
    *,
    factor_daily: float = 0.001,
    rf_daily: float = 0.0001,
) -> pd.DataFrame:
    """Simple business-day-like calendar (Mon–Fri) with constant factor returns."""
    rows = []
    d = start
    while len(rows) < n_days:
        if d.weekday() < 5:
            row = {"RF": rf_daily}
            for c in FACTOR_COLS:
                row[c] = factor_daily
            rows.append((pd.Timestamp(d), row))
        d += timedelta(days=1)
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame([r[1] for r in rows], index=idx)


def _synthetic_ff_calendar_varying(start: date, n_days: int, seed: int = 42) -> pd.DataFrame:
    """Business-day calendar with uncorrelated factor draws (well-conditioned OLS)."""
    rng = np.random.default_rng(seed)
    rows = []
    d = start
    while len(rows) < n_days:
        if d.weekday() < 5:
            row = {"RF": float(rng.uniform(0.00005, 0.0002))}
            for c in FACTOR_COLS:
                row[c] = float(rng.normal(0, 0.012))
            rows.append((pd.Timestamp(d), row))
        d += timedelta(days=1)
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame([r[1] for r in rows], index=idx)


def test_resolve_attribution_window_last_n_and_mtd(monkeypatch: pytest.MonkeyPatch) -> None:
    ff = _synthetic_ff_calendar(date(2020, 1, 1), 80)
    a0, a1, w = resolve_attribution_window(ff, "21")
    assert a0 is not None and a1 is not None
    assert (a1 - a0).days >= 0

    a0m, a1m, _ = resolve_attribution_window(ff, "mtd")
    assert a0m <= a1m


def test_resolve_custom_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    ff = _synthetic_ff_calendar(date(2024, 6, 1), 40)
    first = ff.index.min().date()
    last = ff.index.max().date()
    a0, a1, w = resolve_attribution_window(
        ff,
        "custom",
        first - timedelta(days=400),
        last + timedelta(days=5),
    )
    assert a0 == first and a1 == last
    assert w  # clamp warning


def test_regress_one_ticker_recoveries() -> None:
    """Known betas on synthetic aligned panel."""
    n = MIN_OVERLAP_DAYS + 20
    ff = _synthetic_ff_calendar_varying(date(2019, 1, 1), n + 50)
    ff = ff.iloc[:n]
    # stock excess = 1.0 * Mkt-RF + 0.5 * SMB + RF (close-to-close space)
    stock_r = pd.Series(
        ff["Mkt-RF"].values + 0.5 * ff["SMB"].values + ff["RF"].values,
        index=ff.index,
    )
    betas, r2, n_obs = _regress_one_ticker(stock_r, ff)
    assert betas is not None and r2 is not None
    assert n_obs >= MIN_OVERLAP_DAYS
    assert betas["Mkt-RF"] == pytest.approx(1.0, abs=0.05)
    assert betas["SMB"] == pytest.approx(0.5, abs=0.05)


def test_build_factor_attribution_residual_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Single-ticker portfolio: if returns match factor model exactly, cumulative residual ~ 0.
    """
    # Long calendar: estimation window then attribution window
    full = _synthetic_ff_calendar_varying(date(2018, 1, 1), MIN_OVERLAP_DAYS + ATTR_MIN_DAYS + 80)
    # Attribution: last ATTR_MIN_DAYS + 5 French rows
    attr_slice = full.iloc[-(ATTR_MIN_DAYS + 5) :]
    attr_start = attr_slice.index.min().date()
    attr_end = attr_slice.index.max().date()

    # Ticker return = Mkt-RF + RF (alpha 0, beta 1 on Mkt-RF, 0 on other factors)
    r_series = pd.Series(full["Mkt-RF"].values + full["RF"].values, index=full.index)

    def _fake_fetch(_ticker: str, _years: int) -> pd.DataFrame:
        close = (1.0 + r_series).cumprod()
        return pd.DataFrame({"Close": close})

    monkeypatch.setattr(
        "factor_exposure.load_ff5_factors",
        lambda force_refresh=False: (full, []),
    )

    positions = [{"ticker": "TST", "current_value": 1.0}]
    out = build_factor_attribution(positions, _fake_fetch, attr_start, attr_end, period_years=3)
    assert out.available
    assert out.n_attribution_days >= ATTR_MIN_DAYS
    # Explained tracks realized for this DGP → residual sum near zero
    assert abs(out.cumulative_residual) < 5e-4
    assert out.portfolio_factor_betas.get("Mkt-RF", 0.0) == pytest.approx(1.0, abs=0.15)


def test_build_factor_attribution_no_positions() -> None:
    def _noop(_t: str, _y: int) -> Optional[pd.DataFrame]:
        return None

    out = build_factor_attribution([], _noop, date(2020, 1, 1), date(2020, 2, 1))
    assert not out.available
