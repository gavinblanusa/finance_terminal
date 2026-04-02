"""Unit tests for macro mover vol / z helpers (Approach B)."""

from __future__ import annotations

import pandas as pd

from macro_context import (
    _daily_pct_returns,
    realized_vol_20d_from_closes,
)


def test_daily_pct_returns_empty_and_simple() -> None:
    assert _daily_pct_returns([]) == []
    assert _daily_pct_returns([100.0]) == []
    assert _daily_pct_returns([100.0, 101.0]) == [1.0]


def test_realized_vol_insufficient_points() -> None:
    s = pd.Series([100.0 + i * 0.1 for i in range(10)])
    assert realized_vol_20d_from_closes(s) is None


def test_realized_vol_constant_returns_zero_stdev_handled() -> None:
    # 21 identical closes -> 20 zero returns -> stdev 0 -> function returns None
    s = pd.Series([100.0] * 21)
    assert realized_vol_20d_from_closes(s) is None


def test_realized_vol_nonzero() -> None:
    # 21 closes with varying daily % moves so sample stdev > 0
    vals = [100.0]
    mults = [1.01, 0.99, 1.02, 0.98, 1.005] * 5  # 25 steps, use first 20
    for m in mults[:20]:
        vals.append(vals[-1] * m)
    s = pd.Series(vals)
    v = realized_vol_20d_from_closes(s)
    assert v is not None
    assert v > 0
