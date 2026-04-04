"""Tests for optional MDW OHLCV adapter (app/market_warehouse.py)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from market_warehouse import (
    ohlcv_df_sufficient_for_request,
    try_load_ohlcv_from_warehouse,
)


def test_try_load_no_env(monkeypatch):
    monkeypatch.delenv("GFT_MARKET_WAREHOUSE_BRONZE", raising=False)
    monkeypatch.delenv("GFT_MARKET_WAREHOUSE_DUCKDB", raising=False)
    end = date.today()
    start = end - timedelta(days=400)
    assert try_load_ohlcv_from_warehouse("AAPL", start, end) is None


def test_ohlcv_sufficient_rejects_short():
    idx = pd.bdate_range(end=date.today(), periods=100)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
        index=idx,
    )
    start = (date.today() - timedelta(days=500)).replace(day=1)
    end = date.today()
    assert not ohlcv_df_sufficient_for_request(df, start, end, 2)


def test_ohlcv_sufficient_accepts_long_panel():
    idx = pd.bdate_range(end=date.today(), periods=600)
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=idx,
    )
    end = date.today()
    start_req = end - timedelta(days=2 * 365)
    assert ohlcv_df_sufficient_for_request(df, start_req, end, 2)


def test_bronze_parquet_roundtrip(tmp_path, monkeypatch):
    root = tmp_path / "data-lake" / "bronze"
    eq_dir = root / "asset_class=equity" / "symbol=ZZOT"
    eq_dir.mkdir(parents=True)
    dates = pd.bdate_range(end=date.today(), periods=600)
    raw = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol_id": [1] * len(dates),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1_000_000,
        }
    )
    raw.to_parquet(eq_dir / "data.parquet", index=False)

    monkeypatch.setenv("GFT_MARKET_WAREHOUSE_BRONZE", str(root))
    monkeypatch.delenv("GFT_MARKET_WAREHOUSE_DUCKDB", raising=False)

    end = date.today()
    start = end - timedelta(days=2 * 365)
    out = try_load_ohlcv_from_warehouse("ZZOT", start, end)
    assert out is not None
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) >= 200
    assert ohlcv_df_sufficient_for_request(out, start, end, 2)


def test_fetch_ohlcv_prefers_warehouse(monkeypatch, tmp_path):
    """When warehouse returns a sufficient panel, OpenBB is not used."""
    from market_data import fetch_ohlcv

    root = tmp_path / "bronze"
    eq_dir = root / "asset_class=equity" / "symbol=WWXWH"
    eq_dir.mkdir(parents=True)
    dates = pd.bdate_range(end=date.today(), periods=600)
    raw = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol_id": [1] * len(dates),
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.0,
            "adj_close": 50.0,
            "volume": 500_000,
        }
    )
    raw.to_parquet(eq_dir / "data.parquet", index=False)
    monkeypatch.setenv("GFT_MARKET_WAREHOUSE_BRONZE", str(root))
    monkeypatch.delenv("GFT_MARKET_WAREHOUSE_DUCKDB", raising=False)

    with patch("openbb_adapter.fetch_ohlcv_openbb") as obb:
        df = fetch_ohlcv("WWXWH", period_years=2)
    obb.assert_not_called()
    assert df is not None
    assert len(df) >= 200
    assert float(df["Close"].iloc[-1]) == 50.0
