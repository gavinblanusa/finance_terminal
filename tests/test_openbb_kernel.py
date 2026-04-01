"""Unit tests for openbb_fetch kernel and adapter integration (mocked, no network)."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pathlib import Path

from openbb_adapter import fetch_quote_openbb
from openbb_fetch import OpenBBFetchResult, run_provider_chain
from openbb_provider_registry import OPENBB_PROVIDER_CHAINS, chain_arrow


@pytest.fixture(autouse=True)
def _reset_obb_singleton() -> None:
    import openbb_fetch

    prev = openbb_fetch._obb
    openbb_fetch._obb = None
    yield
    openbb_fetch._obb = prev


def test_run_provider_chain_skip_disabled() -> None:
    with patch("openbb_fetch.USE_OPENBB", False):
        res = run_provider_chain("equity.test", "AAPL", ("yfinance",), lambda o, p: o)
    assert res.ok is False
    assert res.data is None
    assert res.error_kind == "skip_disabled"


def test_run_provider_chain_no_obb_instance() -> None:
    with patch("openbb_fetch._get_obb", return_value=None):
        res = run_provider_chain("equity.test", "AAPL", ("yfinance",), lambda o, p: o)
    assert res.ok is False
    assert res.error_kind == "skip_disabled"


def test_run_provider_chain_all_providers_empty() -> None:
    mock_obb = object()

    def invoke(_obb: object, _provider: str) -> MagicMock:
        r = MagicMock()
        r.results = None
        return r

    with patch("openbb_fetch._get_obb", return_value=mock_obb):
        res = run_provider_chain("equity.test", "AAPL", ("a", "b"), invoke)
    assert res.ok is False
    assert res.error_kind in ("empty", "no_provider")


def test_run_provider_chain_first_provider_ok() -> None:
    mock_obb = object()
    df = pd.DataFrame({"close": [100.0]})
    raw = MagicMock()
    raw.results = True
    raw.to_df = MagicMock(return_value=df)

    def invoke(_obb: object, provider: str) -> MagicMock:
        return raw

    with patch("openbb_fetch._get_obb", return_value=mock_obb):
        res = run_provider_chain("equity.test", "AAPL", ("yfinance",), invoke)
    assert res.ok is True
    assert res.provider_used == "yfinance"
    assert res.data is raw
    assert res.error_kind is None
    assert res.elapsed_ms >= 0


def test_run_provider_chain_timeout_advances() -> None:
    mock_obb = object()

    def slow(_obb: object, _provider: str) -> None:
        time.sleep(0.3)
        r = MagicMock()
        r.results = True
        return r

    with patch("openbb_fetch._get_obb", return_value=mock_obb), patch(
        "openbb_fetch.OPENBB_REQUEST_TIMEOUT_SEC", 0.05
    ):
        res = run_provider_chain("equity.test", "AAPL", ("slow",), slow)
    assert res.ok is False
    assert res.error_kind == "timeout"


def test_run_provider_chain_exception_then_ok() -> None:
    mock_obb = object()
    df = pd.DataFrame({"last_price": [42.5]})
    good = MagicMock()
    good.results = True
    good.to_df = MagicMock(return_value=df)
    calls: list[str] = []

    def invoke(_obb: object, provider: str) -> MagicMock:
        calls.append(provider)
        if provider == "bad":
            raise RuntimeError("provider down")
        return good

    with patch("openbb_fetch._get_obb", return_value=mock_obb):
        res = run_provider_chain("equity.quote", "ZZZ", ("bad", "good"), invoke)
    assert res.ok is True
    assert res.provider_used == "good"
    assert calls == ["bad", "good"]


def test_fetch_quote_openbb_uses_decimal_from_kernel() -> None:
    raw = MagicMock()
    raw.results = True
    raw.to_df = MagicMock(return_value=pd.DataFrame({"last_price": [123.456]}))
    ok = OpenBBFetchResult(True, raw, "yfinance", 12, None)
    with patch("openbb_adapter.run_provider_chain", return_value=ok):
        out = fetch_quote_openbb("TEST")
    assert out == Decimal("123.46")


def test_fetch_quote_openbb_none_when_chain_fails() -> None:
    fail = OpenBBFetchResult(False, None, None, 0, "no_provider")
    with patch("openbb_adapter.run_provider_chain", return_value=fail):
        assert fetch_quote_openbb("TEST") is None


def test_openbb_coverage_doc_matches_multihop_registry() -> None:
    """Keep docs/OPENBB_COVERAGE.md aligned with openbb_provider_registry (same check as CI script)."""
    root = Path(__file__).resolve().parent.parent
    text = (root / "docs" / "OPENBB_COVERAGE.md").read_text(encoding="utf-8")
    for dataset_id, providers in OPENBB_PROVIDER_CHAINS.items():
        if len(providers) < 2:
            continue
        assert chain_arrow(providers) in text, f"missing chain for {dataset_id}"


def test_openbb_package_importable() -> None:
    """CI smoke: pinned openbb install imports (major upgrades should not break silently)."""
    pytest.importorskip("openbb")
    from openbb import obb

    assert obb is not None


def test_macro_fred_ob_result_to_df_normalizes_columns() -> None:
    import openbb_adapter as oa

    raw = MagicMock()
    raw.to_df = MagicMock(
        return_value=pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [1.0, 2.0]}
        )
    )
    df = oa._macro_fred_ob_result_to_df(raw, "DGS10")
    assert df is not None
    assert list(df.columns) == ["value"]
    assert len(df) == 2


def test_macro_skips_openbb_chain_without_fred_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import openbb_adapter as oa

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("USE_OPENBB", "true")
    stub = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2020-01-01"], name="date"))
    with patch.object(oa, "run_provider_chain") as m_chain, patch.object(
        oa, "_fetch_macro_via_pandas_datareader", return_value=stub
    ) as m_dr:
        out = oa.fetch_macro_data_openbb("cpi")
    m_chain.assert_not_called()
    m_dr.assert_called_once_with("cpi")
    assert out is not None


def test_macro_uses_openbb_when_fred_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import openbb_adapter as oa

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setenv("USE_OPENBB", "true")
    raw = MagicMock()
    raw.to_df = MagicMock(
        return_value=pd.DataFrame({"date": pd.to_datetime(["2021-06-01"]), "value": [4.5]})
    )
    ok = OpenBBFetchResult(True, raw, "fred", 8, None)
    with patch.object(oa, "run_provider_chain", return_value=ok) as m_chain, patch.object(
        oa, "_fetch_macro_via_pandas_datareader"
    ) as m_dr:
        out = oa.fetch_macro_data_openbb("cpi")
    m_chain.assert_called_once()
    m_dr.assert_not_called()
    assert out is not None
    assert float(out["value"].iloc[0]) == 4.5
