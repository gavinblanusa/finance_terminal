"""
HTTP contract tests for ``terminal_api`` (FastAPI).

Uses mocks for network/DB-backed builders so the suite stays deterministic.
Follows REST semantics: correct status codes (401 / 422 / 400 / 503), OpenAPI
surface, and JSON shapes for happy paths.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from macro_context import MacroContextResult
from options_iv_term import IVTermPoint, IVTermStructureResult
from tca_estimate import TCAEstimateResult
from terminal_api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def open_gft_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Bearer auth is not required (isolated from developer shell env)."""
    monkeypatch.delenv("GFT_API_KEY", raising=False)


@pytest.fixture
def gft_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = "test-api-key-for-pytest"
    monkeypatch.setenv("GFT_API_KEY", key)
    return key


def _auth_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_health_open_ok(client: TestClient, open_gft_auth: None) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["auth_configured"] is False
    assert "rate_limit" in body


def test_health_reflects_auth_env(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GFT_API_KEY", "x")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_configured"] is True


def test_openapi_lists_v1_paths(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert "/v1/macro" in paths
    assert "/v1/portfolio" in paths
    assert "/v1/analytics/dashboard" in paths
    assert "/v1/options/iv-term" in paths
    assert "/v1/analytics/tca" in paths
    assert "/v1/options/black-scholes" in paths


def test_v1_routes_require_bearer_when_key_set(client: TestClient, gft_key: str) -> None:
    r = client.get("/v1/macro")
    assert r.status_code == 401
    assert r.json()["detail"] == "Missing or invalid API key"

    r = client.get("/v1/macro", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


@patch("terminal_api.build_macro_context")
def test_get_macro_200(mock_macro, client: TestClient, gft_key: str) -> None:
    mock_macro.return_value = MacroContextResult(
        movers=[], rates=[], fred_configured=False, errors=[]
    )
    r = client.get("/v1/macro", headers=_auth_headers(gft_key))
    assert r.status_code == 200
    data = r.json()
    assert data["fred_configured"] is False
    assert data["movers"] == []
    assert data["rates"] == []


@patch("terminal_api.build_macro_context")
def test_get_macro_503_on_builder_error(mock_macro, client: TestClient, gft_key: str) -> None:
    mock_macro.side_effect = RuntimeError("upstream")
    r = client.get("/v1/macro", headers=_auth_headers(gft_key))
    assert r.status_code == 503
    assert "upstream" in r.json()["detail"]


@patch("terminal_api.build_fi_context_strip")
def test_get_fi_200(mock_fi, client: TestClient, gft_key: str) -> None:
    mock_fi.return_value = ([], [])
    r = client.get("/v1/fi", headers=_auth_headers(gft_key))
    assert r.status_code == 200
    assert r.json() == []


@patch("terminal_api.fetch_portfolio_snapshot_dict")
def test_get_portfolio_503_when_unavailable(mock_snap, client: TestClient, gft_key: str) -> None:
    mock_snap.return_value = None
    r = client.get("/v1/portfolio", headers=_auth_headers(gft_key))
    assert r.status_code == 503
    assert r.json()["detail"] == "portfolio_unavailable"


@patch("terminal_api.fetch_portfolio_snapshot_dict")
def test_get_portfolio_200(mock_snap, client: TestClient, gft_key: str) -> None:
    mock_snap.return_value = {"positions": [], "as_of": "2026-01-01"}
    r = client.get("/v1/portfolio", headers=_auth_headers(gft_key))
    assert r.status_code == 200
    assert r.json()["positions"] == []


@patch("terminal_api.build_rest_dashboard_payload")
def test_get_dashboard_analytics_200(mock_dash, client: TestClient, gft_key: str) -> None:
    mock_dash.return_value = {"macro": {}, "snapshot": True}
    r = client.get(
        "/v1/analytics/dashboard",
        params={"include_factors": "false"},
        headers=_auth_headers(gft_key),
    )
    assert r.status_code == 200
    assert r.json() == {"macro": {}, "snapshot": True}
    mock_dash.assert_called_once_with(include_factors=False)


@patch("options_iv_term.build_iv_term_structure")
def test_get_iv_term_200(mock_iv, client: TestClient, gft_key: str) -> None:
    mock_iv.return_value = IVTermStructureResult(
        ticker="AAPL",
        spot_used=200.0,
        points=[
            IVTermPoint(
                expiry="2026-06-20",
                dte=80,
                iv_atm=0.25,
                strike=200.0,
                source="avg",
            )
        ],
        data_warnings=[],
    )
    r = client.get(
        "/v1/options/iv-term",
        params={"ticker": "aapl", "spot": 200},
        headers=_auth_headers(gft_key),
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ticker"] == "AAPL"
    assert j["spot_used"] == 200.0
    assert len(j["points"]) == 1


def test_get_iv_term_400_empty_ticker(client: TestClient, gft_key: str) -> None:
    r = client.get(
        "/v1/options/iv-term",
        params={"ticker": "   "},
        headers=_auth_headers(gft_key),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "ticker required"


@patch("tca_estimate.estimate_trade_impact")
def test_post_tca_200(mock_est, client: TestClient, gft_key: str) -> None:
    mock_est.return_value = TCAEstimateResult(
        ticker="AAPL",
        side="buy",
        shares=100.0,
        notional_usd=15_000.0,
        adv_shares=5_000_000.0,
        adv_dollar=750_000_000.0,
        participation_rate=0.00002,
        daily_volatility=0.018,
        annualized_volatility=0.29,
        estimated_impact_frac=0.0001,
        estimated_impact_bps=1.0,
        estimated_impact_usd=15.0,
        price_ref=150.0,
        data_warnings=[],
    )
    r = client.post(
        "/v1/analytics/tca",
        headers=_auth_headers(gft_key),
        json={"ticker": "AAPL", "shares": 100, "side": "buy"},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ticker"] == "AAPL"
    assert j["side"] == "buy"
    assert j["shares"] == 100.0


def test_post_tca_422_invalid_body(client: TestClient, gft_key: str) -> None:
    r = client.post(
        "/v1/analytics/tca",
        headers=_auth_headers(gft_key),
        json={"ticker": "AAPL", "shares": -1, "side": "buy"},
    )
    assert r.status_code == 422


@patch("tca_estimate.estimate_trade_impact")
def test_post_tca_400_bad_side(mock_est, client: TestClient, gft_key: str) -> None:
    r = client.post(
        "/v1/analytics/tca",
        headers=_auth_headers(gft_key),
        json={"ticker": "AAPL", "shares": 100, "side": "hold"},
    )
    assert r.status_code == 400
    assert "buy or sell" in r.json()["detail"]


@patch("tca_estimate.estimate_trade_impact")
def test_post_tca_400_no_estimate(mock_est, client: TestClient, gft_key: str) -> None:
    mock_est.return_value = None
    r = client.post(
        "/v1/analytics/tca",
        headers=_auth_headers(gft_key),
        json={"ticker": "AAPL", "shares": 100, "side": "sell"},
    )
    assert r.status_code == 400


def test_post_black_scholes_200(client: TestClient, gft_key: str) -> None:
    r = client.post(
        "/v1/options/black-scholes",
        headers=_auth_headers(gft_key),
        json={
            "spot": 100.0,
            "strike": 100.0,
            "time_years": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
            "dividend_yield": 0.0,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["call_price"] > 0
    assert j["put_price"] >= 0
    assert "d1" in j and "d2" in j
    assert j["inputs"]["spot"] == 100.0


def test_post_black_scholes_expiry_intrinsic(client: TestClient, gft_key: str) -> None:
    r = client.post(
        "/v1/options/black-scholes",
        headers=_auth_headers(gft_key),
        json={
            "spot": 100.0,
            "strike": 95.0,
            "time_years": 0.0,
            "rate": 0.05,
            "volatility": 0.25,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["call_price"] == pytest.approx(5.0, rel=1e-6)
    assert j["put_price"] == pytest.approx(0.0, abs=1e-9)


def test_post_black_scholes_422_invalid_spot(client: TestClient, gft_key: str) -> None:
    r = client.post(
        "/v1/options/black-scholes",
        headers=_auth_headers(gft_key),
        json={
            "spot": 0.0,
            "strike": 100.0,
            "time_years": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
        },
    )
    assert r.status_code == 422


def test_macro_without_auth_when_key_unset(client: TestClient, open_gft_auth: None) -> None:
    """Optional auth: unset key allows /v1/* without Authorization (local dev only)."""
    with patch("terminal_api.build_macro_context") as mock_macro:
        mock_macro.return_value = MacroContextResult(
            movers=[], rates=[], fred_configured=False, errors=[]
        )
        r = client.get("/v1/macro")
        assert r.status_code == 200
