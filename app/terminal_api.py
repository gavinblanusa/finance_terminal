"""
Read-only HTTP API for macro, FI proxies, portfolio snapshot, and dashboard analytics.

Run from project root with app on PYTHONPATH:

    PYTHONPATH=app uvicorn terminal_api:app --host 127.0.0.1 --port 8800

Optional auth: set ``GFT_API_KEY`` in ``.env`` and send ``Authorization: Bearer <key>``.
CORS: set ``GFT_CORS_ORIGINS`` to ``*`` or comma-separated origins (e.g. ``http://localhost:3000``).
Rate limit: ``GFT_RATE_LIMIT`` defaults to ``60/minute``; set to ``0`` or ``off`` to disable (very high cap).

Interactive docs: http://127.0.0.1:8800/docs
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from analytics_export import build_rest_dashboard_payload
from data_schemas import iv_term_structure_to_schema, macro_context_to_schema, tca_to_schema
from fi_context import build_fi_context_strip, fi_rows_to_records
from macro_context import build_macro_context
from portfolio_snapshot import fetch_portfolio_snapshot_dict


def _gft_api_key() -> str:
    """Read at request time so tests (and process managers) can toggle auth without reloading the app."""
    return os.getenv("GFT_API_KEY", "").strip()


def _rate_limit_string() -> str:
    s = os.getenv("GFT_RATE_LIMIT", "60/minute").strip()
    if s.lower() in ("0", "off", "none", ""):
        return "100000/minute"
    return s


RLIMIT = _rate_limit_string()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Gavin Financial Terminal API",
    description="Read-only access to dashboard data builders. Personal use; do not expose without TLS and auth.",
    version="0.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _setup_cors(application: FastAPI) -> None:
    raw = os.getenv("GFT_CORS_ORIGINS", "").strip()
    if not raw:
        return
    if raw == "*":
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return
    origins = [x.strip() for x in raw.split(",") if x.strip()]
    if not origins:
        return
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_setup_cors(app)


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    key = _gft_api_key()
    if not key:
        return
    if authorization is None or authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


DepKey = Depends(require_api_key)


class HealthResponse(BaseModel):
    status: str = "ok"
    auth_configured: bool = Field(description="True when GFT_API_KEY is set")
    cors_configured: bool = Field(description="True when GFT_CORS_ORIGINS is set")
    rate_limit: str = Field(description="Effective slowapi limit string")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        auth_configured=bool(_gft_api_key()),
        cors_configured=bool(os.getenv("GFT_CORS_ORIGINS", "").strip()),
        rate_limit=RLIMIT,
    )


@app.get("/v1/macro", dependencies=[DepKey], tags=["macro"])
@limiter.limit(RLIMIT)
def get_macro(request: Request) -> Dict[str, Any]:
    try:
        ctx = build_macro_context()
        return macro_context_to_schema(ctx).model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/fi", dependencies=[DepKey], tags=["macro"])
@limiter.limit(RLIMIT)
def get_fi_strip(request: Request) -> List[Dict[str, Any]]:
    try:
        rows, _errs = build_fi_context_strip()
        return fi_rows_to_records(rows)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/portfolio", dependencies=[DepKey], tags=["portfolio"])
@limiter.limit(RLIMIT)
def get_portfolio(request: Request) -> Dict[str, Any]:
    try:
        snap = fetch_portfolio_snapshot_dict()
        if snap is None:
            raise HTTPException(status_code=503, detail="portfolio_unavailable")
        return snap
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/analytics/dashboard", dependencies=[DepKey], tags=["analytics"])
@limiter.limit(RLIMIT)
def get_dashboard_analytics(
    request: Request,
    include_factors: bool = True,
) -> Dict[str, Any]:
    """
    Macro + PORT-lite + optional Fama–French factors (same JSON shape as dashboard export,
    without per-session TCA).
    """
    try:
        return build_rest_dashboard_payload(include_factors=include_factors)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/options/iv-term", dependencies=[DepKey], tags=["options"])
@limiter.limit(RLIMIT)
def get_iv_term(
    request: Request,
    ticker: str,
    spot: Optional[float] = None,
    max_expirations: int = 12,
) -> Dict[str, Any]:
    """ATM implied vol term structure (Yahoo chains)."""
    from options_iv_term import build_iv_term_structure

    sym = (ticker or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="ticker required")
    so = float(spot) if spot is not None and spot > 0 else None
    try:
        res = build_iv_term_structure(sym, max_expirations=max_expirations, spot_override=so)
        return iv_term_structure_to_schema(res).model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


class TCARequest(BaseModel):
    ticker: str
    shares: float = Field(gt=0, description="Order size in shares")
    side: str = Field(default="buy", description="buy or sell")


@app.post("/v1/analytics/tca", dependencies=[DepKey], tags=["analytics"])
@limiter.limit(RLIMIT)
def post_tca(request: Request, body: TCARequest) -> Dict[str, Any]:
    """Illustrative pre-trade impact (same model as dashboard TCA)."""
    from market_data import fetch_ohlcv
    from tca_estimate import estimate_trade_impact

    t = body.ticker.upper().strip()
    side = (body.side or "buy").strip().lower()
    if side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="side must be buy or sell")
    r = estimate_trade_impact(t, float(body.shares), side, fetch_ohlcv, period_years=2)
    if r is None:
        raise HTTPException(status_code=400, detail="invalid ticker or shares")
    return tca_to_schema(r).model_dump(mode="json")


class BlackScholesRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    time_years: float = Field(ge=0, le=50)
    rate: float = Field(ge=0, le=0.5, description="Annual risk-free, decimal (e.g. 0.04)")
    volatility: float = Field(gt=0, le=5.0, description="Annual vol, decimal (e.g. 0.25)")
    dividend_yield: float = Field(default=0.0, ge=0, le=0.5)


@app.post("/v1/options/black-scholes", dependencies=[DepKey], tags=["options"])
@limiter.limit(RLIMIT)
def post_black_scholes(request: Request, body: BlackScholesRequest) -> Dict[str, Any]:
    """European Black–Scholes call/put (theory prices)."""
    from options_black_scholes import black_scholes_european

    out = black_scholes_european(
        body.spot,
        body.strike,
        body.time_years,
        body.rate,
        body.volatility,
        body.dividend_yield,
    )
    return {
        "call_price": out.call_price,
        "put_price": out.put_price,
        "d1": out.d1,
        "d2": out.d2,
        "inputs": {
            "spot": out.spot,
            "strike": out.strike,
            "time_years": out.time_years,
            "rate": out.rate,
            "volatility": out.volatility,
            "dividend_yield": out.dividend_yield,
        },
    }
