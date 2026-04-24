"""
Gavin Financial Terminal - Main Streamlit Application.

A personal financial intelligence platform with portfolio tracking,
tax optimization, market analysis, and IPO vintage tracking.
"""

import json
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import html
from textwrap import dedent
from decimal import Decimal
from sqlalchemy.orm import Session
from db import get_db_session, init_db
from models import Trades, TradeType, Watchlist, IPO_Registry
from tax_engine import TaxEngine, import_trades_from_csv, fetch_prices_batch
from ipo_service import (
    fetch_ipo_calendar,
    get_vintage_performance,
    get_ipo_price_history,
    check_vintage_anniversaries,
    clear_ipo_cache,
    IPOEntry,
    VintagePerformance
)
from market_data import (
    fetch_ohlcv,
    calculate_signals,
    calculate_tradingview_signals,
    get_ticker_summary,
    get_current_signal,
    clear_cache,
    get_valuation_chart_data,
    get_company_profile,
    get_fundamentals_ratios,
    fetch_company_news,
    fetch_insider_transactions,
    save_tv_signals_to_cache,
    load_tv_signals_from_cache,
    check_tv_signals_in_cache,
    get_competitors,
    clear_peers_cache,
)
from edgar_service import (
    get_partnership_events,
    hydrate_partnership_market_caps,
    partnership_events_caps_deferred,
    refresh_edgar_data,
)
from plotly_chart_rescale import render_plotly_chart_with_y_rescale
from streamlit_lightweight_charts import renderLightweightCharts
from chart_utils import df_to_technical_chart_data, build_technical_chart_config
from partnerships_config import FILER_CAP_USD_MAX, FILER_CAP_USD_MIN
from thirteenf_config import THIRTEENF_INSTITUTIONS
from thirteenf_service import (
    get_13f_filings_for_institution,
    get_13f_holdings,
    get_13f_holdings_by_quarter,
    get_13f_compare,
    get_holders_by_cusip,
    get_overlap_holdings,
)
from macro_data import fetch_macro_indicator
from macro_context import build_macro_context, macro_context_to_dataframes
from portfolio_insights import build_portfolio_insights
import yfinance as yf

from data_schemas import build_dashboard_export_payload, tca_to_schema
from factor_exposure import (
    FactorAttributionResult,
    build_factor_attribution,
    build_factor_exposure,
    load_ff5_factors,
    resolve_attribution_window,
)
from fi_context import build_fi_context_strip, fi_rows_to_records
from options_black_scholes import black_scholes_european
from options_iv_term import build_iv_term_structure
from tca_estimate import estimate_trade_impact
from portfolio_snapshot import fetch_portfolio_snapshot_dict
from relevant_news import build_relevant_news
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root (caches at Invest/ root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(ttl=900, show_spinner="Loading portfolio snapshot…")  # Cache for 15 minutes
def get_portfolio_data():
    """
    Fetch portfolio summary with caching to reduce API calls.

    Returns cached data for 15 minutes to avoid Yahoo Finance rate limits.
    """
    return fetch_portfolio_snapshot_dict()


@st.cache_data(ttl=900, show_spinner="Loading macro snapshot…")
def _cached_macro_context():
    return build_macro_context()


@st.cache_data(ttl=900, show_spinner="Loading portfolio risk snapshot…")
def _cached_portfolio_insights(positions_key: Tuple[Tuple[str, float], ...]):
    positions = [{"ticker": t, "current_value": v} for t, v in positions_key]
    return build_portfolio_insights(positions, get_company_profile, fetch_ohlcv)


def _ff_factors_cache_mtime() -> float:
    p = _PROJECT_ROOT / ".market_cache" / "ff5_factors_daily.csv"
    try:
        return float(os.path.getmtime(p))
    except OSError:
        return 0.0


@st.cache_data(ttl=900, show_spinner="Loading factor loadings…")
def _cached_factor_exposure(
    positions_key: Tuple[Tuple[str, float], ...],
    ff_cache_mtime: float,
):
    """ff_cache_mtime keys the cache so a refreshed FF file on disk recomputes loadings."""
    _ = ff_cache_mtime
    positions = [{"ticker": t, "current_value": v} for t, v in positions_key]
    return build_factor_exposure(positions, fetch_ohlcv, period_years=3)


def _empty_factor_attribution(warnings: List[str]) -> FactorAttributionResult:
    return FactorAttributionResult(
        attribution_start=None,
        attribution_end=None,
        estimation_start=None,
        estimation_end=None,
        n_attribution_days=0,
        portfolio_alpha=0.0,
        portfolio_factor_betas={},
        cumulative_excess_return=0.0,
        cumulative_factor_contributions={},
        cumulative_alpha_component=0.0,
        cumulative_residual=0.0,
        data_warnings=list(warnings),
        available=False,
    )


@st.cache_data(ttl=900, show_spinner="Loading factor attribution…")
def _cached_factor_attribution(
    positions_key: Tuple[Tuple[str, float], ...],
    ff_cache_mtime: float,
    preset: str,
    custom_start_iso: str,
    custom_end_iso: str,
) -> FactorAttributionResult:
    """Attribution window + ff_cache_mtime in the cache key (3A)."""
    from dataclasses import replace

    _ = ff_cache_mtime
    positions: List[Dict[str, Any]] = [{"ticker": t, "current_value": v} for t, v in positions_key]
    ff, w_ff = load_ff5_factors()
    preset_l = (preset or "21").strip().lower()
    if preset_l == "custom":
        if not (custom_start_iso and custom_end_iso):
            return _empty_factor_attribution(
                list(w_ff) + ["Pick start and end dates for custom attribution."]
            )
        try:
            cs = date.fromisoformat(custom_start_iso)
            ce = date.fromisoformat(custom_end_iso)
        except ValueError:
            return _empty_factor_attribution(list(w_ff) + ["Invalid custom attribution dates."])
        a0, a1, w_bounds = resolve_attribution_window(ff, "custom", cs, ce)
    else:
        a0, a1, w_bounds = resolve_attribution_window(ff, preset_l)
    if a0 is None or a1 is None:
        return _empty_factor_attribution(list(w_ff) + list(w_bounds))
    fa = build_factor_attribution(positions, fetch_ohlcv, a0, a1, period_years=3)
    merged = list(w_ff) + list(w_bounds) + list(fa.data_warnings)
    return replace(fa, data_warnings=merged)


@st.cache_data(ttl=900, show_spinner="Estimating trade impact…")
def _cached_tca_estimate(ticker: str, shares: float, side: str):
    return estimate_trade_impact(
        ticker,
        shares,
        side,
        fetch_ohlcv,
        period_years=2,
    )


@st.cache_data(ttl=600, show_spinner="Loading options IV term…")
def _cached_iv_term_structure(ticker: str, spot_key: float):
    """spot_key is rounded spot for cache stability; 0 means let yfinance infer spot."""
    override = float(spot_key) if spot_key and spot_key > 0 else None
    return build_iv_term_structure(ticker, max_expirations=12, spot_override=override)


@st.cache_data(ttl=900, show_spinner="Loading credit & duration proxies…")
def _cached_fi_context_strip():
    return build_fi_context_strip()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_tnx_last_percent() -> float:
    """^TNX last close in yield percent points (e.g. 4.25). Fallback 4.25."""
    try:
        h = yf.Ticker("^TNX").history(period="5d", interval="1d", auto_adjust=True)
        if h is not None and not h.empty and "Close" in h.columns:
            v = float(h["Close"].dropna().iloc[-1])
            if v > 0:
                return v
    except Exception:
        pass
    return 4.25


@st.cache_data(ttl=600, show_spinner=False)
def _cached_relevant_news(port_tuple: Tuple[str, ...], watch_tuple: Tuple[str, ...]):
    return build_relevant_news(
        list(port_tuple),
        list(watch_tuple),
        lambda t, lim: fetch_company_news(t, lim),
    )


# -----------------------------------------------------------------------------
# Cached data loaders (st.cache_data) to avoid redundant work on reruns.
# Only idempotent, serializable-return functions; TTLs keep data reasonably fresh.
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Loading ticker summary…")
def _cached_get_ticker_summary(ticker: str, alert_price: Optional[float] = None):
    return get_ticker_summary(ticker, alert_price=alert_price)


@st.cache_data(ttl=600, show_spinner="Loading company profile…")
def _cached_get_company_profile(ticker: str):
    return get_company_profile(ticker)


@st.cache_data(ttl=600, show_spinner="Loading valuation data…")
def _cached_get_valuation_chart_data(ticker: str, years: int, skip_db: bool):
    return get_valuation_chart_data(ticker, years, skip_db=skip_db)


@st.cache_data(ttl=600, show_spinner="Loading fundamentals…")
def _cached_get_fundamentals_ratios(ticker: str):
    return get_fundamentals_ratios(ticker)


@st.cache_data(ttl=600, show_spinner="Loading company news…")
def _cached_fetch_company_news(ticker: str, limit: int):
    return fetch_company_news(ticker, limit)


@st.cache_data(ttl=600, show_spinner="Loading insider transactions…")
def _cached_fetch_insider_transactions(ticker: str):
    return fetch_insider_transactions(ticker)


@st.cache_data(ttl=1800, show_spinner="Loading IPO calendar…")
def _cached_fetch_ipo_calendar(days_ahead: int):
    return fetch_ipo_calendar(days_ahead=days_ahead)


@st.cache_data(ttl=900, show_spinner="Loading partnership events…")
def _cached_get_partnership_events(limit: int):
    return get_partnership_events(limit=limit, defer_yfinance=True)


@st.cache_data(ttl=900, show_spinner="Loading 13F filings…")
def _cached_get_13f_filings_for_institution(cik: str):
    return get_13f_filings_for_institution(cik)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_get_company_website(ticker: str, company_name: str) -> str:
    import urllib.parse
    try:
        import yfinance as yf
        url = yf.Ticker(ticker).info.get("website")
        if url:
            return url
    except Exception:
        pass
    query = urllib.parse.quote_plus(f"{company_name}")
    return f"https://www.google.com/search?q={query}"


@st.cache_data(ttl=900, show_spinner="Loading 13F holdings…")
def _cached_get_13f_holdings_by_quarter(cik: str, year: int, quarter: int):
    return get_13f_holdings_by_quarter(cik, year, quarter)


@st.cache_data(ttl=900, show_spinner="Comparing 13F filings…")
def _cached_get_13f_compare(cik: str, accession_a: str, accession_b: str):
    return get_13f_compare(cik, accession_a, accession_b)


@st.cache_data(ttl=900, show_spinner="Loading 13F holders…")
def _cached_get_holders_by_cusip(cusip: str, institution_ciks: tuple, year: int, quarter: int):
    return get_holders_by_cusip(cusip, list(institution_ciks), year, quarter)


@st.cache_data(ttl=900, show_spinner="Loading overlap holdings…")
def _cached_get_overlap_holdings(cik_list: tuple, year: int, quarter: int):
    return get_overlap_holdings(list(cik_list), year, quarter)

@st.cache_data(ttl=86400, show_spinner="Loading macro indicator…")
def _cached_fetch_macro_indicator(metric: str):
    return fetch_macro_indicator(metric)


# Page configuration
st.set_page_config(
    page_title="Gavin Financial Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS overrides (base theme is in .streamlit/config.toml)
# Dashboard sections use a "terminal noir" treatment: amber accent, Sora + IBM Plex Sans.
st.markdown(
    dedent("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
    /* Tabular figures for tape-style numbers (see DESIGN.md) */
    .gft-tabular {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-variant-numeric: tabular-nums;
    }
    @media (prefers-reduced-motion: reduce) {
        .gft-dash-section { animation: none !important; }
    }
    .urgent-alert {
        background-color: #FF4B4B;
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background-color: var(--secondary-background-color, #262730);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    @keyframes gftDashReveal {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .gft-dash-page-tagline {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.82rem;
        color: #94a3b8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: -0.5rem 0 1rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(232, 168, 56, 0.2);
        max-width: 28rem;
    }
    :root {
        --gft-bg: #0E1117;
        --gft-surface: #262730;
        --gft-surface-deep: #151922;
        --gft-surface-soft: rgba(38, 39, 48, 0.64);
        --gft-border: rgba(148, 163, 184, 0.28);
        --gft-border-strong: rgba(232, 168, 56, 0.42);
        --gft-text: #F8FAFC;
        --gft-muted: #94A3B8;
        --gft-muted-2: #64748B;
        --gft-accent: #E8A838;
        --gft-accent-hover: #FBBF24;
        --gft-accent-soft: rgba(232, 168, 56, 0.12);
        --gft-research-accent: #2DD4BF;
        --gft-exec-accent: #A5B4FC;
        --gft-positive: #4ADE80;
        --gft-negative: #F87171;
        --gft-radius: 6px;
    }
    .gft-dash-section-stack-research {
        border-color: rgba(45, 212, 191, 0.24) !important;
        animation-delay: 0.08s;
    }
    .gft-dash-section-stack-research .gft-dash-kicker {
        color: var(--gft-research-accent) !important;
    }
    .gft-dash-section-stack-exec {
        border-color: rgba(165, 180, 252, 0.24) !important;
        animation-delay: 0.14s;
    }
    .gft-dash-section-stack-exec .gft-dash-kicker {
        color: var(--gft-exec-accent) !important;
    }
    .gft-dash-section {
        animation: gftDashReveal 0.55s ease-out;
        margin: 1.1rem 0 0.65rem 0;
        padding: 1rem 1.2rem 1.1rem 1.2rem;
        background:
            radial-gradient(ellipse 120% 80% at 0% 0%, rgba(232, 168, 56, 0.09), transparent 55%),
            linear-gradient(180deg, rgba(38, 39, 48, 0.55), rgba(14, 17, 23, 0.15));
        border: 1px solid rgba(232, 168, 56, 0.14);
        border-radius: 6px;
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
    }
    .gft-dash-kicker {
        font-family: 'Sora', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: #e8a838;
        font-weight: 600;
        display: block;
        margin-bottom: 0.3rem;
    }
    .gft-dash-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.28rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .gft-dash-sub {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0.45rem 0 0 0;
        line-height: 1.5;
    }
    .gft-fred-subhead {
        font-family: 'Sora', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #cbd5e1;
        margin: 1rem 0 0.4rem 0;
    }
    .gft-fred-hint {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.78rem;
        color: #64748b;
        margin: 0.35rem 0 0 0;
    }
    .gft-news-hero-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1rem 0;
    }
    .gft-news-card {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(155deg, rgba(38,39,48,0.92) 0%, rgba(18, 20, 28, 0.88) 100%);
        border: 1px solid rgba(232, 168, 56, 0.12);
        border-radius: 8px;
        padding: 0.9rem 1rem 0.85rem 1rem;
        position: relative;
        overflow: hidden;
    }
    .gft-news-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e8a838, rgba(45, 212, 191, 0.6), transparent);
        opacity: 0.95;
    }
    .gft-news-card-meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.45rem;
        flex-wrap: wrap;
    }
    .gft-news-score {
        font-family: 'Sora', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #0f172a;
        background: #e8a838;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    .gft-news-time {
        font-size: 0.68rem;
        color: #64748b;
    }
    .gft-news-tickers {
        font-size: 0.65rem;
        color: #2dd4bf;
        letter-spacing: 0.04em;
    }
    .gft-news-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin: 0.35rem 0 0.15rem 0;
    }
    .gft-news-tag {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 0.08rem 0.35rem;
        border-radius: 3px;
    }
    .gft-news-headline {
        font-size: 0.88rem;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.35;
        margin: 0 0 0.5rem 0;
    }
    .gft-news-card a {
        font-size: 0.72rem;
        color: #e8a838;
        text-decoration: none;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .gft-news-card a:hover {
        color: #fbbf24;
        text-decoration: underline;
    }
    .gft-alert-panel {
        margin: 0.5rem 0 1rem 0;
    }
    .gft-alert-banner {
        font-family: 'Sora', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fecaca;
        background: linear-gradient(105deg, rgba(248, 113, 113, 0.18), rgba(232, 168, 56, 0.06));
        border-left: 3px solid #f87171;
        padding: 0.65rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.65rem;
    }
    .gft-alert-card {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(180deg, rgba(38, 39, 48, 0.9), rgba(22, 24, 32, 0.85));
        border: 1px solid rgba(248, 113, 113, 0.22);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.45rem;
        font-size: 0.84rem;
        color: #e2e8f0;
        line-height: 1.45;
    }
    .gft-empty-state {
        font-family: 'IBM Plex Sans', sans-serif;
        border: 1px dashed rgba(232, 168, 56, 0.28);
        background: radial-gradient(ellipse 80% 60% at 0% 0%, rgba(232, 168, 56, 0.06), transparent),
                    rgba(38, 39, 48, 0.35);
        color: #94a3b8;
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .gft-cache-hint {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 0.6rem;
        letter-spacing: 0.02em;
    }
    .gft-dash-msg {
        font-family: 'IBM Plex Sans', sans-serif;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.65rem 0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .gft-dash-msg-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .gft-dash-msg-body {
        color: #cbd5e1;
        word-break: break-word;
    }
    .gft-dash-msg-error {
        border: 1px solid rgba(248, 113, 113, 0.45);
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.14), rgba(38, 39, 48, 0.65));
    }
    .gft-dash-msg-error .gft-dash-msg-title { color: #fecaca; }
    .gft-dash-msg-warning {
        border: 1px solid rgba(232, 168, 56, 0.42);
        background: linear-gradient(135deg, rgba(232, 168, 56, 0.11), rgba(38, 39, 48, 0.55));
    }
    .gft-dash-msg-warning .gft-dash-msg-title { color: #fbbf24; }
    .gft-dash-msg-info {
        border: 1px solid rgba(148, 163, 184, 0.28);
        background: rgba(38, 39, 48, 0.48);
    }
    .gft-dash-msg-info .gft-dash-msg-title { color: #94a3b8; }

    </style>
    """),
    unsafe_allow_html=True,
)

GLOBAL_STREAMLIT_CHROME_CSS = """
button,input,textarea,[data-baseweb="select"]{font-family:'IBM Plex Sans',sans-serif;}
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="tab"]:focus-visible,[data-baseweb="select"] *:focus-visible,section[data-testid="stSidebar"] label:focus-within{outline:2px solid var(--gft-accent)!important;outline-offset:2px!important;box-shadow:0 0 0 3px rgba(232,168,56,.16)!important;}
[data-testid="stTabs"] [role="tablist"]{gap:.25rem;border-bottom:1px solid rgba(148,163,184,.18);overflow-x:auto;scrollbar-width:thin;}
[data-testid="stTabs"] [role="tab"]{min-height:2.4rem;padding:.45rem .75rem;color:var(--gft-muted);border-radius:var(--gft-radius) var(--gft-radius) 0 0;border-bottom:2px solid transparent;white-space:nowrap;transition:background-color 120ms ease-out,color 120ms ease-out,border-color 120ms ease-out;}
[data-testid="stTabs"] [role="tab"]:hover{color:var(--gft-text);background:var(--gft-accent-soft);}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--gft-accent);border-bottom-color:var(--gft-accent);background:rgba(38,39,48,.46);}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{font-family:'Sora',sans-serif;color:var(--gft-text);}
section[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:.25rem;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{min-height:2.3rem;padding:.35rem .55rem .35rem .7rem;border:1px solid transparent;border-left:3px solid transparent;border-radius:var(--gft-radius);color:var(--gft-muted);transition:background-color 120ms ease-out,border-color 120ms ease-out,color 120ms ease-out;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(148,163,184,.08);color:var(--gft-text);}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:var(--gft-accent-soft);border-color:rgba(232,168,56,.24);border-left-color:var(--gft-accent);color:var(--gft-text);}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stFormSubmitButton"] button{min-height:2.35rem;border-radius:var(--gft-radius);border:1px solid var(--gft-border);background:rgba(38,39,48,.72);color:var(--gft-text);font-weight:500;transition:background-color 120ms ease-out,border-color 120ms ease-out,color 120ms ease-out,transform 80ms ease-out;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover,[data-testid="stFormSubmitButton"] button:hover{border-color:var(--gft-border-strong);background:var(--gft-accent-soft);color:var(--gft-text);}
[data-testid="stButton"] button:active,[data-testid="stDownloadButton"] button:active,[data-testid="stFormSubmitButton"] button:active{transform:translateY(1px);}
[data-testid="stButton"] button[kind="primary"],[data-testid="stButton"] button[data-testid="baseButton-primary"],[data-testid="stDownloadButton"] button[kind="primary"],[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"],[data-testid="stFormSubmitButton"] button[kind="primary"],[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]{background:var(--gft-accent);border-color:var(--gft-accent);color:#0F172A;font-weight:700;}
[data-testid="stButton"] button[kind="primary"]:hover,[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover,[data-testid="stDownloadButton"] button[kind="primary"]:hover,[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"]:hover,[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover{background:var(--gft-accent-hover);border-color:var(--gft-accent-hover);color:#0F172A;}
[data-testid="stButton"] button:disabled,[data-testid="stDownloadButton"] button:disabled,[data-testid="stFormSubmitButton"] button:disabled{opacity:.48;cursor:not-allowed;}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stTextArea"] textarea,[data-baseweb="select"]>div{background:var(--gft-surface-deep);color:var(--gft-text);border-color:var(--gft-border);border-radius:var(--gft-radius);}
[data-testid="stTextInput"] input:hover,[data-testid="stNumberInput"] input:hover,[data-testid="stDateInput"] input:hover,[data-testid="stTextArea"] textarea:hover,[data-baseweb="select"]>div:hover{border-color:rgba(232,168,56,.32);}
[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus,[data-testid="stDateInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:var(--gft-accent)!important;box-shadow:0 0 0 3px rgba(232,168,56,.14)!important;}
[data-testid="stTextInput"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder{color:var(--gft-muted-2);opacity:1;}
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label{color:var(--gft-muted);}
[data-testid="stMetric"]{padding:.55rem .65rem;border:1px solid rgba(148,163,184,.16);border-radius:var(--gft-radius);background:rgba(38,39,48,.32);}
[data-testid="stMetricLabel"]{color:var(--gft-muted);font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stMetricValue"]{color:var(--gft-text);font-family:'JetBrains Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid rgba(148,163,184,.18);border-radius:var(--gft-radius);background:rgba(14,17,23,.36);overflow:hidden;}
[data-testid="stDataFrame"] button{color:var(--gft-muted);}
[data-testid="stDataFrame"] button:hover{color:var(--gft-accent);}
"""
st.markdown(
    f"<style>{' '.join(GLOBAL_STREAMLIT_CHROME_CSS.split())}</style>",
    unsafe_allow_html=True,
)

# Cohesive accent palette for dashboard Plotly charts (sector pie, etc.)
GFT_DASH_CHART_PALETTE = [
    "#e8a838",
    "#2dd4bf",
    "#c45c3e",
    "#818cf8",
    "#c084fc",
    "#4ade80",
    "#f472b6",
    "#38bdf8",
    "#fbbf24",
    "#fb7185",
]


def initialize_database():
    """Initialize database tables if they don't exist."""
    try:
        init_db()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False


def format_currency(value) -> str:
    """Format a value as currency."""
    if value is None:
        return "$0.00"
    val = float(value)
    if val >= 0:
        return f"${val:,.2f}"
    return f"-${abs(val):,.2f}"


def format_percentage(value) -> str:
    """Format a value as percentage."""
    if value is None:
        return "0.00%"
    return f"{float(value):+.2f}%"


def _gft_metrics_container():
    """Borders metric rows on Streamlit >= 1.29; plain container otherwise."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _gft_dash_callout(kind: str, title: str, body: str) -> None:
    """Terminal-styled error / warning / info (Dashboard)."""
    if kind not in ("error", "warning", "info"):
        kind = "info"
    cls = f"gft-dash-msg gft-dash-msg-{kind}"
    st.markdown(
        f'<div class="{cls}"><div class="gft-dash-msg-title">{html.escape(title)}</div>'
        f'<div class="gft-dash-msg-body">{html.escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def _gft_dash_section_header(
    kicker: str,
    title: str,
    subtitle: Optional[str] = None,
    wrap_class: Optional[str] = None,
) -> None:
    sub_html = (
        f'<p class="gft-dash-sub">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    cls = "gft-dash-section"
    if wrap_class:
        cls = f"{cls} {wrap_class}"
    st.markdown(
        f'<div class="{cls}">'
        f'<span class="gft-dash-kicker">{html.escape(kicker)}</span>'
        f'<h2 class="gft-dash-title">{html.escape(title)}</h2>'
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _trim_macro_note_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Note" not in df.columns:
        return df
    col = df["Note"].astype(str).str.strip()
    if col.eq("").all() or col.eq("nan").all():
        return df.drop(columns=["Note"])
    return df


def _style_macro_movers_styler(df: pd.DataFrame):
    df = _trim_macro_note_column(df.copy())

    def _pct_colors(series: pd.Series):
        styles = []
        for v in series:
            if pd.isna(v) or not isinstance(v, (int, float)):
                styles.append("")
            elif v > 0:
                styles.append("color: #2dd4bf; font-weight: 600")
            elif v < 0:
                styles.append("color: #f87171; font-weight: 600")
            else:
                styles.append("")
        return styles

    styler = df.style.set_properties(
        **{
            "font-family": "'JetBrains Mono', ui-monospace, monospace",
            "font-variant-numeric": "tabular-nums",
        }
    )
    if "Change %" in df.columns:
        styler = styler.apply(_pct_colors, subset=["Change %"])
    if "Δ/σ" in df.columns:
        styler = styler.apply(_pct_colors, subset=["Δ/σ"])
    fmt: dict = {}
    if "Last" in df.columns:
        fmt["Last"] = "{:.2f}"
    if "Prev close" in df.columns:
        fmt["Prev close"] = "{:.2f}"
    if "Change %" in df.columns:
        fmt["Change %"] = "{:+.2f}%"
    if "σ 20d %" in df.columns:
        fmt["σ 20d %"] = "{:.2f}"
    if "Δ/σ" in df.columns:
        fmt["Δ/σ"] = "{:+.2f}"
    if fmt:
        styler = styler.format(fmt, na_rep="—")
    return styler


def _style_fred_rates_styler(df: pd.DataFrame):
    if df.empty:
        return df
    styler = df.style.set_properties(
        **{
            "font-family": "'JetBrains Mono', ui-monospace, monospace",
            "font-variant-numeric": "tabular-nums",
        }
    )
    fmt = {}
    if "Value" in df.columns:
        fmt["Value"] = "{:.3f}"
    if fmt:
        styler = styler.format(fmt, na_rep="—")
    return styler


def _gft_tabular_styler(df: pd.DataFrame):
    """JetBrains Mono + tabular figures for tape-style tables (see DESIGN.md)."""
    if df.empty:
        return df
    return df.style.set_properties(
        **{
            "font-family": "'JetBrains Mono', ui-monospace, monospace",
            "font-variant-numeric": "tabular-nums",
        }
    )


def _gft_partnerships_styler(df: pd.DataFrame, dim_row_indices: Optional[set[int]] = None):
    """Tape table + optional muted rows for out-of-cap filers (DESIGN.md secondary text #94A3B8)."""
    if df.empty:
        return df
    styler = df.style.set_properties(
        **{
            "font-family": "'JetBrains Mono', ui-monospace, monospace",
            "font-variant-numeric": "tabular-nums",
        }
    )
    if dim_row_indices:
        muted = "#94A3B8"

        def _dim_row(row):
            if row.name in dim_row_indices:
                return [f"color: {muted}"] * len(row)
            return [""] * len(row)

        styler = styler.apply(_dim_row, axis=1)
    return styler


def _gft_render_news_hero_cards(ranked: list, limit: int = 3) -> None:
    """Top stories as compact cards (RankedNewsItem list)."""
    if not ranked:
        return
    parts: list[str] = ['<div class="gft-news-hero-row">']
    for item in ranked[:limit]:
        ts = item.datetime.strftime("%Y-%m-%d %H:%M") if item.datetime else "—"
        tickers = html.escape(", ".join(item.tickers_matched[:6]))
        if len(item.tickers_matched) > 6:
            tickers += "…"
        head = html.escape(item.headline[:180] + ("…" if len(item.headline) > 180 else ""))
        link = html.escape(item.url) if item.url else ""
        link_html = (
            f'<a href="{link}" target="_blank" rel="noopener noreferrer">Open article ↗</a>'
            if link
            else '<span style="color:#64748b;font-size:0.72rem;">No link</span>'
        )
        tags_html = ""
        if item.event_tags:
            pills = "".join(
                f'<span class="gft-news-tag">{html.escape(t)}</span>'
                for t in item.event_tags[:3]
            )
            tags_html = f'<div class="gft-news-tags" title="Rule-based tags (heuristic)">{pills}</div>'
        parts.append(
            '<div class="gft-news-card">'
            '<div class="gft-news-card-meta">'
            f'<span class="gft-news-score">SCORE {item.score}</span>'
            f'<span class="gft-news-time">{html.escape(ts)}</span>'
            "</div>"
            f'<div class="gft-news-tickers">{tickers}</div>'
            f"{tags_html}"
            f'<p class="gft-news-headline">{head}</p>'
            f"{link_html}"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _gft_render_urgent_tax_alerts(lots: list, format_currency_fn) -> None:
    """Styled alert strip for lots nearing long-term status (dicts from cached portfolio)."""
    if not lots:
        return
    n = len(lots)
    parts: list[str] = [
        '<div class="gft-alert-panel">',
        '<div class="gft-alert-banner">'
        f"Alerts · {n} tax lot(s) nearing long-term status"
        "</div>",
    ]
    for lot in lots:
        t = html.escape(str(lot.get("ticker", "")))
        shares = float(lot.get("shares") or 0)
        pd = html.escape(str(lot.get("purchase_date", "")))
        days = lot.get("days_until_long_term", "")
        cb = format_currency_fn(lot.get("cost_basis", 0))
        cb_e = html.escape(cb)
        parts.append(
            '<div class="gft-alert-card">'
            f'<strong style="color:#e8a838">{t}</strong>'
            f" · {shares:.4f} sh · purchase {pd} · "
            f'<span style="color:#fbbf24;font-weight:600">{html.escape(str(days))}d</span> to LTC'
            f" · basis {cb_e}"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def dashboard_page():
    """Display the main dashboard page with portfolio overview."""
    st.title("Dashboard")
    st.markdown(
        '<p class="gft-dash-page-tagline">Tape · book · tax · flow</p>',
        unsafe_allow_html=True,
    )

    # Add refresh button and optional JSON export (cached portfolio snapshot)
    col_refresh, col_export, col_spacer = st.columns([1, 1, 4])
    with col_refresh:
        if st.button("Refresh prices"):
            get_portfolio_data.clear()
            _cached_macro_context.clear()
            _cached_portfolio_insights.clear()
            _cached_factor_exposure.clear()
            _cached_factor_attribution.clear()
            _cached_tca_estimate.clear()
            _cached_relevant_news.clear()
            _cached_fi_context_strip.clear()
            _cached_tnx_last_percent.clear()
            st.session_state.pop("gft_export_tca", None)
            st.rerun()
    with col_export:
        _snap = get_portfolio_data()
        if _snap and _snap.get("positions"):
            _pk = tuple(
                sorted(
                    (p["ticker"], round(float(p["current_value"]), 2))
                    for p in _snap["positions"]
                )
            )
            try:
                _macro_e = _cached_macro_context()
                _ins_e = _cached_portfolio_insights(_pk)
                _fe_e = _cached_factor_exposure(_pk, _ff_factors_cache_mtime())
                _preset = st.session_state.get("dash_attr_preset", "21")
                _c0 = st.session_state.get("dash_attr_c0")
                _c1 = st.session_state.get("dash_attr_c1")
                _is_custom = str(_preset).strip().lower() == "custom"
                _c0_iso = _c0.isoformat() if _is_custom and _c0 else ""
                _c1_iso = _c1.isoformat() if _is_custom and _c1 else ""
                _fa_e = _cached_factor_attribution(
                    _pk,
                    _ff_factors_cache_mtime(),
                    str(_preset),
                    _c0_iso,
                    _c1_iso,
                )
                _preset_label = "custom" if str(_preset).lower() == "custom" else str(_preset)
                _payload = build_dashboard_export_payload(
                    _macro_e,
                    _ins_e,
                    _fe_e,
                    None,
                    factor_attribution=_fa_e,
                    factor_attribution_preset=_preset_label,
                )
                if st.session_state.get("gft_export_tca"):
                    _payload["tca_estimate"] = st.session_state["gft_export_tca"]
                st.download_button(
                    label="Download JSON snapshot",
                    data=json.dumps(_payload, indent=2, default=str),
                    file_name="gft_dashboard_snapshot.json",
                    mime="application/json",
                    help="Macro, PORT-lite, Fama–French loadings and attribution strip; includes last TCA run from this session if present.",
                    key="dash_export_json",
                )
            except Exception:
                st.caption("Export unavailable")

    _gft_dash_section_header(
        "Context · GMM / BTMM",
        "Macro snapshot",
        "Cross-asset movers (Yahoo Finance, daily closes). σ 20d is stdev of last 20 daily % changes; "
        "Δ/σ is yesterday’s move vs that σ (omitted for VIX). Treasury & Fed via FRED when FRED_API_KEY is set. "
        "Not a Bloomberg terminal—same job: know the tape before you read the story.",
    )
    try:
        _macro = _cached_macro_context()
        _m_df, _r_df = macro_context_to_dataframes(_macro)
        if _m_df.empty:
            st.markdown(
                '<div class="gft-empty-state">No macro rows returned. Check your network connection '
                "and Yahoo Finance availability.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                _style_macro_movers_styler(_m_df),
                use_container_width=True,
                hide_index=True,
            )
        if _macro.fred_configured and not _r_df.empty:
            st.markdown(
                '<p class="gft-fred-subhead">Rates · FRED</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                _style_fred_rates_styler(_r_df),
                use_container_width=True,
                hide_index=True,
            )
        elif not _macro.fred_configured:
            st.markdown(
                '<p class="gft-fred-hint">Set <strong>FRED_API_KEY</strong> in <code>.env</code> for Treasury & Fed snapshot.</p>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        _gft_dash_callout(
            "warning",
            "Macro snapshot unavailable",
            str(e),
        )

    try:
        st.markdown(
            '<p class="gft-fred-subhead">FI · Credit & duration proxies</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "HYG/LQD/TLT/IEF and ^TNX via Yahoo—liquid ETFs and yield index, not TRACE or live bond inventory (SRCH/YAS-lite)."
        )
        _fi_rows, _fi_errs = _cached_fi_context_strip()
        _fi_df = pd.DataFrame(fi_rows_to_records(_fi_rows))
        if not _fi_df.empty:
            if "Note" in _fi_df.columns and _fi_df["Note"].astype(str).str.strip().eq("").all():
                _fi_df = _fi_df.drop(columns=["Note"])
            st.dataframe(
                _fi_df.style.set_properties(
                    **{
                        "font-family": "'JetBrains Mono', ui-monospace, monospace",
                        "font-variant-numeric": "tabular-nums",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if _fi_errs:
            st.caption("Notes: " + " ".join(_fi_errs[:3]))
    except Exception as e:
        _gft_dash_callout("info", "FI context strip unavailable", str(e))

    st.markdown("---")

    try:
        # Use cached portfolio data to avoid rate limits
        summary = get_portfolio_data()

        if summary is None:
            _gft_dash_callout(
                "error",
                "Portfolio data",
                "Could not load portfolio summary. Check the database connection and try Refresh.",
            )
            return

        _gft_dash_section_header(
            "Book · Summary",
            "Portfolio overview",
            "Live weights and P/L from cached quotes (15 min TTL). Refresh updates macro, risk, news, and prices together.",
        )
        with _gft_metrics_container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Positions",
                    len(summary["positions"]),
                    help="Number of unique stocks held",
                )
            with col2:
                st.metric(
                    "Portfolio Value",
                    format_currency(summary["total_value"]),
                    help="Current market value of all holdings",
                )
            with col3:
                delta_color = "normal" if summary["total_unrealized_gain"] >= 0 else "inverse"
                st.metric(
                    "Unrealized Gain/Loss",
                    format_currency(summary["total_unrealized_gain"]),
                    format_percentage(summary["total_unrealized_gain_pct"]),
                    delta_color=delta_color,
                )
            with col4:
                st.metric(
                    "Cost Basis",
                    format_currency(summary["total_cost_basis"]),
                    help="Total amount invested",
                )

        positions_key = tuple(
            sorted(
                (p["ticker"], round(float(p["current_value"]), 2))
                for p in summary["positions"]
            )
        )
        if summary["positions"]:
            st.markdown("---")
            _gft_dash_section_header(
                "Research · PORT",
                "Portfolio risk snapshot",
                "Sector weights from cached profiles; value-weighted β vs SPY (~6mo overlap on daily returns). "
                "Exposure view for a personal book—not a buy-side risk engine.",
            )
            try:
                insights = _cached_portfolio_insights(positions_key)
                if insights.data_warnings:
                    st.caption("Notes: " + " ".join(insights.data_warnings[:8]))
                with _gft_metrics_container():
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    with rc1:
                        st.metric("Top position %", f"{insights.top1_pct:.1f}%")
                    with rc2:
                        st.metric("Top 5 weight %", f"{insights.top5_pct:.1f}%")
                    with rc3:
                        st.metric("HHI (concentration)", f"{insights.herfindahl:.3f}")
                    with rc4:
                        beta_txt = (
                            f"{insights.portfolio_beta:.2f}"
                            if insights.portfolio_beta is not None
                            else "—"
                        )
                        st.metric("Value-weighted β vs SPY", beta_txt)
                if insights.sector_weights:
                    sec_names = list(insights.sector_weights.keys())
                    sec_vals = list(insights.sector_weights.values())
                    n_sec = len(sec_names)
                    pie_colors = (GFT_DASH_CHART_PALETTE * ((n_sec // len(GFT_DASH_CHART_PALETTE)) + 1))[
                        :n_sec
                    ]
                    fig_sec = px.pie(
                        values=sec_vals,
                        names=sec_names,
                        title="Allocation by sector (estimated)",
                        color_discrete_sequence=pie_colors,
                    )
                    fig_sec.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        title_font_size=15,
                        title_font_family="Sora",
                        font_family="IBM Plex Sans",
                        legend_font_size=11,
                    )
                    fig_sec.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        textfont_color="#0f172a",
                        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                        marker=dict(line=dict(color="rgba(15,23,42,0.35)", width=1)),
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)
                if insights.per_ticker_beta:
                    with st.expander("Per-ticker beta (vs SPY)"):
                        beta_rows = [
                            {
                                "Ticker": t,
                                "Beta": round(b, 3),
                                "Weight in β": f"{insights.beta_weights_used.get(t, 0) * 100:.1f}%",
                            }
                            for t, b in sorted(
                                insights.per_ticker_beta.items(),
                                key=lambda x: -insights.beta_weights_used.get(x[0], 0),
                            )
                        ]
                        st.dataframe(
                            _gft_tabular_styler(pd.DataFrame(beta_rows)),
                            use_container_width=True,
                            hide_index=True,
                        )
            except Exception as e:
                _gft_dash_callout(
                    "warning",
                    "Portfolio risk snapshot unavailable",
                    str(e),
                )

            st.markdown("---")
            _gft_dash_section_header(
                "Research · factors",
                "Fama–French 5 loadings (value-weighted)",
                "Daily excess returns regressed on Mkt-RF, SMB, HML, RMW, CMA from the "
                "Kenneth French Data Library. Illustrative—not a buy-side risk engine.",
                wrap_class="gft-dash-section-stack-research",
            )
            st.markdown(
                '<p class="gft-dash-msg gft-dash-msg-info" style="margin:0.35rem 0 0.85rem 0">'
                "<strong class=\"gft-dash-msg-title\">Holdings vs exposures</strong> · "
                "Sector weights describe what you own; factor loadings describe sensitivity to "
                "systematic factors over the <em>estimation</em> window. They are not the same "
                "coordinate system.</p>",
                unsafe_allow_html=True,
            )
            try:
                fe = _cached_factor_exposure(positions_key, _ff_factors_cache_mtime())
                if fe.data_warnings:
                    st.caption("Notes: " + " ".join(fe.data_warnings[:8]))
                if fe.factors_available and any(abs(v) > 1e-9 for v in fe.portfolio_factor_betas.values()):
                    sorted_betas = sorted(
                        fe.portfolio_factor_betas.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )
                    labels = [x[0] for x in sorted_betas]
                    vals = [x[1] for x in sorted_betas]
                    bar_colors = [
                        GFT_DASH_CHART_PALETTE[i % len(GFT_DASH_CHART_PALETTE)]
                        for i in range(len(labels))
                    ]
                    fig_f = go.Figure(
                        go.Bar(
                            x=vals,
                            y=labels,
                            orientation="h",
                            marker_color=bar_colors,
                            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                        )
                    )
                    fig_f.update_layout(
                        title={
                            "text": "Portfolio factor loadings rank by |β|",
                            "font": {"family": "Sora", "size": 15, "color": "#f8fafc"},
                        },
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        font_family="IBM Plex Sans",
                        xaxis=dict(
                            gridcolor="rgba(148,163,184,0.12)",
                            zeroline=True,
                            zerolinecolor="rgba(248,250,252,0.35)",
                            title="Loading (OLS coefficient)",
                        ),
                        yaxis=dict(autorange="reversed", title=""),
                        margin=dict(l=8, r=12, t=56, b=48),
                    )
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.caption(
                        f"Estimation window: {fe.regression_start} → {fe.regression_end} · "
                        "Ken French daily 5-factor (2×3) · as-of "
                        f"{fe.as_of or '—'}"
                    )
                    st.markdown(
                        '<p class="gft-fred-subhead" style="margin-top:1.1rem">Attribution</p>',
                        unsafe_allow_html=True,
                    )
                    _attr_preset = st.selectbox(
                        "Attribution window",
                        options=["21", "63", "mtd", "custom"],
                        format_func=lambda x: {
                            "21": "Last 21 trading days",
                            "63": "Last 63 trading days",
                            "mtd": "Month to date",
                            "custom": "Custom range",
                        }[x],
                        key="dash_attr_preset",
                    )
                    if _attr_preset == "custom":
                        _ac1, _ac2 = st.columns(2)
                        with _ac1:
                            st.date_input(
                                "Attribution start",
                                value=date.today() - timedelta(days=30),
                                key="dash_attr_c0",
                            )
                        with _ac2:
                            st.date_input(
                                "Attribution end",
                                value=date.today(),
                                key="dash_attr_c1",
                            )
                    _c0_iso_ui = ""
                    _c1_iso_ui = ""
                    if _attr_preset == "custom":
                        _d0 = st.session_state.get("dash_attr_c0")
                        _d1 = st.session_state.get("dash_attr_c1")
                        _c0_iso_ui = _d0.isoformat() if _d0 else ""
                        _c1_iso_ui = _d1.isoformat() if _d1 else ""
                    _fa = _cached_factor_attribution(
                        positions_key,
                        _ff_factors_cache_mtime(),
                        _attr_preset,
                        _c0_iso_ui,
                        _c1_iso_ui,
                    )
                    if _fa.data_warnings:
                        st.caption("Attribution notes: " + " ".join(_fa.data_warnings[:6]))
                    if _fa.available:
                        _attr_items: List[Tuple[str, float]] = [
                            (fn, float(_fa.cumulative_factor_contributions.get(fn, 0.0)))
                            for fn in fe.factor_names
                        ]
                        _attr_items.append(("Alpha", float(_fa.cumulative_alpha_component)))
                        _attr_items.append(("Residual", float(_fa.cumulative_residual)))
                        _attr_items.sort(key=lambda x: abs(x[1]), reverse=True)
                        _alabels = [x[0] for x in _attr_items]
                        _avals = [x[1] for x in _attr_items]
                        _acolors: List[str] = []
                        for lb in _alabels:
                            if lb == "Alpha":
                                _acolors.append("#94a3b8")
                            elif lb == "Residual":
                                _acolors.append("#64748b")
                            else:
                                try:
                                    _fi = fe.factor_names.index(lb)
                                except ValueError:
                                    _fi = 0
                                _acolors.append(
                                    GFT_DASH_CHART_PALETTE[_fi % len(GFT_DASH_CHART_PALETTE)]
                                )
                        fig_attr = go.Figure(
                            go.Bar(
                                x=_avals,
                                y=_alabels,
                                orientation="h",
                                marker_color=_acolors,
                                hovertemplate="%{y}: %{x:.4f} (cumulative)<extra></extra>",
                            )
                        )
                        fig_attr.update_layout(
                            title={
                                "text": "Cumulative attribution vs factors + residual",
                                "font": {"family": "Sora", "size": 15, "color": "#f8fafc"},
                            },
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e2e8f0",
                            font_family="IBM Plex Sans",
                            xaxis=dict(
                                gridcolor="rgba(148,163,184,0.12)",
                                zeroline=True,
                                zerolinecolor="rgba(248,250,252,0.35)",
                                title="Cumulative contribution (decimal)",
                            ),
                            yaxis=dict(autorange="reversed", title=""),
                            margin=dict(l=8, r=12, t=56, b=48),
                        )
                        st.plotly_chart(fig_attr, use_container_width=True)
                        st.caption(
                            f"Estimation (β fixed): {_fa.estimation_start} → {_fa.estimation_end} · "
                            f"Attribution: {_fa.attribution_start} → {_fa.attribution_end} "
                            f"({_fa.n_attribution_days} days). "
                            "Residual = sum of daily (portfolio excess − factor-explained)."
                        )
                    elif not _fa.data_warnings:
                        st.caption("Attribution unavailable for the selected window.")
                    wmap = {t: v for t, v in positions_key}
                    tickers_by_w = sorted(
                        fe.per_ticker_betas.keys(),
                        key=lambda tk: -float(wmap.get(tk, 0)),
                    )
                    with st.expander("Per-ticker factor loadings"):
                        if len(tickers_by_w) <= 15:
                            zmat = [
                                [fe.per_ticker_betas[tk].get(fn, 0.0) for fn in fe.factor_names]
                                for tk in tickers_by_w
                            ]
                            fig_hm = go.Figure(
                                data=go.Heatmap(
                                    z=zmat,
                                    x=list(fe.factor_names),
                                    y=list(tickers_by_w),
                                    zmid=0,
                                    colorscale="RdBu",
                                    hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
                                )
                            )
                            fig_hm.update_layout(
                                title="Loadings heatmap (top names by portfolio weight)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0",
                                font_family="IBM Plex Sans",
                                title_font_family="Sora",
                                title_font_size=14,
                                margin=dict(l=8, r=8, t=48, b=8),
                            )
                            st.plotly_chart(fig_hm, use_container_width=True)
                        rows = []
                        for tk in tickers_by_w:
                            row: dict = {
                                "Ticker": tk,
                                "R²": round(fe.per_ticker_r2.get(tk, 0.0), 3),
                                "n": fe.per_ticker_n_obs.get(tk, 0),
                            }
                            for fn in fe.factor_names:
                                row[fn] = round(fe.per_ticker_betas[tk].get(fn, 0.0), 3)
                            rows.append(row)
                        st.dataframe(
                            _gft_tabular_styler(pd.DataFrame(rows)),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    _gft_dash_callout(
                        "info",
                        "Factor model unavailable",
                        "Ken French factors could not be loaded or no ticker produced a long enough "
                        "overlap. Use value-weighted β vs SPY in the PORT block above as a single-factor fallback.",
                    )
                    try:
                        ins_fb = _cached_portfolio_insights(positions_key)
                        if ins_fb.portfolio_beta is not None:
                            st.metric("Fallback · β vs SPY", f"{ins_fb.portfolio_beta:.2f}")
                    except Exception:
                        pass
            except Exception as e:
                _gft_dash_callout("warning", "Factor exposure unavailable", str(e))
        else:
            st.markdown(
                '<div class="gft-empty-state">No positions yet — add trades to unlock sector exposure, '
                "β vs SPY, factor loadings, and the risk metrics deck.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        _gft_dash_section_header(
            "Execution · TCA",
            "Pre-trade impact (illustrative)",
            "Square-root participation vs recent ADV and realized vol. For sizing intuition only—"
            "not a calibrated Almgren–Chriss implementation and not execution advice.",
            wrap_class="gft-dash-section-stack-exec",
        )
        pos_tickers = [p["ticker"] for p in summary["positions"]] if summary["positions"] else []
        tc1, tc2, tc3 = st.columns([2, 1, 1])
        with tc1:
            if pos_tickers:
                mode = st.radio(
                    "Ticker source",
                    ["From portfolio", "Other symbol"],
                    horizontal=True,
                    key="dash_tca_mode",
                )
                if mode == "From portfolio":
                    tca_sym = st.selectbox("Ticker", pos_tickers, key="dash_tca_pick")
                else:
                    tca_sym = st.text_input("Symbol", "", key="dash_tca_sym").strip().upper()
            else:
                tca_sym = st.text_input("Symbol", "", key="dash_tca_sym_only").strip().upper()
        with tc2:
            tca_shares = st.number_input(
                "Shares",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="dash_tca_shares",
            )
        with tc3:
            tca_side = st.radio("Side", ["buy", "sell"], horizontal=True, key="dash_tca_side")
        if tca_sym and tca_shares > 0:
            tca_res = _cached_tca_estimate(tca_sym, float(tca_shares), tca_side)
            if tca_res is not None:
                try:
                    st.session_state["gft_export_tca"] = tca_to_schema(tca_res).model_dump(
                        mode="json"
                    )
                except Exception:
                    pass
                if tca_res.data_warnings:
                    st.caption("Notes: " + " ".join(tca_res.data_warnings[:5]))
                with _gft_metrics_container():
                    tm1, tm2, tm3, tm4 = st.columns(4)
                    with tm1:
                        st.metric("Est. impact (bps)", f"{tca_res.estimated_impact_bps:.1f}")
                    with tm2:
                        st.metric("Est. impact ($)", format_currency(tca_res.estimated_impact_usd))
                    with tm3:
                        part_pct = tca_res.participation_rate * 100
                        st.metric(
                            "Participation vs ADV",
                            f"{part_pct:.1f}%" if tca_res.adv_shares > 0 else "—",
                        )
                    with tm4:
                        st.metric("Ann. vol (realized)", f"{tca_res.annualized_volatility * 100:.1f}%")
                if tca_res.adv_shares > 0:
                    part_vis = min(tca_res.participation_rate * 100.0, 200.0)
                    fig_tca = go.Figure(
                        go.Bar(
                            x=[part_vis],
                            y=["Order vs ADV"],
                            orientation="h",
                            marker_color="#818cf8",
                            hovertemplate="Participation: %{x:.1f}% of one-day ADV<extra></extra>",
                        )
                    )
                    fig_tca.add_vline(
                        x=25,
                        line_width=2,
                        line_dash="dash",
                        line_color="rgba(251, 191, 36, 0.85)",
                    )
                    fig_tca.update_layout(
                        title={
                            "text": "Participation vs average daily volume (capped at 200% for chart scale)",
                            "font": {"family": "Sora", "size": 14, "color": "#f8fafc"},
                        },
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#cbd5e1",
                        font_family="IBM Plex Sans",
                        xaxis=dict(title="% of one-day ADV", gridcolor="rgba(148,163,184,0.12)"),
                        yaxis=dict(showticklabels=False),
                        showlegend=False,
                        margin=dict(l=8, r=12, t=56, b=40),
                        annotations=[
                            dict(
                                x=25,
                                y=1.15,
                                yref="paper",
                                text="25% ADV reference",
                                showarrow=False,
                                font=dict(size=11, color="#fbbf24"),
                            )
                        ],
                    )
                    st.plotly_chart(fig_tca, use_container_width=True)

        st.markdown("---")

        _gft_dash_section_header(
            "Tax · HIFO",
            "Tax liability summary",
            "Unrealized gains by character (short-term vs long-term) from open lots. Not tax advice.",
        )
        with _gft_metrics_container():
            tax_col1, tax_col2, tax_col3 = st.columns(3)
            with tax_col1:
                st.metric(
                    "Short-Term Gains",
                    format_currency(summary["short_term_gain"]),
                    help="Gains on positions held ≤ 365 days (taxed as ordinary income)",
                )
            with tax_col2:
                st.metric(
                    "Long-Term Gains",
                    format_currency(summary["long_term_gain"]),
                    help="Gains on positions held > 365 days (preferential tax rates)",
                )
            with tax_col3:
                total_gain = summary["short_term_gain"] + summary["long_term_gain"]
                st.metric(
                    "Total Unrealized Gains",
                    format_currency(total_gain),
                    help="Combined short-term and long-term gains",
                )
        
        if summary["urgent_lots"]:
            st.markdown("---")
            _gft_render_urgent_tax_alerts(summary["urgent_lots"], format_currency)

        st.markdown("---")
        _gft_dash_section_header(
            "Flow · TOP",
            "Headlines for your book",
            "Merged feed for portfolio + watchlist, ranked with transparent heuristics (mentions, keywords). "
            "Rule-based event tags (max 3) are assistive only; high-impact tags add +1 to score. "
            "Not Bloomberg TOP—same intent: less noise before you drill.",
        )
        try:
            db_wl = get_db_session()
            try:
                wl_rows = db_wl.query(Watchlist.ticker).all()
                watch_tuple = tuple(sorted((r[0] or "").upper().strip() for r in wl_rows if r[0]))
            finally:
                db_wl.close()
            port_tuple = tuple(
                sorted({(p["ticker"] or "").upper().strip() for p in summary["positions"] if p.get("ticker")})
            )
            if not port_tuple and not watch_tuple:
                st.markdown(
                    '<div class="gft-empty-state">Add portfolio positions or watchlist tickers on '
                    "<strong>Market Analysis</strong> to populate this feed.</div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Loading headlines…"):
                    ranked = _cached_relevant_news(port_tuple, watch_tuple)
                if not ranked:
                    st.markdown(
                        '<div class="gft-empty-state">No headlines returned. Configure <strong>FINNHUB_API_KEY</strong> '
                        "or OpenBB for company news.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    _gft_render_news_hero_cards(ranked, limit=3)
                    st.caption("Full feed")
                    news_rows = []
                    for item in ranked[:50]:
                        ts = item.datetime.strftime("%Y-%m-%d %H:%M") if item.datetime else ""
                        news_rows.append(
                            {
                                "Score": item.score,
                                "Time": ts,
                                "Tickers": ", ".join(item.tickers_matched),
                                "Tags": ", ".join(item.event_tags) if item.event_tags else "",
                                "Headline": item.headline,
                                "Source": item.source,
                                "Link": item.url or None,
                            }
                        )
                    news_df = pd.DataFrame(news_rows)
                    st.dataframe(
                        _gft_tabular_styler(news_df),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Link": st.column_config.LinkColumn(
                                "Article",
                                help="Open in browser",
                                display_text="Open ↗",
                            ),
                            "Headline": st.column_config.TextColumn(
                                "Headline",
                                width="large",
                            ),
                            "Score": st.column_config.NumberColumn("Score", format="%d"),
                        },
                    )
        except Exception as e:
            _gft_dash_callout(
                "warning",
                "Headlines unavailable",
                str(e),
            )

        st.markdown("---")

        if summary["positions"]:
            _gft_dash_section_header(
                "Analytics · Allocation",
                "Portfolio analytics",
                "Ticker weights vs tax-status P/L. Same palette as sector view above for a single visual language.",
            )

            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                MAX_SLICES = 8
                sorted_positions = sorted(
                    summary['positions'],
                    key=lambda p: p['current_value'],
                    reverse=True,
                )
                if len(sorted_positions) > MAX_SLICES:
                    top = sorted_positions[:MAX_SLICES]
                    other_value = sum(p['current_value'] for p in sorted_positions[MAX_SLICES:])
                    chart_names = [p['ticker'] for p in top] + ['Other']
                    chart_values = [p['current_value'] for p in top] + [other_value]
                else:
                    chart_names = [p['ticker'] for p in sorted_positions]
                    chart_values = [p['current_value'] for p in sorted_positions]

                n_t = len(chart_names)
                pie_colors_t = (GFT_DASH_CHART_PALETTE * ((n_t // len(GFT_DASH_CHART_PALETTE)) + 1))[:n_t]
                fig_pie = px.pie(
                    values=chart_values,
                    names=chart_names,
                    title="Portfolio Allocation by Ticker",
                    color_discrete_sequence=pie_colors_t,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title_font_size=15,
                    title_font_family="Sora",
                    font_family="IBM Plex Sans",
                    legend_font_size=11,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    textfont_color="#0f172a",
                    hovertemplate="%{label}: %{value:$,.2f} (%{percent})<extra></extra>",
                    marker=dict(line=dict(color="rgba(15,23,42,0.35)", width=1)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                if len(sorted_positions) > MAX_SLICES:
                    st.caption(
                        f"Showing top {MAX_SLICES} positions; "
                        f"{len(sorted_positions) - MAX_SLICES} smaller position(s) grouped as 'Other'."
                    )
            
            with viz_col2:
                # Bar chart - Gains by tax status
                gains_data = {
                    'Tax Status': ['Short-Term', 'Long-Term'],
                    'Gain/Loss': [summary['short_term_gain'], summary['long_term_gain']]
                }
                
                colors = ["#f87171" if g < 0 else "#2dd4bf" for g in gains_data["Gain/Loss"]]

                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=gains_data["Tax Status"],
                            y=gains_data["Gain/Loss"],
                            marker_color=colors,
                            text=[format_currency(g) for g in gains_data["Gain/Loss"]],
                            textposition="outside",
                        )
                    ]
                )
                fig_bar.update_layout(
                    title="Gains by Tax Status",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title_font_size=15,
                    title_font_family="Sora",
                    font_family="IBM Plex Sans",
                    yaxis_title="Gain/Loss ($)",
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            _gft_dash_section_header(
                "Ledger · Positions",
                "Position details",
                "Per-name lots, cost, and mark. Gain columns use terminal colors (teal / rose).",
            )

            positions_rows = []
            for p in summary['positions']:
                avg_cost = p['total_cost_basis'] / p['total_shares'] if p['total_shares'] else 0
                positions_rows.append({
                    "Ticker": p['ticker'],
                    "Shares": p['total_shares'],
                    "Avg Cost": avg_cost,
                    "Current Price": p['current_price'],
                    "Market Value": p['current_value'],
                    "Gain/Loss": p['unrealized_gain'],
                    "Gain %": p['unrealized_gain_pct'],
                    "ST Shares": p['short_term_shares'],
                    "LT Shares": p['long_term_shares'],
                })

            df_positions = (
                pd.DataFrame(positions_rows)
                .sort_values("Market Value", ascending=False)
                .reset_index(drop=True)
            )

            st.dataframe(
                df_positions.style.format({
                    "Shares": "{:,.4f}",
                    "Avg Cost": "${:,.2f}",
                    "Current Price": "${:,.2f}",
                    "Market Value": "${:,.2f}",
                    "Gain/Loss": "${:+,.2f}",
                    "Gain %": "{:+.2f}%",
                    "ST Shares": "{:,.4f}",
                    "LT Shares": "{:,.4f}",
                }).map(
                    lambda v: "color: #2dd4bf" if isinstance(v, (int, float)) and v > 0
                    else "color: #f87171" if isinstance(v, (int, float)) and v < 0
                    else "",
                    subset=["Gain/Loss", "Gain %"],
                ),
                use_container_width=True,
                hide_index=True,
            )
            
            st.markdown(
                '<p class="gft-cache-hint">Quotes refresh every 15 minutes by cache — use <strong>Refresh Prices</strong> to pull latest.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="gft-empty-state">No open positions yet. Add trades on <strong>Portfolio & Taxes</strong> '
                "to populate overview, risk, analytics, and the position grid.</div>",
                unsafe_allow_html=True,
            )
            
    except Exception as e:
        _gft_dash_callout(
            "error",
            "Dashboard load failed",
            str(e),
        )
        _gft_dash_callout(
            "info",
            "Checklist",
            "Confirm PostgreSQL is running, .env DB_* variables are correct, and tables initialize on first connect.",
        )


def portfolio_taxes_page():
    """Display portfolio and tax management page with trade entry and CSV import."""
    st.title("Portfolio & Taxes")
    st.markdown("---")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Manual entry", "CSV import", "Tax lots"])
    
    # Tab 1: Manual Trade Entry
    with tab1:
        st.subheader("Add New Trade")
        
        with st.form("trade_entry_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                ticker = st.text_input(
                    "Ticker Symbol",
                    placeholder="e.g., AAPL",
                    help="Enter the stock ticker symbol"
                ).upper()
                
                trade_date = st.date_input(
                    "Trade Date",
                    value=date.today(),
                    help="Select the date of the trade"
                )
                
                shares = st.number_input(
                    "Number of Shares",
                    min_value=0.0001,
                    step=0.0001,
                    format="%.4f",
                    help="Enter the number of shares"
                )
            
            with col2:
                price_per_share = st.number_input(
                    "Price per Share ($)",
                    min_value=0.01,
                    step=0.01,
                    format="%.2f",
                    help="Enter the price per share"
                )
                
                trade_type = st.selectbox(
                    "Trade Type",
                    options=["Buy", "Sell"],
                    help="Select whether this is a buy or sell transaction"
                )
            
            submitted = st.form_submit_button("Add Trade", use_container_width=True)
            
            if submitted:
                if not ticker:
                    st.error("Please enter a ticker symbol.")
                elif shares <= 0:
                    st.error("Number of shares must be greater than 0.")
                elif price_per_share <= 0:
                    st.error("Price per share must be greater than 0.")
                else:
                    try:
                        db: Session = get_db_session()
                        
                        new_trade = Trades(
                            ticker=ticker,
                            trade_date=trade_date,
                            shares=Decimal(str(shares)),
                            price_per_share=Decimal(str(price_per_share)),
                            trade_type=TradeType.BUY if trade_type == "Buy" else TradeType.SELL
                        )
                        
                        db.add(new_trade)
                        db.commit()
                        db.close()
                        
                        # Clear cache so new trade shows up
                        st.cache_data.clear()
                        
                        st.success(
                            f"Added {trade_type} trade: {shares} shares of {ticker} at ${price_per_share:.2f}"
                        )
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error adding trade: {str(e)}")
                        if db:
                            db.rollback()
                            db.close()
    
    # Tab 2: CSV Import
    with tab2:
        st.subheader("Import Trades from CSV")
        
        st.markdown("""
        Upload a CSV file with your trade history. The CSV should contain columns for:
        - **Ticker**: Stock symbol (e.g., AAPL)
        - **Date**: Trade date (e.g., 2024-01-15 or 01/15/2024)
        - **Quantity**: Number of shares
        - **Price**: Price per share
        - **Action**: Buy or Sell
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your trade history CSV"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.write("**Preview of uploaded data:**")
                st.dataframe(_gft_tabular_styler(df.head(10)), use_container_width=True)
                
                st.markdown("---")
                st.subheader("Map Your Columns")
                st.write("Select which column in your CSV corresponds to each field:")
                
                csv_columns = list(df.columns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    ticker_col = st.selectbox(
                        "Ticker Column",
                        options=csv_columns,
                        help="Column containing stock symbols"
                    )
                    
                    date_col = st.selectbox(
                        "Date Column",
                        options=csv_columns,
                        help="Column containing trade dates"
                    )
                    
                    quantity_col = st.selectbox(
                        "Quantity Column",
                        options=csv_columns,
                        help="Column containing number of shares"
                    )
                
                with col2:
                    price_col = st.selectbox(
                        "Price Column",
                        options=csv_columns,
                        help="Column containing price per share"
                    )
                    
                    action_col = st.selectbox(
                        "Action Column",
                        options=csv_columns,
                        help="Column containing Buy/Sell indicator"
                    )
                
                # Option to handle duplicates
                skip_dupes = st.checkbox(
                    "Skip duplicate trades",
                    value=True,
                    help="If checked, trades that already exist in the database will be skipped"
                )
                
                if st.button("Import Trades", use_container_width=True, type="primary"):
                    column_mapping = {
                        'ticker': ticker_col,
                        'date': date_col,
                        'quantity': quantity_col,
                        'price': price_col,
                        'action': action_col
                    }
                    
                    try:
                        db = get_db_session()
                        successful, failed, duplicates = import_trades_from_csv(
                            db, df, column_mapping, skip_duplicates=skip_dupes
                        )
                        db.close()
                        
                        if successful > 0:
                            st.success(f"Imported {successful} trades")
                            # Clear cache so new trades show up
                            st.cache_data.clear()
                        if duplicates > 0:
                            st.info(f"{duplicates} duplicate trades were skipped")
                        if failed > 0:
                            st.warning(f"{failed} trades could not be imported (check data format)")
                        if successful == 0 and duplicates == 0 and failed == 0:
                            st.info("No valid trades found in CSV. Check that your Action column contains Buy/Sell transactions.")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error importing trades: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Tab 3: Tax Lots View
    with tab3:
        st.subheader("Tax Lot Details")
        
        try:
            # Use cached portfolio data
            summary = get_portfolio_data()
            
            if summary is None:
                st.error("Error loading portfolio data.")
                return
            
            if not summary['positions']:
                st.info("No tax lots found. Add some trades first!")
            else:
                # Filter by ticker
                tickers = ["All"] + [p['ticker'] for p in summary['positions']]
                selected_ticker = st.selectbox("Filter by Ticker", tickers)
                
                for position in summary['positions']:
                    if selected_ticker != "All" and position['ticker'] != selected_ticker:
                        continue
                    
                    with st.expander(
                        f"{position['ticker']} — {position['total_shares']:.4f} shares",
                        expanded=True,
                    ):
                        # Position summary metrics
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        
                        with m_col1:
                            st.metric("Current Price", format_currency(position['current_price']))
                        with m_col2:
                            st.metric("Market Value", format_currency(position['current_value']))
                        with m_col3:
                            st.metric("Unrealized Gain", format_currency(position['unrealized_gain']))
                        with m_col4:
                            st.metric("Gain %", format_percentage(position['unrealized_gain_pct']))
                        
                        # Tax lots table
                        st.markdown("**Individual Tax Lots:**")
                        
                        lots_data = []
                        for lot in position["tax_lots"]:
                            if lot["is_long_term"]:
                                status_label = "LT"
                            elif lot["is_near_long_term"]:
                                status_label = "Near LT"
                            else:
                                status_label = "ST"
                            
                            current_price = position['current_price']
                            lot_gain = (current_price - lot['cost_basis']) * lot['shares']
                            total_cost = lot['cost_basis'] * lot['shares']
                            
                            lots_data.append({
                                "Status": status_label,
                                "Purchase Date": lot['purchase_date'],
                                "Shares": lot['shares'],
                                "Cost Basis": format_currency(lot['cost_basis']),
                                "Total Cost": format_currency(total_cost),
                                "Current Value": format_currency(lot['shares'] * current_price),
                                "Gain/Loss": format_currency(lot_gain),
                                "Holding Period": f"{lot['holding_days']} days",
                                "Tax Status": lot['tax_status'],
                                "Days to LT": lot['days_until_long_term'] if not lot['is_long_term'] else "—"
                            })
                        
                        st.dataframe(
                            _gft_tabular_styler(pd.DataFrame(lots_data)),
                            use_container_width=True,
                            hide_index=True,
                        )
                        
                        # Alert for lots near long-term
                        if position['lots_near_long_term']:
                            st.warning(
                                f"{len(position['lots_near_long_term'])} lot(s) approaching long-term status. "
                                "Consider holding to qualify for lower tax rates."
                            )
                        
        except Exception as e:
            st.error(f"Error loading tax lots: {str(e)}")
    
    st.markdown("---")
    
    # Recent trades section
    st.subheader("Recent trades")
    try:
        db: Session = get_db_session()
        recent_trades = db.query(Trades).order_by(Trades.trade_date.desc()).limit(20).all()
        db.close()
        
        if recent_trades:
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    "ID": trade.id,
                    "Ticker": trade.ticker,
                    "Date": trade.trade_date.strftime("%Y-%m-%d"),
                    "Shares": float(trade.shares),
                    "Price": format_currency(trade.price_per_share),
                    "Type": trade.trade_type.value,
                    "Total": format_currency(trade.shares * trade.price_per_share)
                })
            
            st.dataframe(
                _gft_tabular_styler(pd.DataFrame(trades_data)),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No trades recorded yet. Add your first trade using the form above.")
            
    except Exception as e:
        st.error(f"Error loading trades: {str(e)}")


def create_bloomberg_chart(df: pd.DataFrame, ticker: str, insider_transactions=None):
    """
    Create a Bloomberg-style dual chart with candlestick and RSI.
    Optionally overlay insider buy/sell markers (small) on the price panel.
    insider_transactions: list of dicts with 'date', 'transaction' (Buy/Sale), 'shares', 'value', 'name', etc.
    """
    # Bloomberg Dark color palette
    COLORS = {
        'background': '#0d1117',
        'grid': '#21262d',
        'text': '#c9d1d9',
        'gain': '#00ff41',      # Neon green
        'loss': '#ff073a',       # Neon red
        'sma_50': '#58a6ff',     # Blue
        'sma_200': '#f78166',    # Orange
        'bb_fill': 'rgba(88, 166, 255, 0.1)',
        'bb_line': '#8b949e',
        'rsi_line': '#d2a8ff',   # Purple
        'overbought': '#ff6b6b',
        'oversold': '#69db7c',
        'volume_up': 'rgba(0, 255, 65, 0.5)',
        'volume_down': 'rgba(255, 7, 58, 0.5)',
    }
    
    # Create subplot figure with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Action', 'RSI (14)')
    )
    
    # === TOP CHART: Candlestick + Indicators ===
    
    # Bollinger Bands fill
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            mode='lines',
            line=dict(width=1, color=COLORS['bb_line'], dash='dot'),
            name='BB Upper',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            mode='lines',
            line=dict(width=1, color=COLORS['bb_line'], dash='dot'),
            fill='tonexty',
            fillcolor=COLORS['bb_fill'],
            name='BB Lower',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=COLORS['gain'],
            decreasing_line_color=COLORS['loss'],
            increasing_fillcolor=COLORS['gain'],
            decreasing_fillcolor=COLORS['loss'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # 50-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            line=dict(width=2, color=COLORS['sma_50']),
            name='SMA 50'
        ),
        row=1, col=1
    )
    
    # 200-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            mode='lines',
            line=dict(width=2, color=COLORS['sma_200']),
            name='SMA 200'
        ),
        row=1, col=1
    )
    
    # Add signal markers
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    golden_cross = df[df['Signal'] == 'GOLDEN CROSS']
    death_cross = df[df['Signal'] == 'DEATH CROSS']
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color=COLORS['gain'],
                    line=dict(width=1, color='white')
                ),
                name='BUY Signal',
                hovertext='BUY'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color=COLORS['loss'],
                    line=dict(width=1, color='white')
                ),
                name='SELL Signal',
                hovertext='SELL'
            ),
            row=1, col=1
        )
    
    if not golden_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=golden_cross.index,
                y=golden_cross['SMA_50'],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=16,
                    color='#ffd700',
                    line=dict(width=1, color='white')
                ),
                name='Golden Cross',
                hovertext='GOLDEN CROSS'
            ),
            row=1, col=1
        )
    
    if not death_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=death_cross.index,
                y=death_cross['SMA_50'],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=14,
                    color=COLORS['loss'],
                    line=dict(width=2, color=COLORS['loss'])
                ),
                name='Death Cross',
                hovertext='DEATH CROSS'
            ),
            row=1, col=1
        )

    # Insider buy/sell markers (small, cap 20)
    if insider_transactions and len(insider_transactions) > 0:
        try:
            date_to_idx = {}
            for i in range(len(df)):
                ts = df.index[i]
                d = ts.date() if hasattr(ts, "date") and callable(getattr(ts, "date")) else (ts if isinstance(ts, date) else None)
                if d is not None:
                    date_to_idx.setdefault(d, []).append(i)
            limited = insider_transactions[:20]
            buy_x, buy_y, buy_hover = [], [], []
            sell_x, sell_y, sell_hover = [], [], []
            for t in limited:
                d = t.get("date")
                if d is None:
                    continue
                d = d.date() if hasattr(d, "date") and callable(getattr(d, "date")) else d
                if d not in date_to_idx:
                    continue
                idx = date_to_idx[d][0]
                ts = df.index[idx]
                low_val = df["Low"].iloc[idx]
                high_val = df["High"].iloc[idx]
                name = t.get("name") or ""
                sh = t.get("shares") or 0
                h = f"{name} {t.get('transaction', '')} {sh} shares"
                if t.get("transaction") == "Buy":
                    buy_x.append(ts)
                    buy_y.append(float(low_val) * 0.97)
                    buy_hover.append(h)
                else:
                    sell_x.append(ts)
                    sell_y.append(float(high_val) * 1.02)
                    sell_hover.append(h)
            if buy_x:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x, y=buy_y, mode="markers",
                        marker=dict(symbol="circle", size=8, color=COLORS["gain"], line=dict(width=1, color="white")),
                        name="Insider buy", hovertext=buy_hover,
                    ),
                    row=1, col=1,
                )
            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x, y=sell_y, mode="markers",
                        marker=dict(symbol="circle", size=8, color=COLORS["loss"], line=dict(width=1, color="white")),
                        name="Insider sell", hovertext=sell_hover,
                    ),
                    row=1, col=1,
                )
        except Exception:
            pass

    # === BOTTOM CHART: RSI ===
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI_14'],
            mode='lines',
            line=dict(width=2, color=COLORS['rsi_line']),
            name='RSI',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Overbought line (70)
    fig.add_hline(
        y=70, 
        line_dash='dash', 
        line_color=COLORS['overbought'],
        line_width=1,
        row=2, col=1,
        annotation_text='Overbought (70)',
        annotation_position='right'
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30, 
        line_dash='dash', 
        line_color=COLORS['oversold'],
        line_width=1,
        row=2, col=1,
        annotation_text='Oversold (30)',
        annotation_position='right'
    )
    
    # Middle line (50)
    fig.add_hline(
        y=50, 
        line_dash='dot', 
        line_color=COLORS['grid'],
        line_width=1,
        row=2, col=1
    )
    
    # Add RSI fill zones
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor='rgba(255, 107, 107, 0.1)',
        line_width=0,
        row=2, col=1
    )
    
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor='rgba(105, 219, 124, 0.1)',
        line_width=0,
        row=2, col=1
    )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> Technical Analysis',
            font=dict(size=24, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Monaco, monospace'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        height=800,
        margin=dict(l=60, r=60, t=100, b=60),
        xaxis_rangeslider_visible=False
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    # Price panel y-axis: scale to visible data so dropdown/custom range views are readable
    y_min = float(df['Low'].min()) * 0.99
    y_max = float(df['High'].max()) * 1.01
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
    # RSI y-axis range
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig


def create_valuation_chart(valuation_data: dict) -> go.Figure:
    """
    Create a stacked panel valuation chart with P/E Multiple and Revenue Growth.
    
    Top panel: P/E Multiple over time
    Bottom panel: Revenue Growth YoY % (historical + projected)
    """
    # Color palette matching Bloomberg dark theme
    COLORS = {
        'background': '#0d1117',
        'grid': '#21262d',
        'text': '#c9d1d9',
        'pe_line': '#00D4AA',         # Teal for P/E
        'revenue_line': '#FF6B6B',     # Coral for revenue
        'projected_line': '#FF6B6B',   # Same color, dashed
        'zero_line': '#8b949e',
        'positive_zone': 'rgba(0, 212, 170, 0.1)',
        'negative_zone': 'rgba(255, 107, 107, 0.1)',
    }
    
    ticker = valuation_data['ticker']
    pe_df = valuation_data['pe_data']
    revenue_df = valuation_data['revenue_data']
    
    # Check if revenue data has QoQ growth
    has_qoq_data = False
    if not revenue_df.empty and 'growth_type' in revenue_df.columns:
        has_qoq_data = (revenue_df['growth_type'] == 'qoq').any()
    
    revenue_title = '<b>Revenue Growth %</b> <span style="font-size:12px;color:#8b949e">(YoY where available, QoQ otherwise)</span>' if has_qoq_data else '<b>Revenue Growth YoY %</b> <span style="font-size:12px;color:#8b949e">(Quarterly)</span>'
    
    # Determine P/E data source for title
    pe_source = valuation_data.get('pe_source', 'unknown')
    pe_source_labels = {
        'alpha_vantage': 'via Alpha Vantage',
        'finnhub': 'via Finnhub',
        'fmp': 'via Financial Modeling Prep',
        'calculated': 'Calculated from EPS',
        'estimated': 'Estimated',
        'unknown': ''
    }
    pe_source_label = pe_source_labels.get(pe_source, '')
    pe_title = f'<b>P/E Multiple</b> <span style="font-size:12px;color:#8b949e">(TTM{" - " + pe_source_label if pe_source_label else ""})</span>'
    
    # Create subplot figure with shared x-axis concept (but separate time scales)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            pe_title,
            revenue_title
        )
    )
    
    # === TOP PANEL: P/E Multiple ===
    if not pe_df.empty and len(pe_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=pe_df['date'],
                y=pe_df['pe'],
                mode='lines+markers',
                line=dict(width=3, color=COLORS['pe_line']),
                marker=dict(size=8, color=COLORS['pe_line'], 
                           line=dict(width=1, color='white')),
                name='P/E Ratio (quarter-end)',
                hovertemplate='<b>P/E:</b> %{y:.1f}x<br><b>Date:</b> %{x|%b %Y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add "Today" point so current P/E is visible and distinct from last quarterly point
        if valuation_data['current_pe']:
            from datetime import datetime as _dt_now
            now = _dt_now.now()
            current_pe = valuation_data['current_pe']
            last_qtr_date = pe_df['date'].iloc[-1]
            last_qtr_pe = pe_df['pe'].iloc[-1]
            # Connector line from last quarter to today (dashed) if they differ
            if abs(last_qtr_pe - current_pe) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=[last_qtr_date, now],
                        y=[last_qtr_pe, current_pe],
                        mode='lines',
                        line=dict(width=2, color=COLORS['pe_line'], dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            fig.add_trace(
                go.Scatter(
                    x=[now],
                    y=[current_pe],
                    mode='markers',
                    marker=dict(size=12, color=COLORS['pe_line'], symbol='diamond',
                               line=dict(width=2, color='white')),
                    name='Today (live)',
                    hovertemplate='<b>Today (live) P/E:</b> %{y:.1f}x<extra></extra>'
                ),
                row=1, col=1
            )
            fig.add_annotation(
                x=now,
                y=current_pe,
                text=f"Today: {current_pe:.1f}x",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLORS['pe_line'],
                font=dict(size=11, color=COLORS['text']),
                bgcolor='rgba(0,0,0,0.7)',
                borderpad=4,
                ax=40,
                ay=0,
                row=1, col=1
            )
        
        # Add forward P/E reference line (show even when Current P/E is N/A, e.g. SNOW)
        if valuation_data.get('forward_pe'):
            current_pe_val = valuation_data.get('current_pe')
            if current_pe_val is None or abs(valuation_data['forward_pe'] - current_pe_val) > 1:
                fig.add_hline(
                    y=valuation_data['forward_pe'],
                    line_dash='dot',
                    line_color='#58a6ff',
                    line_width=1,
                    annotation_text=f"Forward P/E: {valuation_data['forward_pe']:.1f}x",
                    annotation_position='right',
                    annotation_font_color='#58a6ff',
                    row=1, col=1
                )
    elif valuation_data['current_pe']:
        # Show current P/E as a single point if we have it but no history
        import datetime
        now = datetime.datetime.now()
        fig.add_trace(
            go.Scatter(
                x=[now],
                y=[valuation_data['current_pe']],
                mode='markers+text',
                marker=dict(size=16, color=COLORS['pe_line'],
                           line=dict(width=2, color='white')),
                text=[f"{valuation_data['current_pe']:.1f}x"],
                textposition='top center',
                textfont=dict(size=14, color=COLORS['text']),
                name='Current P/E',
                hovertemplate='<b>Current P/E:</b> %{y:.1f}x<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add forward P/E as reference line
        if valuation_data['forward_pe']:
            fig.add_hline(
                y=valuation_data['forward_pe'],
                line_dash='dot',
                line_color='#58a6ff',
                line_width=2,
                annotation_text=f"Forward P/E: {valuation_data['forward_pe']:.1f}x",
                annotation_position='right',
                annotation_font_color='#58a6ff',
                row=1, col=1
            )
    else:
        # Show placeholder if no P/E data at all
        fig.add_annotation(
            text="P/E data unavailable (negative earnings)",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['text']),
            row=1, col=1
        )
    
    # === BOTTOM PANEL: Revenue Growth ===
    if not revenue_df.empty and len(revenue_df) > 0:
        # Separate historical and projected data
        historical = revenue_df[revenue_df['is_projected'] == False].copy()
        projected = revenue_df[revenue_df['is_projected'] == True].copy()
        
        # Check if we have mixed growth types (YoY and QoQ)
        has_qoq = 'growth_type' in historical.columns and (historical['growth_type'] == 'qoq').any()
        growth_label = 'Revenue Growth (YoY/QoQ)' if has_qoq else 'Revenue Growth (YoY)'
        
        # Historical revenue growth (solid line)
        if not historical.empty:
            # Create hover text that shows growth type
            hover_texts = []
            for _, row in historical.iterrows():
                growth_type = row.get('growth_type', 'yoy').upper()
                hover_texts.append(f"<b>{growth_type}:</b> {row['growth']:+.1f}%<br><b>Quarter:</b> {row['date'].strftime('%b %Y')}")
            
            fig.add_trace(
                go.Scatter(
                    x=historical['date'],
                    y=historical['growth'],
                    mode='lines+markers',
                    line=dict(width=3, color=COLORS['revenue_line']),
                    marker=dict(size=8, color=COLORS['revenue_line'],
                               line=dict(width=1, color='white')),
                    name=growth_label,
                    hovertext=hover_texts,
                    hoverinfo='text'
                ),
                row=2, col=1
            )
        
        # Projected revenue growth (dashed line)
        if not projected.empty:
            # Connect projected to last historical point
            if not historical.empty:
                connection_x = [historical['date'].iloc[-1], projected['date'].iloc[0]]
                connection_y = [historical['growth'].iloc[-1], projected['growth'].iloc[0]]
                
                fig.add_trace(
                    go.Scatter(
                        x=connection_x,
                        y=connection_y,
                        mode='lines',
                        line=dict(width=2, color=COLORS['projected_line'], dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
            
            fig.add_trace(
                go.Scatter(
                    x=projected['date'],
                    y=projected['growth'],
                    mode='lines+markers',
                    line=dict(width=3, color=COLORS['projected_line'], dash='dash'),
                    marker=dict(size=8, color=COLORS['projected_line'],
                               symbol='diamond',
                               line=dict(width=1, color='white')),
                    name='Revenue Growth (Projected)',
                    hovertemplate='<b>Projected Growth:</b> %{y:+.1f}%<br><b>Period:</b> %{x|%b %Y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add zero line for reference
        fig.add_hline(
            y=0,
            line_dash='solid',
            line_color=COLORS['zero_line'],
            line_width=1,
            row=2, col=1
        )
        
        # Add positive/negative zones
        all_growth = revenue_df['growth'].dropna()
        if not all_growth.empty:
            max_growth = max(all_growth.max(), 10)
            min_growth = min(all_growth.min(), -10)
            
            # Positive zone (above 0)
            fig.add_hrect(
                y0=0, y1=max_growth + 5,
                fillcolor=COLORS['positive_zone'],
                line_width=0,
                row=2, col=1
            )
            
            # Negative zone (below 0)
            fig.add_hrect(
                y0=min_growth - 5, y1=0,
                fillcolor=COLORS['negative_zone'],
                line_width=0,
                row=2, col=1
            )
    else:
        # Show placeholder if no revenue data
        fig.add_annotation(
            text="Revenue data unavailable",
            xref="x2 domain", yref="y2 domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['text']),
            row=2, col=1
        )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> Valuation Analysis',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Monaco, monospace'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=620,
        margin=dict(l=64, r=64, t=90, b=56),
        hovermode='x unified',
        dragmode='zoom',
    )
    
    # Apply time range from toggle so both panels show the same x-axis window
    requested_years = valuation_data.get('requested_years', 2)
    if requested_years:
        from datetime import datetime as _dt_now, timedelta as _td
        range_end = _dt_now.now()
        range_start = range_end - _td(days=int(requested_years * 365.25))
        fig.update_xaxes(
            range=[range_start, range_end],
            row=1, col=1
        )
        fig.update_xaxes(
            range=[range_start, range_end],
            row=2, col=1
        )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10),
        tickformat='%b %Y'
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    # Y-axis formatting
    fig.update_yaxes(title_text="P/E Multiple", ticksuffix="x", row=1, col=1)
    fig.update_yaxes(title_text="YoY Growth", ticksuffix="%", row=2, col=1)
    
    # Ensure P/E panel y-axis includes Forward P/E line when it sits below/above the data
    if not pe_df.empty and len(pe_df) > 0:
        y_vals = list(pe_df['pe'])
        if valuation_data.get('current_pe'):
            y_vals.append(valuation_data['current_pe'])
        if valuation_data.get('forward_pe'):
            y_vals.append(valuation_data['forward_pe'])
        y_min = max(0, min(y_vals) - 5)
        y_max = max(y_vals) + 10
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
    # Update subplot title colors
    fig.update_annotations(font=dict(color=COLORS['text']))
    
    return fig


def create_tradingview_chart(df: pd.DataFrame, ticker: str, timeframe: str = '1W') -> go.Figure:
    """
    Create a TradingView-style chart with three panels.
    
    Top panel: Momentum oscillator (-50 to +50)
    Middle panel: Candlestick with supply/demand zones and signals
    Bottom panel: Secondary oscillator (-40 to +40)
    
    Args:
        df: DataFrame with OHLCV data and TradingView signals calculated
        ticker: Stock ticker symbol
        timeframe: Display timeframe label (e.g., '1W', '1D')
        
    Returns:
        Plotly figure object
    """
    # TradingView-inspired color palette
    COLORS = {
        'background': '#131722',      # TradingView dark background
        'grid': '#1e222d',
        'text': '#d1d4dc',
        'candle_up': '#26a69a',       # Teal/cyan for bullish
        'candle_down': '#ef5350',     # Coral/pink for bearish (lighter than red)
        'candle_up_body': '#26a69a',
        'candle_down_body': '#ef5350',
        'zone_upper': 'rgba(233, 30, 99, 0.15)',    # Pink zones
        'zone_lower': 'rgba(233, 30, 99, 0.15)',
        'zone_line': 'rgba(233, 30, 99, 0.5)',
        'support_line': '#ffd700',    # Gold support line
        'momentum_pos': '#26a69a',    # Teal when positive
        'momentum_neg': '#ef5350',    # Pink when negative
        'oscillator_pos': '#26a69a',
        'oscillator_neg': '#ef5350',
        'signal_buy': '#00e5ff',      # Cyan diamond for buy
        'signal_sell': '#ff4081',     # Pink diamond for sell
    }
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.15, 0.65, 0.20],
        subplot_titles=('', f'{ticker} · {timeframe}', '')
    )
    
    # === TOP PANEL: Momentum Oscillator ===
    if 'TV_Momentum' in df.columns:
        momentum = df['TV_Momentum'].fillna(0)
        
        # Create color array based on positive/negative values
        colors = [COLORS['momentum_pos'] if v >= 0 else COLORS['momentum_neg'] for v in momentum]
        
        # Area fill for momentum
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=momentum,
                mode='lines',
                fill='tozeroy',
                line=dict(width=1, color=COLORS['momentum_pos']),
                fillcolor='rgba(38, 166, 154, 0.3)',
                name='Momentum',
                showlegend=False,
                hovertemplate='Momentum: %{y:.1f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add negative fill separately
        momentum_neg = momentum.where(momentum < 0, 0)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=momentum_neg,
                mode='lines',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(239, 83, 80, 0.3)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash='solid', line_color=COLORS['grid'], line_width=1, row=1, col=1)
        
        # Add current value annotation
        current_momentum = momentum.iloc[-1]
        fig.add_annotation(
            x=df.index[-1],
            y=current_momentum,
            text=f"{current_momentum:.1f}",
            showarrow=False,
            font=dict(size=10, color=COLORS['momentum_pos'] if current_momentum >= 0 else COLORS['momentum_neg']),
            bgcolor=COLORS['background'],
            xanchor='left',
            xshift=5,
            row=1, col=1
        )
        
        # Add sell signals on momentum panel (pink diamonds at peaks)
        if 'TV_Signal' in df.columns:
            sell_signals = df[(df['TV_Signal'] == 'SELL') & (df['TV_Momentum'] > 15)]
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['TV_Momentum'],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=COLORS['signal_sell'],
                            line=dict(width=1, color='white')
                        ),
                        name='Sell Signal (Momentum)',
                        showlegend=False,
                        hovertemplate='SELL Signal<extra></extra>'
                    ),
                    row=1, col=1
                )
    
    # === MIDDLE PANEL: Candlestick with Zones ===
    
    # Supply zone (upper pink band)
    if 'TV_Zone_Outer_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Outer_Upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Inner_Upper'],
                mode='lines',
                line=dict(width=1, color=COLORS['zone_line']),
                fill='tonexty',
                fillcolor=COLORS['zone_upper'],
                name='Supply Zone',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Demand zone (lower pink band)
    if 'TV_Zone_Inner_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Inner_Lower'],
                mode='lines',
                line=dict(width=1, color=COLORS['zone_line']),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Outer_Lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=COLORS['zone_lower'],
                name='Demand Zone',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=COLORS['candle_up'],
            decreasing_line_color=COLORS['candle_down'],
            increasing_fillcolor=COLORS['candle_up_body'],
            decreasing_fillcolor=COLORS['candle_down_body'],
            name='Price',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Gold support line (50 SMA)
    if 'TV_Support' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Support'],
                mode='lines',
                line=dict(width=2, color=COLORS['support_line']),
                name='Support (SMA 50)',
                hovertemplate='Support: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add price annotation
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_color = COLORS['candle_up'] if current_price >= prev_price else COLORS['candle_down']
    
    fig.add_annotation(
        x=df.index[-1],
        y=current_price,
        text=f"${current_price:.2f}",
        showarrow=False,
        font=dict(size=11, color=price_color, family='Monaco, monospace'),
        bgcolor=price_color,
        bordercolor=price_color,
        borderwidth=1,
        borderpad=3,
        xanchor='left',
        xshift=10,
        row=2, col=1
    )
    
    # Buy signals on price chart (cyan diamonds)
    if 'TV_Signal' in df.columns:
        buy_signals = df[df['TV_Signal'] == 'BUY']
        sell_signals = df[df['TV_Signal'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Low'] * 0.98,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=COLORS['signal_buy'],
                        line=dict(width=1, color='white')
                    ),
                    name='Buy Signal',
                    hovertemplate='BUY Signal<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['High'] * 1.02,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=COLORS['signal_sell'],
                        line=dict(width=1, color='white')
                    ),
                    name='Sell Signal',
                    hovertemplate='SELL Signal<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # === BOTTOM PANEL: Secondary Oscillator ===
    if 'TV_Oscillator' in df.columns:
        oscillator = df['TV_Oscillator'].fillna(0)
        
        # Area fill for oscillator
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=oscillator,
                mode='lines',
                fill='tozeroy',
                line=dict(width=1, color=COLORS['oscillator_pos']),
                fillcolor='rgba(38, 166, 154, 0.3)',
                name='Oscillator',
                showlegend=False,
                hovertemplate='Oscillator: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Negative fill
        oscillator_neg = oscillator.where(oscillator < 0, 0)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=oscillator_neg,
                mode='lines',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(239, 83, 80, 0.3)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash='solid', line_color=COLORS['grid'], line_width=1, row=3, col=1)
        
        # Current value annotation
        current_osc = oscillator.iloc[-1]
        fig.add_annotation(
            x=df.index[-1],
            y=current_osc,
            text=f"{current_osc:.1f}",
            showarrow=False,
            font=dict(size=10, color=COLORS['oscillator_pos'] if current_osc >= 0 else COLORS['oscillator_neg']),
            bgcolor=COLORS['background'],
            xanchor='left',
            xshift=5,
            row=3, col=1
        )
        
        # Add buy signals on oscillator panel (cyan diamonds at troughs)
        if 'TV_Signal' in df.columns:
            buy_signals = df[(df['TV_Signal'] == 'BUY') & (df['TV_Oscillator'] < -10)]
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['TV_Oscillator'],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=COLORS['signal_buy'],
                            line=dict(width=1, color='white')
                        ),
                        name='Buy Signal (Oscillator)',
                        showlegend=False,
                        hovertemplate='BUY Signal<extra></extra>'
                    ),
                    row=3, col=1
                )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> TradingView Analysis · {timeframe}',
            font=dict(size=20, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Trebuchet MS, sans-serif'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=10),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        height=600,
        margin=dict(l=60, r=80, t=80, b=40),
        xaxis_rangeslider_visible=False
    )
    
    # Update all x-axes
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=False,  # Hide labels except bottom
        row=1, col=1
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=False,
        row=2, col=1
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=True,
        row=3, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        range=[-50, 50],
        title_text='',
        row=1, col=1
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        tickprefix='$',
        title_text='',
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        range=[-40, 40],
        title_text='',
        row=3, col=1
    )
    
    # Update subplot title styling
    fig.update_annotations(font=dict(color=COLORS['text'], size=14))
    
    return fig


def get_signal_badge(signal: str) -> str:
    """Return HTML badge for signal display."""
    if signal == 'BUY':
        return '<span style="background-color: #00ff41; color: black; padding: 4px 12px; border-radius: 4px; font-weight: bold;">BUY</span>'
    elif signal == 'SELL':
        return '<span style="background-color: #ff073a; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">SELL</span>'
    elif signal == 'GOLDEN CROSS':
        return '<span style="background-color: #ffd700; color: black; padding: 4px 12px; border-radius: 4px; font-weight: bold;">GOLDEN CROSS</span>'
    elif signal == 'DEATH CROSS':
        return '<span style="background-color: #8b0000; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">DEATH CROSS</span>'
    else:
        return '<span style="background-color: #6c757d; color: white; padding: 4px 12px; border-radius: 4px;">NEUTRAL</span>'


def market_analysis_page():
    """Display market analysis page with technical indicators and signals."""
    st.title("Market Analysis")
    
    # Custom CSS for Bloomberg dark theme
    st.markdown("""
        <style>
        .signal-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }
        .metric-highlight {
            font-size: 28px;
            font-weight: bold;
            font-family: 'Monaco', monospace;
        }
        .gain { color: #00ff41; }
        .loss { color: #ff073a; }
        .neutral { color: #8b949e; }
        .indicator-label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === TICKER SEARCH ===
    col_search, col_refresh = st.columns([4, 1])
    
    with col_search:
        ticker_input = st.text_input(
            "Search ticker",
            placeholder="Enter ticker symbol (e.g., AAPL, GOOGL, MSFT)",
            help="Enter a stock ticker to analyze",
            key="market_analysis_ticker",
        ).upper().strip()
    
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear cache", help="Clear cached data and refresh"):
            clear_cache()
            _cached_iv_term_structure.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    # === MAIN ANALYSIS VIEW ===
    if ticker_input:
        with st.spinner(f"Loading {ticker_input} data..."):
            df = fetch_ohlcv(ticker_input, period_years=50)
        
        if df is None or df.empty:
            st.error(f"Could not fetch data for '{ticker_input}'. Please check the ticker symbol.")
        else:
            # Calculate signals
            df = calculate_signals(df)
            summary = _cached_get_ticker_summary(ticker_input)
            
            if summary:
                # === GLOBAL ANCHOR HEADER ===
                st.markdown(f"### {ticker_input} Overview")
                head_col1, head_col2 = st.columns([4, 1])
                with head_col1:
                
                    m1, m2, m3, m4, m5 = st.columns(5)
                
                    with m1:
                        st.metric(
                            "Price",
                            f"${summary['current_price']:,.2f}",
                            f"{summary['daily_change_pct']:+.2f}%",
                            delta_color="normal" if summary['daily_change'] >= 0 else "inverse"
                        )
                
                    with m2:
                        rsi_val = summary["rsi"]
                        rsi_zone = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
                        st.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_zone, delta_color="off")
                
                    with m3:
                        st.metric("SMA 50", f"${summary['sma_50']:,.2f}" if summary['sma_50'] else "N/A")
                
                    with m4:
                        st.metric("SMA 200", f"${summary['sma_200']:,.2f}" if summary['sma_200'] else "N/A")
                
                    with m5:
                        st.metric("Trend", summary["trend"])
                
                with head_col2:
                    st.markdown('### Current signal')
                    st.markdown(get_signal_badge(summary['signal']), unsafe_allow_html=True)
                st.markdown("---")
                # === TABBED LAYOUT ===
                tab_tech, tab_val, tab_corp = st.tabs(["Technicals", "Valuation", "Corporate Activity"])

                with tab_tech:
                    # === DUAL CHART ===
                    st.markdown("### Technical chart")
                
                    # Trading days per year for tail slicing
                    TRADING_DAYS_PER_YEAR = 252
                    time_range_options = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '25Y', '50Y', 'Max', 'Custom']
                    date_range = st.selectbox(
                        "Time Range",
                        options=time_range_options,
                        index=3,
                        help="Select the time range to display"
                    )
                    chart_type = "line" if st.checkbox("Line chart (close only)", value=False, help="Show closing price as a line instead of candlesticks.") else "candlestick"
                    show_signals = st.checkbox(
                        "Show buy/sell signals",
                        value=True,
                        help="Show BUY/SELL and cross markers on the price chart.",
                    )
                    strong_signals_only = st.checkbox(
                        "Strong signals only",
                        value=False,
                        help="When on, only show BUY when RSI < 25 and SELL when RSI > 75 (fewer markers).",
                    )

                    if date_range == 'Custom':
                        from datetime import timedelta as _td
                        _today = date.today()
                        _default_start = _today - _td(days=365)
                        custom_col1, custom_col2 = st.columns(2)
                        with custom_col1:
                            custom_start = st.date_input("Start date", value=_default_start, key="tech_chart_start")
                        with custom_col2:
                            custom_end = st.date_input("End date", value=_today, key="tech_chart_end")
                        if custom_start and custom_end and custom_start <= custom_end:
                            df_display = df[(df.index.date >= custom_start) & (df.index.date <= custom_end)]
                            if df_display.empty:
                                st.warning("No data in the selected date range.")
                                df_display = df.tail(TRADING_DAYS_PER_YEAR)
                        else:
                            df_display = df.tail(TRADING_DAYS_PER_YEAR)
                    elif date_range == '3M':
                        df_display = df.tail(63)
                    elif date_range == '6M':
                        df_display = df.tail(126)
                    elif date_range == '1Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR)
                    elif date_range == '2Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 2)
                    elif date_range == '5Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 5)
                    elif date_range == '10Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 10)
                    elif date_range == '15Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 15)
                    elif date_range == '25Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 25)
                    elif date_range == '50Y':
                        df_display = df.tail(TRADING_DAYS_PER_YEAR * 50)
                    else:
                        df_display = df
                
                    # Edge-case captions: data shorter than requested; SMA 200 partial
                    if not df_display.empty and date_range not in ('Custom', 'Max'):
                        expected_days = {
                            '5Y': TRADING_DAYS_PER_YEAR * 5, '10Y': TRADING_DAYS_PER_YEAR * 10,
                            '15Y': TRADING_DAYS_PER_YEAR * 15, '25Y': TRADING_DAYS_PER_YEAR * 25,
                            '50Y': TRADING_DAYS_PER_YEAR * 50,
                        }
                        expected = expected_days.get(date_range)
                        if expected and len(df_display) < expected * 0.9:
                            first_ts = df_display.index.min()
                            first_str = first_ts.strftime("%Y-%m-%d") if hasattr(first_ts, "strftime") else str(first_ts)[:10]
                            st.caption(f"Data from {first_str} (all available).")
                    if len(df) < 200:
                        st.caption("SMA 200 is shown with partial data (fewer than 200 trading days).")
                
                    # Insider transactions for chart overlay (optional)
                    insider_list = []
                    try:
                        insider_list = _cached_fetch_insider_transactions(ticker_input)
                    except Exception:
                        pass
                    # TradingView Lightweight Charts: candlestick or line + volume + support lines + markers + RSI (zoom + double-click reset)
                    tech_data = df_to_technical_chart_data(df_display, strong_signals_only=strong_signals_only)
                    # Show markers when user wants any signals: all (show_signals) or strong-only (strong_signals_only)
                    markers_to_show = (tech_data["markers"] or None) if (show_signals or strong_signals_only) else None
                    chart_config = build_technical_chart_config(
                        ticker_input,
                        tech_data["candles"],
                        tech_data["volume"],
                        dark_theme=True,
                        rsi=tech_data["rsi"] or None,
                        sma_50=tech_data["sma_50"] or None,
                        sma_200=tech_data["sma_200"] or None,
                        bb_upper=tech_data["bb_upper"] or None,
                        bb_lower=tech_data["bb_lower"] or None,
                        markers=markers_to_show,
                        price_series_type=chart_type,
                    )
                    renderLightweightCharts(chart_config, key=f"technical_chart_{ticker_input}")

                    with st.expander("Chart guide — lines and signals", expanded=False):
                        st.markdown("""
                        Use **Line chart (close only)** to show closing price as a line instead of candlesticks.
                        **Show buy/sell signals** = all markers. **Strong signals only** = only stricter signals (BUY when RSI &lt; 25, SELL when RSI &gt; 75). You can use either or both.

                        **Price panel**
                        | Line / marker | Meaning |
                        |----------------|---------|
                        | **Blue line** | SMA 50 (50-day simple moving average) |
                        | **Orange line** | SMA 200 (200-day simple moving average) |
                        | **Gray dotted lines** | Bollinger Bands (upper and lower) |
                        | **Green ↑** | Buy signal (price &lt; lower band and RSI &lt; 35) |
                        | **Red ↓** | Sell signal (price &gt; upper band and RSI &gt; 65) |
                        | **Gold dot** | Golden cross (SMA 50 crosses above SMA 200) |
                        | **Dark red dot** | Death cross (SMA 50 crosses below SMA 200) |
                        | **Green/teal bars** | Volume (lower section) |

                        **RSI panel:** Purple line = RSI (14). Values above 70 = overbought; below 30 = oversold.
                        """)
                    # === RECENT SIGNALS TABLE ===
                    st.markdown("---")
                    st.markdown("### Recent signals")
                
                    recent_signals = df[df['Signal'] != ''].tail(10).copy()
                    if not recent_signals.empty:
                        recent_signals = recent_signals[['Close', 'RSI_14', 'SMA_50', 'SMA_200', 'Signal']].copy()
                        recent_signals.columns = ['Price', 'RSI', 'SMA 50', 'SMA 200', 'Signal']
                        recent_signals = recent_signals.round(2)
                        recent_signals.index = recent_signals.index.strftime('%Y-%m-%d')
                        recent_signals = recent_signals.iloc[::-1]  # Most recent first
                    
                        st.dataframe(
                            _gft_tabular_styler(recent_signals),
                            use_container_width=True,
                            column_config={
                                "Signal": st.column_config.TextColumn("Signal", width="medium"),
                                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                                "SMA 50": st.column_config.NumberColumn("SMA 50", format="$%.2f"),
                                "SMA 200": st.column_config.NumberColumn("SMA 200", format="$%.2f"),
                            },
                        )
                    else:
                        st.info("No trading signals generated in the selected period.")
                
                    # === TRADINGVIEW-STYLE CHART ===
                    st.markdown("---")
                    st.markdown("### TradingView-style analysis")
                    st.caption("Multi-panel momentum analysis with supply/demand zones and trading signals.")
                
                    # Timeframe selector for TradingView chart
                    tv_col1, tv_col2 = st.columns([1, 4])
                    with tv_col1:
                        tv_timeframe = st.selectbox(
                            "Timeframe",
                            options=['1W', '1D', '4H'],
                            index=0,
                            key="tv_timeframe",
                            help="Select chart timeframe"
                        )
                
                    # Determine data to display based on timeframe
                    # For weekly, resample daily data to weekly
                    if tv_timeframe == '1W':
                        # Resample to weekly data
                        df_tv = df.copy()
                        df_tv_resampled = df_tv.resample('W').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                        # Take last 52 weeks (1 year)
                        df_tv_display = df_tv_resampled.tail(52)
                        timeframe_label = '1W'
                    elif tv_timeframe == '4H':
                        # For 4H, use daily data but show more recent period
                        df_tv_display = df.tail(60)  # ~60 days
                        timeframe_label = '4H (Daily proxy)'
                    else:  # 1D
                        df_tv_display = df.tail(126)  # 6 months daily
                        timeframe_label = '1D'
                
                    # Use cached TradingView signals if fresh; otherwise compute and auto-save
                    df_tv_with_signals = load_tv_signals_from_cache(ticker_input, tv_timeframe)
                    if df_tv_with_signals is None:
                        df_tv_with_signals = calculate_tradingview_signals(df_tv_display)
                        save_tv_signals_to_cache(ticker_input, df_tv_with_signals, tv_timeframe)
                
                    # Create and display TradingView chart
                    tv_fig = create_tradingview_chart(df_tv_with_signals, ticker_input, timeframe_label)
                    st.plotly_chart(tv_fig, use_container_width=True, config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{ticker_input}_tradingview_{tv_timeframe}',
                            'height': 900,
                            'width': 1400,
                            'scale': 2
                        }
                    })
                
                    # Show current signal status
                    tv_signal_col1, tv_signal_col2, tv_signal_col3 = st.columns(3)
                
                    with tv_signal_col1:
                        current_momentum = df_tv_with_signals['TV_Momentum'].iloc[-1] if 'TV_Momentum' in df_tv_with_signals.columns else 0
                        st.metric("Momentum", f"{current_momentum:.1f}")
                
                    with tv_signal_col2:
                        current_osc = df_tv_with_signals['TV_Oscillator'].iloc[-1] if 'TV_Oscillator' in df_tv_with_signals.columns else 0
                        st.metric("Oscillator", f"{current_osc:.1f}")
                
                    with tv_signal_col3:
                        # Get latest TV signal
                        tv_signals = df_tv_with_signals[df_tv_with_signals['TV_Signal'] != '']
                        if not tv_signals.empty:
                            latest_tv_signal = tv_signals['TV_Signal'].iloc[-1]
                            latest_tv_strength = tv_signals['TV_Signal_Strength'].iloc[-1]
                            st.metric("Latest signal", f"{latest_tv_signal} ({latest_tv_strength})")
                        else:
                            st.metric("Latest signal", "None")
                
                    # TradingView chart interpretation guide
                    with st.expander("TradingView chart guide"):
                        st.markdown("""
                        **Top Panel - Momentum Oscillator (-50 to +50):**
                        - Combines Stochastic RSI and Rate of Change
                        - **Cyan fill**: Positive momentum (bullish)
                        - **Pink fill**: Negative momentum (bearish)
                        - **Pink diamonds**: Potential sell signals at peaks
                    
                        **Middle Panel - Price Action:**
                        - **Cyan/Teal candles**: Bullish (close > open)
                        - **Pink/Red candles**: Bearish (close < open)
                        - **Pink shaded zones**: Supply (upper) and Demand (lower) zones
                        - **Gold line**: 50-period moving average support
                        - **Cyan diamonds**: Buy signals
                        - **Pink diamonds**: Sell signals
                    
                        **Bottom Panel - Secondary Oscillator (-40 to +40):**
                        - MACD-style momentum histogram
                        - Confirms signals from top panel
                        - **Cyan diamonds**: Buy confirmation
                    
                        **Signal Logic:**
                        | Condition | Signal Type |
                        |-----------|-------------|
                        | Price at demand zone + both oscillators negative turning up | **STRONG BUY** |
                        | Price near lower band + momentum turning up | **MODERATE BUY** |
                        | Price at supply zone + both oscillators positive turning down | **STRONG SELL** |
                        | Price near upper band + momentum turning down | **MODERATE SELL** |
                    
                        **"Triple Blue" Alignment:**
                        When price, momentum, and oscillator are all positive/bullish simultaneously, this indicates strong upward momentum.
                        """)
                
                    # Save button for TradingView signals
                    tv_cache_status = check_tv_signals_in_cache(ticker_input, tv_timeframe)
                
                    tv_save_col1, tv_save_col2 = st.columns([3, 1])
                    with tv_save_col1:
                        if tv_cache_status['has_data']:
                            tv_note = "Fresh" if tv_cache_status["is_fresh"] else "Stale"
                            cache_time = (
                                tv_cache_status["timestamp"].strftime("%Y-%m-%d %H:%M")
                                if tv_cache_status["timestamp"]
                                else "Unknown"
                            )
                            st.caption(
                                f"**{tv_note}:** **{ticker_input}** TradingView signals cached (saved: {cache_time})"
                            )
                        else:
                            st.caption(f"**{ticker_input}** TradingView signals not saved yet")
                
                    with tv_save_col2:
                        if tv_cache_status['is_fresh']:
                            st.success("Cached")
                        else:
                            if st.button("Save TV signals", key=f"save_tv_{ticker_input}_{tv_timeframe}", help="Save TradingView signals to cache"):
                                if save_tv_signals_to_cache(ticker_input, df_tv_with_signals, tv_timeframe):
                                    st.success(f"Saved TradingView signals for {ticker_input}")
                                    st.rerun()
                                else:
                                    st.error("Failed to save TradingView signals.")
                
                with tab_val:
                    # === VALUATION CHART ===
                    st.markdown("---")
                    st.markdown("### Valuation analysis")
                    st.caption("P/E Multiple and Revenue Growth trends help assess if a stock is fairly valued relative to its growth.")
                
                    # Time range selector and optional refetch for valuation chart
                    val_time_col1, val_time_col2 = st.columns([1, 4])
                    with val_time_col1:
                        valuation_years = st.selectbox(
                            "Time Range",
                            options=[2, 5, 10],
                            index=0,
                            format_func=lambda x: f"{x} Years",
                            key="valuation_time_range",
                            help="Select historical time range for valuation analysis"
                        )
                        # Force refetch from API (skip DB) to get best-available P/E from all sources
                        refetch_key = f"valuation_refetch_{ticker_input}"
                        if st.button("Refetch from API", key=refetch_key, help="Skip database and fetch fresh P/E and revenue from APIs (best available data)"):
                            st.session_state["valuation_skip_db"] = True
                
                    skip_db = st.session_state.pop("valuation_skip_db", False)
                    with st.spinner(f"Loading {valuation_years}-year valuation data..." + (" (from API)" if skip_db else "")):
                        valuation_data = _cached_get_valuation_chart_data(ticker_input, valuation_years, skip_db)
                
                    if valuation_data:
                        # Show key valuation metrics
                        val_col1, val_col2, val_col3, val_col4, val_col5 = st.columns(5)
                    
                        with val_col1:
                            pe_val = valuation_data['current_pe']
                            if pe_val:
                                pe_band = "lower" if pe_val < 20 else "mid" if pe_val < 35 else "elevated"
                                st.metric("Current P/E", f"{pe_val:.1f}x", delta=pe_band, delta_color="off")
                            else:
                                st.metric("Current P/E", "N/A")
                    
                        with val_col2:
                            fwd_pe = valuation_data['forward_pe']
                            if fwd_pe:
                                st.metric("Forward P/E", f"{fwd_pe:.1f}x")
                            else:
                                st.metric("Forward P/E", "N/A")
                    
                        with val_col3:
                            peg = valuation_data.get('peg_ratio')
                            earnings_growth = valuation_data.get('earnings_growth')
                        
                            if peg and peg > 0:
                                # Color code PEG ratio
                                # < 1 = potentially undervalued, 1-2 = fair, > 2 = expensive
                                if peg < 1:
                                    peg_label = "Undervalued"
                                elif peg < 2:
                                    peg_label = "Fair"
                                else:
                                    peg_label = "Expensive"
                                st.metric(
                                    "PEG Ratio",
                                    f"{peg:.2f}",
                                    peg_label,
                                    delta_color="off",
                                )
                            elif earnings_growth:
                                # Show earnings growth instead if PEG unavailable (no delta = no arrow)
                                growth_pct = earnings_growth * 100
                                st.metric(
                                    "Earnings Growth (YoY)",
                                    f"{growth_pct:+.1f}%",
                                    delta=None,
                                    delta_color="off",
                                )
                            else:
                                st.metric("PEG Ratio", "N/A")
                    
                        with val_col4:
                            sector = valuation_data.get('sector')
                            st.metric("Sector", sector if sector else "N/A")
                    
                        with val_col5:
                            industry = valuation_data.get('industry')
                            st.metric("Industry", industry[:18] + "..." if industry and len(industry) > 18 else (industry or "N/A"))
                    
                        # Second row: Market cap, 52w high, 52w low (from summary)
                        def _fmt_market_cap(mc):
                            if mc is None or mc <= 0:
                                return "N/A"
                            if mc >= 1e12:
                                return f"${mc / 1e12:.2f}T"
                            if mc >= 1e9:
                                return f"${mc / 1e9:.2f}B"
                            if mc >= 1e6:
                                return f"${mc / 1e6:.2f}M"
                            return f"${mc:,.0f}"
                        v2_1, v2_2, v2_3, v2_4, v2_5 = st.columns(5)
                        with v2_1:
                            st.metric("Market cap", _fmt_market_cap(summary.get("market_cap")))
                        with v2_2:
                            h52 = summary.get("high_52w")
                            st.metric("52w high", f"${h52:,.2f}" if h52 is not None else "N/A")
                        with v2_3:
                            l52 = summary.get("low_52w")
                            st.metric("52w low", f"${l52:,.2f}" if l52 is not None else "N/A")
                        with v2_4:
                            pct_52 = summary.get("pct_from_52w_high")
                            st.metric("% from 52w high", f"{pct_52:+.1f}%" if pct_52 is not None else "N/A")
                        with v2_5:
                            st.metric("", "", help="")
                    
                        # Create and display valuation chart (scroll to zoom, double-click to reset)
                        valuation_fig = create_valuation_chart(valuation_data)
                        st.plotly_chart(valuation_fig, use_container_width=True, config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{ticker_input}_valuation',
                                'height': 600,
                                'width': 1200,
                                'scale': 2
                            }
                        })
                    
                        # Database save button and status
                        from market_data import save_valuation_to_db, check_ticker_in_db
                    
                        db_status = check_ticker_in_db(ticker_input)
                    
                        save_col1, save_col2 = st.columns([3, 1])
                        with save_col1:
                            if db_status['has_data']:
                                status_note = "Fresh" if db_status["is_fresh"] else "Stale"
                                st.caption(
                                    f"**{status_note}:** **{ticker_input}** saved in database: {db_status['quarters']} quarters "
                                    f"(most recent: {db_status['most_recent']})"
                                )
                            else:
                                st.caption(f"**{ticker_input}** not saved in database yet")
                    
                        with save_col2:
                            # Check if data came from database
                            from_db = valuation_data.get('from_database', False)
                        
                            if from_db:
                                st.success("Loaded from database")
                            else:
                                if st.button("Save", key=f"save_valuation_{ticker_input}", help="Save this ticker's valuation data to your database"):
                                    # Get the raw data to save
                                    pe_history = valuation_data.get('pe_data', pd.DataFrame())
                                    revenue_data = valuation_data.get('revenue_data', pd.DataFrame())
                                
                                    # Convert DataFrames back to list format
                                    pe_list = []
                                    if not pe_history.empty:
                                        for _, row in pe_history.iterrows():
                                            pe_list.append({
                                                'date': row['date'],
                                                'pe': row['pe'],
                                                'ttm_eps': row.get('ttm_eps'),
                                                'price': row.get('price'),
                                                'source': row.get('source', 'api')
                                            })
                                
                                    revenue_list = []
                                    if not revenue_data.empty:
                                        for _, row in revenue_data.iterrows():
                                            revenue_list.append({
                                                'date': row['date'],
                                                'revenue': row.get('revenue'),
                                                'yoy_growth': row.get('growth') if row.get('growth_type') == 'yoy' else None,
                                                'qoq_growth': row.get('growth') if row.get('growth_type') == 'qoq' else None
                                            })
                                
                                    if save_valuation_to_db(ticker_input, pe_list, revenue_list):
                                        st.success(f"Saved {ticker_input} to database")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save. Check database connection.")
                    
                        # Valuation interpretation guide
                        with st.expander("How to interpret this chart"):
                            st.markdown("""
                            **P/E Multiple (Top Panel):**
                            - Shows how much investors are willing to pay per dollar of earnings
                            - **Lower P/E** (< 15-20x): May indicate undervaluation or slower growth expectations
                            - **Higher P/E** (> 30-40x): May indicate overvaluation or high growth expectations
                            - **Rising P/E**: Investors becoming more optimistic
                            - **Falling P/E**: Investors becoming more cautious
                        
                            **Revenue Growth YoY % (Bottom Panel):**
                            - Shows the year-over-year revenue growth rate each quarter
                            - **Solid line**: Actual reported results
                            - **Dashed line**: Analyst projections (if available)
                            - **Accelerating growth**: Revenue growth rate increasing → bullish
                            - **Decelerating growth**: Revenue growth rate decreasing → watch for multiple compression
                        
                            ---
                        
                            **PEG ratio (P/E to growth):**
                        
                            The PEG ratio divides P/E by expected annual earnings growth rate. It answers: *"Am I paying a fair price for this growth?"*
                        
                            | PEG range | Interpretation |
                            |-----------|----------------|
                            | **< 1.0** | Potentially **undervalued** relative to growth |
                            | **1.0 - 2.0** | **Fairly valued** |
                            | **> 2.0** | Potentially **expensive** relative to growth |
                        
                            *Example:* A stock with P/E of 30x and 30% earnings growth has PEG = 1.0 (fair). The same P/E with only 15% growth has PEG = 2.0 (expensive).
                        
                            ---
                        
                            **The Key Relationship:**
                            | P/E Trend | Revenue Trend | Interpretation |
                            |-----------|---------------|----------------|
                            | ↑ Rising | ↑ Accelerating | Growth justified premium |
                            | ↑ Rising | ↓ Decelerating | Potential overvaluation (watch) |
                            | ↓ Falling | ↑ Accelerating | Potential opportunity |
                            | ↓ Falling | ↓ Decelerating | Fundamentals weakening |
                            """)
                    else:
                        st.info(
                            f"Valuation data not available for {ticker_input}. "
                            "This may be due to limited financial disclosures or the stock being too new."
                        )
                
                    # === DCF SANDBOX ===
                    st.markdown("---")
                    st.markdown("### Discounted Cash Flow (DCF)")
                    st.caption("Auto-populated with Wall Street consensus and historical data. Adjust assumptions to see real-time implied value.")
                
                    with st.spinner("Loading DCF baseline data..."):
                        from market_data import _cached_get_dcf_baseline
                        dcf_baseline = _cached_get_dcf_baseline(ticker_input)
                
                    if not dcf_baseline:
                        st.info("Some baseline metrics could not be fetched. Please input them manually below.")
                        dcf_baseline = {}
                
                    # Check for negative cash flow exception
                    initial_fcf = dcf_baseline.get('fcf')
                    has_negative_fcf = initial_fcf is not None and initial_fcf < 0
                    if has_negative_fcf:
                        st.warning(f"Company has negative trailing FCF (${initial_fcf:,.0f}). DCF model requires forward-looking positive cash flow assumptions. Defaulting to 0.")
                        initial_fcf = 0.0
                
                    dcf_col1, dcf_col2 = st.columns([1, 2])
                    with dcf_col1:
                        st.markdown("#### Input Assumptions")
                        with st.container(border=True):
                            def_fcf = float(initial_fcf) if initial_fcf else 0.0
                            def_growth = float(dcf_baseline.get('growth_y1_5', 0.02)) * 100
                            def_wacc = float(dcf_baseline.get('wacc', 0.10)) * 100
                        
                            user_fcf = st.number_input("Initial FCF (TTM) $", value=def_fcf, format="%.0f", step=1000000.0)
                            user_g_1_5 = st.slider("Growth Y1-5 (%)", min_value=-20.0, max_value=100.0, value=def_growth, step=0.5)
                            user_g_6_10 = st.slider("Growth Y6-10 (%)", min_value=-20.0, max_value=50.0, value=max(2.0, def_growth/2), step=0.5)
                        
                            # Dynamic max for terminal growth (must be < WACC)
                            wacc_limit = max(1.0, def_wacc - 0.5)
                            user_tg = st.slider("Terminal Growth (%)", min_value=0.0, max_value=min(5.0, wacc_limit), value=2.5, step=0.1)
                            user_wacc = st.slider("WACC / Discount Rate (%)", min_value=max(user_tg + 0.1, 5.0), max_value=25.0, value=def_wacc, step=0.1, help=f"Auto-calculated at {def_wacc:.1f}% using CAPM")
                        
                    with dcf_col2:
                        st.markdown("#### Valuation Output")
                        shares = dcf_baseline.get('shares_outstanding')
                        debt = dcf_baseline.get('total_debt', 0.0)
                        cash = dcf_baseline.get('total_cash', 0.0)
                        current_price = dcf_baseline.get('current_price') or summary.get('current_price')
                    
                        if not shares or shares <= 0:
                            st.info("Missing shares outstanding. Showing Enterprise Value only.")
                    
                        # Math Engine
                        wacc_dec = user_wacc / 100.0
                        g1_dec = user_g_1_5 / 100.0
                        g2_dec = user_g_6_10 / 100.0
                        tg_dec = user_tg / 100.0
                    
                        cf_projections = []
                        pv_cfs = []
                        current_cf = user_fcf
                    
                        # Years 1-5
                        for t in range(1, 6):
                            current_cf *= (1 + g1_dec)
                            cf_projections.append(current_cf)
                            pv_cfs.append(current_cf / ((1 + wacc_dec) ** t))
                        
                        # Years 6-10
                        for t in range(6, 11):
                            current_cf *= (1 + g2_dec)
                            cf_projections.append(current_cf)
                            pv_cfs.append(current_cf / ((1 + wacc_dec) ** t))
                        
                        pv_10yr_fcf = sum(pv_cfs)
                        fcf_yr10 = cf_projections[-1] if cf_projections else 0.0
                    
                        # Terminal Value
                        if wacc_dec > tg_dec:
                            terminal_value = (fcf_yr10 * (1 + tg_dec)) / (wacc_dec - tg_dec)
                            pv_tv = terminal_value / ((1 + wacc_dec) ** 10)
                        else:
                            pv_tv = 0.0
                        
                        enterprise_value = pv_10yr_fcf + pv_tv
                        equity_value = enterprise_value + cash - debt
                    
                        implied_price = equity_value / shares if shares and shares > 0 else 0.0
                    
                        # Headline Metric
                        if implied_price > 0 and current_price and current_price > 0:
                            delta = (implied_price - current_price) / current_price
                            delta_color = "normal" if delta >= 0 else "inverse"
                            st.metric("DCF Implied Price", f"${implied_price:.2f}", f"{delta*100:+.1f}% vs Current (${current_price:.2f})", delta_color=delta_color)
                        elif enterprise_value > 0 or enterprise_value < 0:
                            st.metric("Implied Enterprise Value", f"${enterprise_value:,.0f}", None)
                        
                        # Plotly Waterfall
                        if enterprise_value != 0:
                            # Scale down massive numbers for cleaner display
                            scale = 1
                            suffix = ""
                            max_val = max(abs(enterprise_value), abs(equity_value))
                            if max_val >= 1e12:
                                scale = 1e12
                                suffix = "T"
                            elif max_val >= 1e9:
                                scale = 1e9
                                suffix = "B"
                            elif max_val >= 1e6:
                                scale = 1e6
                                suffix = "M"
                            
                            waterfall = go.Figure(go.Waterfall(
                                orientation="v",
                                measure=["relative", "relative", "relative", "relative", "total"],
                                x=["10Y FCF (PV)", "Terminal (PV)", "Cash", "Debt", "Equity"],
                                textposition="outside",
                                text=[
                                    f"${pv_10yr_fcf/scale:,.1f}{suffix}",
                                    f"${pv_tv/scale:,.1f}{suffix}",
                                    f"${cash/scale:,.1f}{suffix}",
                                    f"-${debt/scale:,.1f}{suffix}",
                                    f"${equity_value/scale:,.1f}{suffix}"
                                ],
                                textfont=dict(family='Monaco, monospace', size=11, color='#8b949e'),
                                y=[pv_10yr_fcf, pv_tv, cash, -debt, equity_value],
                                connector={"line":{"color":"#30363d"}},
                                decreasing={"marker":{"color":"#ff073a"}},
                                increasing={"marker":{"color":"#00ff41"}},
                                totals={"marker":{"color":"#1f77b4"}}
                            ))
                        
                            waterfall.update_layout(
                                title=None,
                                waterfallgap=0.3,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=30, b=40, l=10, r=10),
                                height=350,
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#8b949e')),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                            st.plotly_chart(waterfall, use_container_width=True, config={'displayModeBar': False})
                
                    # === FUNDAMENTALS AND RATIOS (expander) ===
                    try:
                        fundamentals = _cached_get_fundamentals_ratios(ticker_input)
                        st.markdown("### Fundamentals and ratios")
                        if True:
                            if fundamentals is None:
                                st.caption("Fundamentals unavailable.")
                            else:
                                rev = fundamentals.get("revenue_ttm")
                                vals = [rev, fundamentals.get("gross_margin"), fundamentals.get("operating_margin"), fundamentals.get("net_margin"), fundamentals.get("roe"), fundamentals.get("roa")]
                                if all(v is None for v in vals):
                                    st.caption("Fundamentals unavailable.")
                                else:
                                    st.caption(f"**Revenue (TTM):** ${rev:,.0f}" if rev is not None and rev else "**Revenue (TTM):** N/A")
                                    for label, key in [
                                        ("Gross margin", "gross_margin"),
                                        ("Operating margin", "operating_margin"),
                                        ("Net margin", "net_margin"),
                                        ("ROE", "roe"),
                                        ("ROA", "roa"),
                                    ]:
                                        val = fundamentals.get(key)
                                        if val is not None:
                                            if "margin" in key:
                                                st.caption(f"**{label}:** {val * 100:.2f}%" if abs(val) <= 1 else f"**{label}:** {val:.2f}%")
                                            else:
                                                st.caption(f"**{label}:** {val * 100:.2f}%" if abs(val) <= 1 else f"**{label}:** {val:.2f}")
                                        else:
                                            st.caption(f"**{label}:** N/A")
                    except Exception:
                        st.markdown("### Fundamentals and ratios")
                        if True:
                            st.caption("Fundamentals unavailable.")
                    # === OPTIONS · ATM IV TERM (BVOL / OVME-lite) ===
                    st.markdown("---")
                    st.markdown("### Options · ATM implied vol term structure")
                    st.caption(
                        "Implied volatility at the listed strike nearest spot (call/put average when both exist). "
                        "Source: Yahoo Finance option chains—delayed/aggregated, not a dealer surface."
                    )
                    iv_table_df = None
                    try:
                        _spot_iv = float(summary.get("current_price") or 0.0)
                        _sk = round(_spot_iv, 2) if _spot_iv > 0 else 0.0
                        iv_res = _cached_iv_term_structure(ticker_input, _sk)
                        if iv_res.data_warnings:
                            st.caption("Notes: " + " ".join(iv_res.data_warnings[:6]))
                        _iv_rows = [
                            {
                                "Expiry": p.expiry,
                                "DTE": p.dte,
                                "ATM strike": p.strike,
                                "IV %": round(p.iv_atm * 100.0, 2) if p.iv_atm is not None else None,
                                "IV src": p.source,
                            }
                            for p in iv_res.points
                            if p.iv_atm is not None
                        ]
                        if _iv_rows:
                            _df_iv = pd.DataFrame(_iv_rows)
                            iv_table_df = _df_iv
                            _insight = (
                                f"{ticker_input}: ATM IV across {len(_df_iv)} expiries vs days to expiry"
                                + (f" (spot ≈ ${iv_res.spot_used:,.2f})" if iv_res.spot_used else "")
                            )
                            fig_iv = go.Figure(
                                go.Scatter(
                                    x=_df_iv["DTE"],
                                    y=_df_iv["IV %"],
                                    mode="lines+markers",
                                    line=dict(color="#c084fc", width=2),
                                    marker=dict(size=8, color="#e8a838", line=dict(width=1, color="rgba(15,23,42,0.5)")),
                                    hovertemplate="DTE %{x} · IV %{y:.1f}%<extra></extra>",
                                )
                            )
                            fig_iv.update_layout(
                                title={"text": _insight, "font": {"family": "Sora", "size": 14, "color": "#f8fafc"}},
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#cbd5e1",
                                font_family="IBM Plex Sans",
                                xaxis=dict(title="Days to expiry", gridcolor="rgba(148,163,184,0.12)"),
                                yaxis=dict(title="Implied vol (ATM, %)", gridcolor="rgba(148,163,184,0.12)"),
                                margin=dict(l=8, r=8, t=56, b=40),
                            )
                            st.plotly_chart(fig_iv, use_container_width=True)
                            with st.expander("IV term table", expanded=False):
                                st.dataframe(
                                    _gft_tabular_styler(_df_iv),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        else:
                            st.info("No ATM IV points returned for this symbol (try another ticker or check chain availability).")
                    except Exception as _iv_exc:
                        st.warning(f"Options IV section unavailable: {_iv_exc}")
                        iv_table_df = None

                    with st.expander("Options · Black–Scholes (European theory price)", expanded=False):
                        st.caption(
                            "Black–Scholes with continuous yield q. Default rate uses cached ^TNX last close (percent points). "
                            "For comparison only—not a live quote or execution price."
                        )
                        _tnx_bs = _cached_tnx_last_percent()
                        _spot_default = float(summary.get("current_price") or 100.0)
                        bs_spot = st.number_input(
                            "Spot ($)",
                            min_value=0.01,
                            value=max(0.01, _spot_default),
                            key=f"bs_spot_{ticker_input}",
                        )
                        r_pct = st.number_input(
                            "Risk-free (%, annual)",
                            min_value=0.0,
                            max_value=25.0,
                            value=float(_tnx_bs),
                            step=0.05,
                            key=f"bs_r_{ticker_input}",
                            help="Default seeded from ^TNX last yield.",
                        )
                        q_pct = st.number_input(
                            "Dividend yield q (%, annual)",
                            min_value=0.0,
                            max_value=20.0,
                            value=0.0,
                            step=0.05,
                            key=f"bs_q_{ticker_input}",
                        )
                        _preset_labels = ["Manual"]
                        if iv_table_df is not None and not iv_table_df.empty:
                            _preset_labels.extend(
                                f"{row['Expiry']} · DTE {int(row['DTE'])} · IV {row['IV %']}% · K {row['ATM strike']}"
                                for _, row in iv_table_df.iterrows()
                            )
                        pick = st.selectbox(
                            "Inputs from IV table",
                            _preset_labels,
                            key=f"bs_pick_{ticker_input}",
                        )
                        if pick == "Manual":
                            c_a, c_b, c_c = st.columns(3)
                            with c_a:
                                strike_bs = st.number_input(
                                    "Strike ($)",
                                    min_value=0.01,
                                    value=max(0.01, _spot_default),
                                    key=f"bs_k_{ticker_input}",
                                )
                            with c_b:
                                dte_bs = st.number_input(
                                    "Days to expiry",
                                    min_value=1,
                                    max_value=3650,
                                    value=30,
                                    key=f"bs_dte_{ticker_input}",
                                )
                            with c_c:
                                iv_pct_bs = st.number_input(
                                    "IV (%, annual)",
                                    min_value=0.1,
                                    max_value=500.0,
                                    value=35.0,
                                    key=f"bs_iv_{ticker_input}",
                                )
                        else:
                            _pi = _preset_labels.index(pick) - 1
                            _row = iv_table_df.iloc[_pi]
                            strike_bs = float(_row["ATM strike"])
                            dte_bs = int(_row["DTE"])
                            iv_pct_bs = float(_row["IV %"])
                            st.caption(
                                f"Preset: strike **{strike_bs:.2f}** · DTE **{dte_bs}** · IV **{iv_pct_bs:.2f}%**"
                            )
                        _T = float(dte_bs) / 365.0
                        _sig = float(iv_pct_bs) / 100.0
                        _r = float(r_pct) / 100.0
                        _q = float(q_pct) / 100.0
                        _bs_out = black_scholes_european(bs_spot, strike_bs, _T, _r, _sig, _q)
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Call (theory)", f"${_bs_out.call_price:.4f}")
                        with m2:
                            st.metric("Put (theory)", f"${_bs_out.put_price:.4f}")
                        with m3:
                            st.caption(f"d1={_bs_out.d1:.3f} · d2={_bs_out.d2:.3f}")

                with tab_corp:
                    # === COMPANY PROFILE (expander) ===
                    try:
                        profile = _cached_get_company_profile(ticker_input)
                        st.markdown("### Company profile")
                        if profile is None:
                            st.caption("Profile unavailable.")
                        else:
                            full_desc = profile.get("description") or ""
                            if full_desc:
                                st.markdown(full_desc)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption(f"**Sector:** {profile.get('sector') or 'N/A'}")
                                st.caption(f"**Industry:** {profile.get('industry') or 'N/A'}")
                                if profile.get("website"):
                                    st.markdown(f"**Website:** [Link]({profile['website']})")
                                else:
                                    st.caption("**Website:** N/A")
                            with c2:
                                st.caption(f"**Employees:** {profile.get('employees') or 'N/A'}")
                                st.caption(f"**CEO:** {profile.get('ceo') or 'N/A'}")
                    except Exception:
                        st.markdown("### Company profile")
                        st.caption("Profile unavailable.")
                
                    st.markdown("### Insider transactions")
                    if not insider_list:
                        st.caption("No recent insider data or API unavailable. Set FINNHUB_API_KEY for insider transactions.")
                    else:
                        try:
                            profile = _cached_get_company_profile(ticker_input)
                            ceo_name = (profile.get("ceo") or "").strip() if profile else ""
                        except Exception:
                            ceo_name = ""
                        def _name_matches_ceo(name: str, ceo: str) -> bool:
                            if not name or not ceo:
                                return False
                            a = set((name or "").lower().split())
                            b = set((ceo or "").lower().split())
                            return bool(a and b and a == b)
                        rows = []
                        for t in insider_list[:30]:
                            d = t.get("date")
                            d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                            name = (t.get("name") or "").strip()[:40]
                            role = (t.get("relationship") or "").strip()[:30]
                            if not role and ceo_name and _name_matches_ceo(name, ceo_name):
                                role = "CEO"
                            trans_type = t.get("transaction", "")
                            shares = t.get("shares", 0) or 0
                            value = t.get("value", 0) or 0
                            if value <= 0 and shares > 0 and not df_display.empty:
                                try:
                                    d_obj = d.date() if hasattr(d, "date") else d
                                    ts = pd.Timestamp(d_obj)
                                    idx = df_display.index.get_indexer([ts], method="nearest")[0]
                                    if 0 <= idx < len(df_display):
                                        close = float(df_display.iloc[idx]["Close"])
                                        value = int(shares * close)
                                except Exception:
                                    pass
                            sec_link = (t.get("sec_link") or "").strip()
                            rows.append({
                                "date": d_str,
                                "name": name,
                                "role": role or "—",
                                "type": trans_type,
                                "shares": shares,
                                "value": value,
                                "sec_link": sec_link,
                            })
                        # Build HTML table with row colors (green tint = Buy, red tint = Sale)
                        def _esc(s):
                            return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
                        header = "<thead><tr><th>Date</th><th>Name</th><th>Role</th><th>Type</th><th>Shares</th><th>Value ($)</th><th>SEC</th></tr></thead><tbody>"
                        body_parts = []
                        for r in rows:
                            bg = "rgba(0,255,65,0.12)" if r["type"] == "Buy" else "rgba(255,7,58,0.12)"
                            sec_cell = f'<a href="{_esc(r["sec_link"])}" target="_blank" rel="noopener">Form 4</a>' if r["sec_link"] else "—"
                            value_fmt = f"{r['value']:,}" if r["value"] else "—"
                            body_parts.append(
                                f'<tr style="background:{bg}">'
                                f'<td>{_esc(r["date"])}</td><td>{_esc(r["name"])}</td><td>{_esc(r["role"])}</td>'
                                f'<td>{_esc(r["type"])}</td><td>{r["shares"]:,}</td><td>{value_fmt}</td>'
                                f'<td>{sec_cell}</td></tr>'
                            )
                        table_html = f'<table style="width:100%; border-collapse:collapse;"><caption style="text-align:left; margin-bottom:6px;">Buy rows in green, sell in red.</caption>{header}{"".join(body_parts)}</tbody></table>'
                        st.markdown(table_html, unsafe_allow_html=True)

                
                    # === COMPETITORS ===
                    st.markdown("---")
                    st.markdown("### Competitors")
                    st.caption("Same industry and similar market cap; optional sort by description match.")
                    with st.expander("How we pick peers", expanded=False):
                        st.markdown("""
                        Peers are chosen using **industry** and **similar market cap** (via FMP screener).
                        When your company's industry doesn't match our data provider's list, we use sector-level peers and note it above the table.
                        You can sort by **description match** to rank by text similarity to the company description (same candidate set).
                        Matching uses the **full** company description (not the shortened preview in the profile box above).
                        Description match ranks by text similarity, not verified competitive relationship; results can include companies that sound similar but operate in different segments.
                        """)
                    comp_sort = st.radio(
                        "Sort",
                        options=["By industry & size", "By description match"],
                        index=0,
                        key=f"competitors_sort_{ticker_input}",
                        horizontal=True,
                    )
                    sort_by_val = "description" if comp_sort == "By description match" else "industry_size"
                    if "peers_refresh_key" not in st.session_state:
                        st.session_state["peers_refresh_key"] = 0
                    ref_col1, ref_col2 = st.columns([4, 1])
                    with ref_col2:
                        if st.button("Refresh", key=f"competitors_refresh_{ticker_input}", help="Clear peers cache and reload"):
                            clear_peers_cache(ticker_input)
                            st.session_state["peers_refresh_key"] = st.session_state.get("peers_refresh_key", 0) + 1
                            st.rerun()
                    try:
                        comp_result = get_competitors(ticker_input, sort_by_val, max_peers=5)
                        peers_list = comp_result.get("peers") or []
                        fallback_used = comp_result.get("fallback_used", "industry")
                        used_sector_fallback = comp_result.get("used_sector_fallback", False)
                        if fallback_used == "sector":
                            st.caption("Few peers in this industry; showing sector-level (and similar size) peers.")
                        elif fallback_used == "sector_wide_cap":
                            st.caption("Showing sector-level peers with wider size range.")
                        elif fallback_used == "stock_peers":
                            st.caption("Using FMP stock-peers (company-screener not available on your plan). Same sector and similar market cap.")
                        elif used_sector_fallback:
                            st.caption("Showing sector-level peers (industry not matched).")
                        if not peers_list:
                            st.info("No peers found for this industry/market cap. Try another ticker or refresh.")
                        else:
                            def _fmt_mc(mc):
                                if mc is None or mc <= 0:
                                    return "—"
                                if mc >= 1e12:
                                    return f"${mc / 1e12:.2f}T"
                                if mc >= 1e9:
                                    return f"${mc / 1e9:.2f}B"
                                if mc >= 1e6:
                                    return f"${mc / 1e6:.2f}M"
                                return f"${mc:,.0f}"
                            rows = []
                            for p in peers_list:
                                ticker_display = p.get("ticker") or "—"
                                name_display = (p.get("name") or ticker_display)[:50]
                                sector_display = (p.get("sector") or "—")[:20]
                                industry_display = (p.get("industry") or "—")[:25]
                                mc_display = _fmt_mc(p.get("market_cap"))
                                pe_display = f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") is not None else "—"
                                rev = p.get("revenue_ttm")
                                rev_display = f"${rev / 1e9:.2f}B" if rev and rev >= 1e9 else (f"${rev / 1e6:.0f}M" if rev and rev >= 1e6 else ("—" if rev is None else f"${rev:,.0f}"))
                                match_display = str(p.get("description_match_score")) if p.get("description_match_score") is not None else "—"
                                rows.append({
                                    "Ticker": ticker_display,
                                    "Name": name_display,
                                    "Sector": sector_display,
                                    "Industry": industry_display,
                                    "Market cap": mc_display,
                                    "P/E": pe_display,
                                    "Revenue (TTM)": rev_display,
                                    "Match": match_display,
                                })
                            df_comp = pd.DataFrame(rows)
                            if sort_by_val != "description":
                                df_comp = df_comp.drop(columns=["Match"], errors="ignore")
                            st.dataframe(
                                _gft_tabular_styler(df_comp),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption("Analyze a peer:")
                            peer_btns = st.columns(min(len(peers_list), 8))
                            for i, p in enumerate(peers_list[:8]):
                                pt = p.get("ticker")
                                if pt and i < len(peer_btns):
                                    with peer_btns[i]:
                                        if st.button(f"→ {pt}", key=f"analyze_peer_{ticker_input}_{pt}", help=f"Load {pt} in search"):
                                            st.session_state["market_analysis_ticker"] = pt
                                            st.rerun()
                            with st.expander("Edit peers", expanded=False):
                                st.caption("Custom additions/removals are saved and applied to future loads.")
                                add_peer = st.text_input("Add ticker to always include", placeholder="e.g. MSFT", key=f"add_peer_{ticker_input}").upper().strip()
                                if st.button("Add", key=f"add_peer_btn_{ticker_input}") and add_peer:
                                    try:
                                        db = get_db_session()
                                        from models import PeerOverride
                                        existing = db.query(PeerOverride).filter(
                                            PeerOverride.focus_ticker == ticker_input,
                                            PeerOverride.peer_ticker == add_peer,
                                        ).first()
                                        if not existing:
                                            db.add(PeerOverride(focus_ticker=ticker_input, peer_ticker=add_peer, is_excluded=0))
                                            db.commit()
                                            st.success(f"Added {add_peer} to peers for {ticker_input}.")
                                        else:
                                            existing.is_excluded = 0
                                            db.commit()
                                            st.success(f"{add_peer} is already in peers.")
                                        db.close()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to add: {e}")
                                for p in peers_list:
                                    pt = p.get("ticker")
                                    if not pt:
                                        continue
                                    c1, c2 = st.columns([3, 1])
                                    with c2:
                                        if st.button("Remove", key=f"remove_peer_{ticker_input}_{pt}"):
                                            try:
                                                db = get_db_session()
                                                from models import PeerOverride
                                                row = db.query(PeerOverride).filter(
                                                    PeerOverride.focus_ticker == ticker_input,
                                                    PeerOverride.peer_ticker == pt,
                                                ).first()
                                                if row:
                                                    row.is_excluded = 1
                                                else:
                                                    db.add(PeerOverride(focus_ticker=ticker_input, peer_ticker=pt, is_excluded=1))
                                                db.commit()
                                                db.close()
                                                clear_peers_cache(ticker_input)
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed to remove: {e}")
                                    with c1:
                                        st.caption(f"**{pt}** — {p.get('name', '')[:40]}")
                    except Exception as e:
                        st.warning(f"Could not load competitors: {e}")
                
                    # === NEWS (expander) ===
                    try:
                        news_items = _cached_fetch_company_news(ticker_input, 10)
                        st.markdown("### News")
                        if True:
                            if not news_items:
                                st.caption("No recent headlines.")
                            else:
                                for n in news_items:
                                    headline = n.get("headline") or "No title"
                                    url = n.get("url") or "#"
                                    source = n.get("source") or ""
                                    dt = n.get("datetime")
                                    if isinstance(dt, (int, float)):
                                        from datetime import datetime as _dt
                                        try:
                                            dt = _dt.fromtimestamp(dt).strftime("%Y-%m-%d %H:%M") if dt else ""
                                        except Exception:
                                            dt = str(dt) if dt else ""
                                    else:
                                        dt = str(dt) if dt else ""
                                    if url:
                                        st.markdown(f"- [{headline[:80]}{'...' if len(headline) > 80 else ''}]({url})")
                                    else:
                                        st.markdown(f"- {headline[:80]}{'...' if len(headline) > 80 else ''}")
                                    if source or dt:
                                        st.caption(f"  {source} {dt}".strip())
                    except Exception:
                        st.markdown("### News")
                        if True:
                            st.caption("News unavailable.")
    
    # === WATCHLIST SECTION ===
    st.markdown("---")
    st.markdown("## Watchlist monitor")
    
    # Watchlist management
    with st.expander("Manage watchlist", expanded=False):
        add_col, remove_col = st.columns(2)
        
        with add_col:
            new_ticker = st.text_input(
                "Add Ticker to Watchlist",
                placeholder="e.g., NVDA",
                key="add_watchlist_ticker"
            ).upper().strip()
            
            alert_price = st.number_input(
                "Alert Price (optional)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="alert_price"
            )
            
            if st.button("Add to watchlist", use_container_width=True):
                if new_ticker:
                    try:
                        db = get_db_session()
                        existing = db.query(Watchlist).filter(Watchlist.ticker == new_ticker).first()
                        
                        if existing:
                            st.warning(f"{new_ticker} is already in your watchlist.")
                        else:
                            new_watch = Watchlist(
                                ticker=new_ticker,
                                alert_price=Decimal(str(alert_price)) if alert_price > 0 else None
                            )
                            db.add(new_watch)
                            db.commit()
                            st.success(f"Added {new_ticker} to watchlist")
                            st.rerun()
                        
                        db.close()
                    except Exception as e:
                        st.error(f"Error adding to watchlist: {e}")
        
        with remove_col:
            try:
                db = get_db_session()
                watchlist_items = db.query(Watchlist).all()
                db.close()
                
                if watchlist_items:
                    ticker_to_remove = st.selectbox(
                        "Remove Ticker from Watchlist",
                        options=[w.ticker for w in watchlist_items],
                        key="remove_watchlist_ticker"
                    )
                    
                    if st.button("Remove from watchlist", use_container_width=True):
                        try:
                            db = get_db_session()
                            db.query(Watchlist).filter(Watchlist.ticker == ticker_to_remove).delete()
                            db.commit()
                            db.close()
                            st.success(f"Removed {ticker_to_remove} from watchlist")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing from watchlist: {e}")
            except Exception as e:
                st.info("Watchlist is empty. Add some tickers above.")
    
    # Display watchlist summary table
    try:
        db = get_db_session()
        watchlist_items = db.query(Watchlist).all()
        db.close()
        
        if watchlist_items:
            # Header with refresh button
            header_col1, header_col2 = st.columns([4, 1])
            with header_col1:
                st.markdown("### Watchlist summary")
            with header_col2:
                if st.button("Refresh", help="Clear cached data and fetch fresh info", key="refresh_watchlist"):
                    # Clear ticker info cache for all watchlist items
                    cache_dir = _PROJECT_ROOT / ".market_cache"
                    if cache_dir.exists():
                        for item in watchlist_items:
                            info_cache = cache_dir / f"{item.ticker.upper()}_info.json"
                            if info_cache.exists():
                                info_cache.unlink()
                                st.toast(f"Cleared cache for {item.ticker}")
                    st.rerun()
            
            st.caption("Sorted by importance score (technical signals, RSI extremes, earnings proximity)")
            
            watchlist_data = []
            
            with st.spinner("Loading watchlist data..."):
                for item in watchlist_items:
                    alert_price = float(item.alert_price) if item.alert_price else None
                    summary = _cached_get_ticker_summary(item.ticker, alert_price=alert_price)
                    
                    if summary:
                        # Format earnings display
                        earnings_display = None
                        if summary.get('days_to_earnings') is not None:
                            days = summary['days_to_earnings']
                            if days <= 0:
                                earnings_display = "Today!"
                            elif days <= 7:
                                earnings_display = f"{days}d (soon)"
                            elif days <= 14:
                                earnings_display = f"{days}d"
                            else:
                                earnings_display = f"{days}d"
                        elif summary.get('earnings_date'):
                            earnings_display = summary['earnings_date'][:10]  # Just the date part
                        
                        # Format volume vs average
                        vol_display = None
                        if summary.get('vol_vs_avg'):
                            vol_display = f"{summary['vol_vs_avg']:.1f}x"
                        
                        # Format short percent
                        short_display = None
                        if summary.get('short_percent'):
                            short_display = f"{summary['short_percent']:.1f}%"
                        
                        watchlist_data.append({
                            'Score': summary.get('importance_score', 0),
                            'Ticker': summary['ticker'],
                            'Price': summary['current_price'],
                            '3M %': summary.get('change_3m_pct'),
                            'RSI': summary.get('rsi'),
                            'BB%': summary.get('bb_pct'),
                            'Trend': summary.get('trend', 'N/A'),
                            'Signal': summary.get('signal', 'N/A'),
                            'P/E': summary.get('pe_ratio'),
                            'Fwd P/E': summary.get('forward_pe'),
                            'PEG': summary.get('peg_ratio'),
                            '52W %': summary.get('pct_from_52w_high'),
                            'Vol': vol_display,
                            'Cap': summary.get('market_cap_category'),
                            'Earnings': earnings_display,
                            'Short%': short_display,
                            'Target': summary.get('analyst_target'),
                            'Yield %': summary.get('dividend_yield'),
                            'Alert': alert_price,
                            # Store extra data for alerts (not displayed in table)
                            '_signal': summary.get('signal', 'N/A'),
                        })
                    else:
                        watchlist_data.append({
                            'Score': 0,
                            'Ticker': item.ticker,
                            'Price': None,
                            '3M %': None,
                            'RSI': None,
                            'BB%': None,
                            'Trend': 'N/A',
                            'Signal': 'N/A',
                            'P/E': None,
                            'Fwd P/E': None,
                            'PEG': None,
                            '52W %': None,
                            'Vol': None,
                            'Cap': None,
                            'Earnings': None,
                            'Short%': None,
                            'Target': None,
                            'Yield %': None,
                            'Alert': float(item.alert_price) if item.alert_price else None,
                            '_signal': 'N/A',
                        })
            
            if watchlist_data:
                # Sort by importance score (descending)
                watchlist_data.sort(key=lambda x: x.get('Score', 0), reverse=True)
                
                df_watchlist = pd.DataFrame(watchlist_data)
                
                # Remove internal columns before display
                display_columns = ['Score', 'Ticker', 'Price', '3M %', 'RSI', 'BB%', 'Trend', 'Signal', 
                                   'P/E', 'Fwd P/E', 'PEG', '52W %', 'Vol', 'Cap', 'Earnings', 
                                   'Short%', 'Target', 'Yield %', 'Alert']
                df_display = df_watchlist[display_columns]
                
                # Style the dataframe
                st.dataframe(
                    _gft_tabular_styler(df_display),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            help="Importance score based on technical signals, RSI, earnings proximity",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "3M %": st.column_config.NumberColumn(
                            "3M %",
                            help="3-month price change",
                            format="%.1f%%"
                        ),
                        "RSI": st.column_config.NumberColumn(
                            "RSI",
                            help="Relative Strength Index (14-day). <30 oversold, >70 overbought",
                            format="%.1f"
                        ),
                        "BB%": st.column_config.NumberColumn(
                            "BB%",
                            help="Bollinger Band position. <10 near lower band (oversold), >90 near upper (overbought)",
                            format="%.0f"
                        ),
                        "Trend": st.column_config.TextColumn(
                            "Trend",
                            help="Overall trend: BULLISH (SMA50 > SMA200), BEARISH, or NEUTRAL",
                            width="small"
                        ),
                        "Signal": st.column_config.TextColumn(
                            "Signal",
                            help="Technical signal: BUY, SELL, GOLDEN CROSS, DEATH CROSS, or HOLD",
                            width="medium"
                        ),
                        "P/E": st.column_config.NumberColumn(
                            "P/E",
                            help="Trailing Price-to-Earnings ratio",
                            format="%.1f"
                        ),
                        "Fwd P/E": st.column_config.NumberColumn(
                            "Fwd P/E",
                            help="Forward P/E based on expected earnings",
                            format="%.1f"
                        ),
                        "PEG": st.column_config.NumberColumn(
                            "PEG",
                            help="Price/Earnings to Growth ratio. <1 may be undervalued, >2 expensive",
                            format="%.2f"
                        ),
                        "52W %": st.column_config.NumberColumn(
                            "52W %",
                            help="Percent from 52-week high (negative = below high)",
                            format="%.1f%%"
                        ),
                        "Vol": st.column_config.TextColumn(
                            "Vol",
                            help="Today's volume vs 20-day average (e.g., 1.5x = 50% above normal)",
                            width="small"
                        ),
                        "Cap": st.column_config.TextColumn(
                            "Cap",
                            help="Market cap category: Mega ($200B+), Large ($10B+), Mid ($2B+), Small ($300M+), Micro",
                            width="small"
                        ),
                        "Earnings": st.column_config.TextColumn(
                            "Earnings",
                            help="Days until next earnings report",
                            width="small"
                        ),
                        "Short%": st.column_config.TextColumn(
                            "Short%",
                            help="Short interest as % of float. High (>10%) may indicate bearish sentiment or squeeze potential",
                            width="small"
                        ),
                        "Target": st.column_config.NumberColumn(
                            "Target",
                            help="Analyst consensus price target",
                            format="$%.2f"
                        ),
                        "Yield %": st.column_config.NumberColumn(
                            "Yield %",
                            help="Annual dividend yield",
                            format="%.2f%%"
                        ),
                        "Alert": st.column_config.NumberColumn(
                            "Alert",
                            help="Your price alert target",
                            format="$%.2f"
                        ),
                    }
                )
                
                # Alert notifications
                st.markdown("---")
                st.markdown("#### Alerts & Signals")
                
                has_alerts = False
                for item in watchlist_data:
                    alert_price = item.get('Alert')
                    current_price = item.get('Price')
                    signal = item.get('_signal', item.get('Signal', 'N/A'))
                    ticker = item.get('Ticker')
                    
                    # Price alert
                    if alert_price and current_price:
                        if current_price <= alert_price:
                            st.success(
                                f"**{ticker}** reached your alert price. Current: ${current_price:.2f}, alert: ${alert_price:.2f}"
                            )
                            has_alerts = True
                    
                    # Signal alerts
                    if signal in ['BUY', 'GOLDEN CROSS']:
                        st.info(f"**{ticker}** — {signal} signal detected")
                        has_alerts = True
                    elif signal in ['SELL', 'DEATH CROSS']:
                        st.warning(f"**{ticker}** — {signal} signal detected")
                        has_alerts = True
                    
                    # Earnings imminent alert
                    earnings = item.get('Earnings')
                    if earnings and (earnings == "Today!" or "(soon)" in str(earnings)):
                        st.warning(f"**{ticker}** — upcoming earnings: {earnings}")
                        has_alerts = True
                
                if not has_alerts:
                    st.caption("No active alerts at this time.")
                    
        else:
            st.info("Your watchlist is empty. Add tickers above to monitor them.")
            
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")


def ipo_tracker_page():
    """Display IPO Vintage Tracker page with calendar, alerts, and performance analysis."""
    st.title("IPO Vintage Tracker")
    
    # Custom CSS for IPO tracker styling
    st.markdown("""
        <style>
        .ipo-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }
        .vintage-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 12px;
        }
        .vintage-1y { background-color: #3498db; color: white; }
        .vintage-2y { background-color: #9b59b6; color: white; }
        .vintage-3y { background-color: #f39c12; color: white; }
        .pending-badge { background-color: #7f8c8d; color: white; }
        .positive-return { color: #00ff41; }
        .negative-return { color: #ff073a; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Refresh button
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh data"):
            clear_ipo_cache()
            st.cache_data.clear()
            st.rerun()
    
    # ==========================================
    # SECTION 1: THE HORIZON - Upcoming IPOs
    # ==========================================
    with st.expander("Section 1: The Horizon — upcoming IPOs", expanded=True):
        st.markdown("### Upcoming IPO Calendar (Next 30 Days)")
        
        with st.spinner("Loading IPO calendar..."):
            upcoming_ipos = _cached_fetch_ipo_calendar(30)
        
        if upcoming_ipos:
            # Create DataFrame for display
            ipo_data = []
            import urllib.parse
            for ipo in upcoming_ipos:
                price_range = "TBD"
                if ipo.price_range_low and ipo.price_range_high:
                    price_range = f"${ipo.price_range_low:.2f} - ${ipo.price_range_high:.2f}"
                elif ipo.ipo_price:
                    price_range = f"${ipo.ipo_price:.2f}"
                
                shares_str = f"{ipo.shares_offered:,}" if ipo.shares_offered else "TBD"
                days_until = (ipo.ipo_date - date.today()).days
                
                company_disp = ipo.name[:40] + '...' if len(ipo.name) > 40 else ipo.name
                query = urllib.parse.quote_plus(f"{ipo.name}")
                search_url = f"https://www.google.com/search?q={query}"
                company_val = f"{search_url}#_{company_disp}"
                
                ipo_data.append({
                    'Ticker': ipo.ticker,
                    'Company': company_val,
                    'Exchange': ipo.exchange,
                    'Date': ipo.ipo_date.strftime('%Y-%m-%d'),
                    'Days Until': days_until,
                    'Price Range': price_range,
                    'Shares': shares_str,
                    'Status': ipo.status.title()
                })
            
            st.dataframe(
                _gft_tabular_styler(pd.DataFrame(ipo_data)),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Company": st.column_config.LinkColumn("Company", width="medium", display_text=".*#_(.*)"),
                    "Exchange": st.column_config.TextColumn("Exchange", width="small"),
                    "Date": st.column_config.TextColumn("Listing Date", width="small"),
                    "Days Until": st.column_config.NumberColumn("Days Until", width="small"),
                    "Price Range": st.column_config.TextColumn("Price Range", width="small"),
                    "Shares": st.column_config.TextColumn("Shares Offered", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                }
            )
            
            # Follow IPO functionality
            st.markdown("---")
            st.markdown("#### Follow an upcoming IPO")
            st.caption("Save an IPO to your registry for vintage tracking once it goes public.")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                available_tickers = [ipo.ticker for ipo in upcoming_ipos if ipo.ticker != 'N/A']
                selected_ipo_ticker = st.selectbox(
                    "Select IPO to Follow",
                    options=available_tickers if available_tickers else ["No IPOs available"],
                    key="follow_ipo_select"
                )
            
            with col2:
                # Show details of selected IPO
                selected_ipo = next((ipo for ipo in upcoming_ipos if ipo.ticker == selected_ipo_ticker), None)
                if selected_ipo:
                    st.text_input("Company Name", value=selected_ipo.name, disabled=True, key="ipo_name_display")
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Follow IPO", use_container_width=True, type="primary"):
                    if selected_ipo:
                        try:
                            db = get_db_session()
                            
                            # Check if already following
                            existing = db.query(IPO_Registry).filter(
                                IPO_Registry.ticker == selected_ipo.ticker
                            ).first()
                            
                            if existing:
                                st.warning(f"Already following {selected_ipo.ticker}")
                            else:
                                # Determine IPO price
                                ipo_price = None
                                if selected_ipo.ipo_price:
                                    ipo_price = selected_ipo.ipo_price
                                elif selected_ipo.price_range_low and selected_ipo.price_range_high:
                                    ipo_price = (selected_ipo.price_range_low + selected_ipo.price_range_high) / 2
                                
                                new_registry = IPO_Registry(
                                    ticker=selected_ipo.ticker,
                                    company_name=selected_ipo.name,
                                    ipo_date=selected_ipo.ipo_date,
                                    ipo_price=ipo_price,
                                    exchange=selected_ipo.exchange,
                                    shares_offered=selected_ipo.shares_offered,
                                    is_following=1,
                                    created_at=date.today()
                                )
                                db.add(new_registry)
                                db.commit()
                                st.success(f"Now following {selected_ipo.ticker}")
                                st.rerun()
                            
                            db.close()
                        except Exception as e:
                            st.error(f"Error following IPO: {e}")
        else:
            st.info("No upcoming IPOs found in the next 30 days, or API key not configured.")
            st.caption("Set FINNHUB_API_KEY in your .env file to fetch real IPO data.")
    
    # ==========================================
    # SECTION 2: VINTAGE ALERTS
    # ==========================================
    with st.expander("Section 2: Vintage alerts — anniversary notifications", expanded=True):
        st.markdown("### IPOs Approaching Vintage Milestones")
        st.caption("Companies within ±10 days of their 1, 2, or 3-year IPO anniversary.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(IPO_Registry.is_following == 1).all()
            db.close()
            
            if ipo_registries:
                # Convert to dicts for anniversary check
                registry_dicts = [
                    {
                        'ticker': r.ticker,
                        'ipo_date': r.ipo_date,
                        'company_name': r.company_name or r.ticker
                    }
                    for r in ipo_registries
                ]
                
                alerts = check_vintage_anniversaries(registry_dicts, days_threshold=10)
                
                if alerts:
                    for alert in alerts:
                        vintage_tag = f"[{alert['anniversary_years']}Y]"
                        days = alert["days_diff"]
                        if days > 0:
                            time_msg = f"in {days} day{'s' if days != 1 else ''}"
                        elif days < 0:
                            time_msg = f"{abs(days)} day{'s' if abs(days) != 1 else ''} ago"
                        else:
                            time_msg = "today"
                        ipo_d = alert["ipo_date"].strftime("%Y-%m-%d")
                        headline = (
                            f"{vintage_tag} **{alert['ticker']}** ({alert['company_name']}) "
                            f"— {alert['anniversary_years']}-year anniversary {time_msg} (IPO: {ipo_d})"
                        )
                        if alert["status"] == "today":
                            st.success(headline)
                        elif alert["status"] == "upcoming":
                            st.info(headline)
                        else:
                            st.warning(headline)
                else:
                    st.info("No vintage anniversaries within the next 10 days.")
            else:
                st.info("No IPOs in your registry. Follow some IPOs to receive vintage alerts!")
                
        except Exception as e:
            st.error(f"Error loading vintage alerts: {e}")
    
    # ==========================================
    # SECTION 3: PERFORMANCE REVIEW - Leaderboard
    # ==========================================
    with st.expander("Section 3: Performance review — vintage leaderboard", expanded=True):
        st.markdown("### IPO Performance Leaderboard")
        st.caption("Ranking your followed IPOs by total return since debut.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(IPO_Registry.is_following == 1).all()
            db.close()
            
            if ipo_registries:
                leaderboard_data = []
                
                with st.spinner("Calculating vintage performance..."):
                    for registry in ipo_registries:
                        # Skip future IPOs
                        if registry.ipo_date > date.today():
                            continue
                        
                        ipo_price = float(registry.ipo_price) if registry.ipo_price else None
                        vintage = get_vintage_performance(
                            registry.ticker,
                            registry.ipo_date,
                            ipo_price
                        )
                        
                        if vintage:
                            leaderboard_data.append({
                                'Ticker': vintage.ticker,
                                'Company': registry.company_name or vintage.ticker,
                                'IPO Date': vintage.ipo_date.strftime('%Y-%m-%d'),
                                'IPO Price': vintage.ipo_price,
                                'Current Price': vintage.current_price,
                                'Total Return %': vintage.total_return,
                                '1Y Return %': vintage.year_1_return if vintage.year_1_status == "Calculated" else None,
                                '2Y Return %': vintage.year_2_return if vintage.year_2_status == "Calculated" else None,
                                '3Y Return %': vintage.year_3_return if vintage.year_3_status == "Calculated" else None,
                                '1Y Status': vintage.year_1_status,
                                '2Y Status': vintage.year_2_status,
                                '3Y Status': vintage.year_3_status,
                            })
                
                if leaderboard_data:
                    # Sort by total return descending
                    leaderboard_data.sort(key=lambda x: x['Total Return %'] or 0, reverse=True)
                    
                    # Add rank
                    for i, entry in enumerate(leaderboard_data, 1):
                        entry['Rank'] = i
                    
                    # Create display DataFrame
                    display_data = []
                    for entry in leaderboard_data:
                        # Format returns with color indicators
                        total_ret = entry['Total Return %']
                        total_str = f"{total_ret:+.2f}%" if total_ret is not None else "N/A"
                        
                        def format_vintage(ret, status):
                            if status == "Pending":
                                return "Pending"
                            if ret is not None:
                                return f"{ret:+.2f}%"
                            return "N/A"

                        company_disp = entry['Company'][:25] + '...' if len(entry['Company']) > 25 else entry['Company']
                        website_url = _cached_get_company_website(entry['Ticker'], entry['Company'])
                        company_val = f"{website_url}#_{company_disp}"

                        display_data.append({
                            "Rank": str(entry["Rank"]),
                            'Ticker': entry['Ticker'],
                            'Company': company_val,
                            'IPO Date': entry['IPO Date'],
                            'IPO Price': f"${entry['IPO Price']:.2f}",
                            'Current': f"${entry['Current Price']:.2f}",
                            'Total Return': total_str,
                            '1Y': format_vintage(entry['1Y Return %'], entry['1Y Status']),
                            '2Y': format_vintage(entry['2Y Return %'], entry['2Y Status']),
                            '3Y': format_vintage(entry['3Y Return %'], entry['3Y Status']),
                        })
                    
                    st.dataframe(
                        _gft_tabular_styler(pd.DataFrame(display_data)),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.TextColumn("Rank", width="small"),
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Company": st.column_config.LinkColumn("Company", width="medium", display_text=".*#_(.*)"),
                            "IPO Date": st.column_config.TextColumn("IPO Date", width="small"),
                            "IPO Price": st.column_config.TextColumn("IPO $", width="small"),
                            "Current": st.column_config.TextColumn("Now $", width="small"),
                            "Total Return": st.column_config.TextColumn("Total %", width="small"),
                            "1Y": st.column_config.TextColumn("1Y", width="small"),
                            "2Y": st.column_config.TextColumn("2Y", width="small"),
                            "3Y": st.column_config.TextColumn("3Y", width="small"),
                        }
                    )
                    
                    # Summary stats
                    st.markdown("---")
                    st.markdown("#### Registry summary")
                    
                    total_returns = [e['Total Return %'] for e in leaderboard_data if e['Total Return %'] is not None]
                    if total_returns:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best Performer", f"{leaderboard_data[0]['Ticker']}", f"{leaderboard_data[0]['Total Return %']:+.2f}%")
                        with col2:
                            avg_return = sum(total_returns) / len(total_returns)
                            st.metric("Average Return", f"{avg_return:+.2f}%")
                        with col3:
                            winners = len([r for r in total_returns if r > 0])
                            st.metric("Winners", f"{winners}/{len(total_returns)}")
                        with col4:
                            st.metric("Total IPOs Tracked", len(leaderboard_data))
                else:
                    st.info("No performance data available yet. IPOs may not have started trading.")
            else:
                st.info("No IPOs in your registry. Follow some IPOs to track their vintage performance!")
                
        except Exception as e:
            st.error(f"Error loading leaderboard: {e}")
    
    # ==========================================
    # SECTION 4: VIBE CHART - IPO Trajectory Comparison
    # ==========================================
    with st.expander("Vibe chart — IPO debut trajectory comparison", expanded=True):
        st.markdown("### Compare IPO Price Trajectories")
        st.caption("Overlay multiple IPOs aligned by their 'Day 0' (listing date) to compare debut performance.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(
                IPO_Registry.is_following == 1,
                IPO_Registry.ipo_date <= date.today()
            ).all()
            db.close()
            
            if ipo_registries and len(ipo_registries) >= 1:
                # Let user select up to 3 IPOs to compare
                available_tickers = [r.ticker for r in ipo_registries]
                
                selected_tickers = st.multiselect(
                    "Select IPOs to Compare (max 3)",
                    options=available_tickers,
                    default=available_tickers[:min(3, len(available_tickers))],
                    max_selections=3,
                    key="vibe_chart_select"
                )
                
                # Days to show
                days_to_show = st.slider(
                    "Trading Days to Display",
                    min_value=30,
                    max_value=365,
                    value=90,
                    step=30,
                    key="vibe_days_slider"
                )
                
                if selected_tickers:
                    with st.spinner("Loading price trajectories..."):
                        # Collect data for selected IPOs
                        chart_data = []
                        colors = ['#00ff41', '#58a6ff', '#f78166', '#d2a8ff']
                        
                        for i, ticker in enumerate(selected_tickers):
                            registry = next((r for r in ipo_registries if r.ticker == ticker), None)
                            if registry:
                                df = get_ipo_price_history(ticker, registry.ipo_date, days=days_to_show)
                                if df is not None:
                                    df['Color'] = colors[i % len(colors)]
                                    chart_data.append(df)
                        
                        if chart_data:
                            # Create Plotly figure
                            fig = go.Figure()
                            
                            for i, df in enumerate(chart_data):
                                ticker = df['Ticker'].iloc[0]
                                
                                fig.add_trace(go.Scatter(
                                    x=df['Day'],
                                    y=df['Normalized'],
                                    mode='lines',
                                    name=ticker,
                                    line=dict(width=2.5),
                                    hovertemplate=(
                                        f"<b>{ticker}</b><br>"
                                        "Day %{x}<br>"
                                        "Performance: %{y:.1f}%<br>"
                                        "<extra></extra>"
                                    )
                                ))
                            
                            # Add baseline at 100 (IPO price)
                            fig.add_hline(
                                y=100,
                                line_dash="dash",
                                line_color="#8b949e",
                                annotation_text="IPO Price (Day 0)",
                                annotation_position="right"
                            )
                            
                            # Styling
                            fig.update_layout(
                                title=dict(
                                    text="<b>IPO Vibe Chart</b> - Debut Trajectory Comparison",
                                    font=dict(size=20, color='#c9d1d9'),
                                    x=0.5
                                ),
                                paper_bgcolor='#0d1117',
                                plot_bgcolor='#0d1117',
                                font=dict(color='#c9d1d9', family='Monaco, monospace'),
                                legend=dict(
                                    bgcolor='rgba(0,0,0,0.5)',
                                    bordercolor='#21262d',
                                    borderwidth=1,
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='center',
                                    x=0.5
                                ),
                                xaxis=dict(
                                    title="Trading Days Since IPO",
                                    gridcolor='#21262d',
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    title="Normalized Price (100 = IPO Price)",
                                    gridcolor='#21262d',
                                    zeroline=False
                                ),
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show legend/interpretation
                            st.markdown("""
                            **How to Read This Chart:**
                            - All IPOs are **normalized** to start at 100 (their IPO price)
                            - A value of **150** means the stock is up **50%** from IPO
                            - A value of **75** means the stock is down **25%** from IPO
                            - Compare how different IPOs performed in their first days/months of trading
                            """)
                        else:
                            st.warning("Could not load price data for selected IPOs.")
                else:
                    st.info("Select at least one IPO to view the trajectory chart.")
            else:
                st.info("No historical IPOs in your registry. Follow IPOs that have already listed to compare their trajectories.")
                
        except Exception as e:
            st.error(f"Error loading vibe chart: {e}")
    
    # ==========================================
    # MANAGE REGISTRY
    # ==========================================
    st.markdown("---")
    st.markdown("### Manage IPO registry")
    
    with st.expander("View & Edit Your IPO Registry", expanded=False):
        try:
            db = get_db_session()
            all_registries = db.query(IPO_Registry).all()
            db.close()
            
            if all_registries:
                # Display registry
                registry_data = []
                for r in all_registries:
                    registry_data.append({
                        'ID': r.id,
                        'Ticker': r.ticker,
                        'Company': r.company_name or 'Unknown',
                        'IPO Date': r.ipo_date.strftime('%Y-%m-%d') if r.ipo_date else 'N/A',
                        'IPO Price': f"${float(r.ipo_price):.2f}" if r.ipo_price else 'N/A',
                        'Exchange': r.exchange or 'N/A',
                        "Following": "Yes" if r.is_following else "No",
                    })
                
                st.dataframe(
                    _gft_tabular_styler(pd.DataFrame(registry_data)),
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Add manual entry form
                st.markdown("---")
                st.markdown("#### Add IPO manually")
                
                with st.form("manual_ipo_form", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        manual_ticker = st.text_input("Ticker Symbol", placeholder="e.g., UBER").upper()
                        manual_name = st.text_input("Company Name", placeholder="e.g., Uber Technologies Inc.")
                        manual_ipo_date = st.date_input("IPO Date")
                    
                    with col2:
                        manual_ipo_price = st.number_input("IPO Price ($)", min_value=0.01, step=0.01, format="%.2f")
                        manual_exchange = st.selectbox("Exchange", options=["NYSE", "NASDAQ", "AMEX", "Other"])
                    
                    if st.form_submit_button("Add to Registry", use_container_width=True):
                        if manual_ticker and manual_ipo_date:
                            try:
                                db = get_db_session()
                                
                                # Check if exists
                                existing = db.query(IPO_Registry).filter(IPO_Registry.ticker == manual_ticker).first()
                                if existing:
                                    st.warning(f"{manual_ticker} already exists in registry.")
                                else:
                                    new_entry = IPO_Registry(
                                        ticker=manual_ticker,
                                        company_name=manual_name if manual_name else None,
                                        ipo_date=manual_ipo_date,
                                        ipo_price=manual_ipo_price if manual_ipo_price > 0 else None,
                                        exchange=manual_exchange,
                                        is_following=1,
                                        created_at=date.today()
                                    )
                                    db.add(new_entry)
                                    db.commit()
                                    st.success(f"Added {manual_ticker} to registry")
                                    st.rerun()
                                
                                db.close()
                            except Exception as e:
                                st.error(f"Error adding IPO: {e}")
                        else:
                            st.error("Ticker and IPO Date are required.")
                
                # Remove entry
                st.markdown("---")
                st.markdown("#### Remove from registry")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    ticker_to_remove = st.selectbox(
                        "Select IPO to Remove",
                        options=[r['Ticker'] for r in registry_data],
                        key="remove_ipo_select"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Remove", use_container_width=True):
                        try:
                            db = get_db_session()
                            db.query(IPO_Registry).filter(IPO_Registry.ticker == ticker_to_remove).delete()
                            db.commit()
                            db.close()
                            st.success(f"Removed {ticker_to_remove} from registry.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing: {e}")
            else:
                st.info("Your IPO registry is empty. Follow upcoming IPOs or add them manually.")
                
                # Quick add form for empty state
                st.markdown("#### Add your first IPO")
                with st.form("first_ipo_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        first_ticker = st.text_input("Ticker", placeholder="e.g., RIVN").upper()
                        first_date = st.date_input("IPO Date")
                    with col2:
                        first_name = st.text_input("Company Name", placeholder="e.g., Rivian Automotive")
                        first_price = st.number_input("IPO Price ($)", min_value=0.01, value=78.00, step=0.01)
                    
                    if st.form_submit_button("Add IPO"):
                        if first_ticker:
                            try:
                                db = get_db_session()
                                new_entry = IPO_Registry(
                                    ticker=first_ticker,
                                    company_name=first_name,
                                    ipo_date=first_date,
                                    ipo_price=first_price,
                                    is_following=1,
                                    created_at=date.today()
                                )
                                db.add(new_entry)
                                db.commit()
                                db.close()
                                st.success(f"Added {first_ticker}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
        except Exception as e:
            st.error(f"Error loading registry: {e}")


def _gft_fmt_usd_cap(value) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v <= 0:
        return "—"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def partnerships_page():
    """Display Partnerships (8-K Watch) page: SEC EDGAR Item 1.01 filings, signal scoring, excerpts."""
    st.title("Partnerships")
    st.caption(
        "8-K **Item 1.01** · strategic material agreements. Structured disclosure often **lags** fast "
        "narrative; use this as a filing-first queue, not a rumor feed."
    )
    st.markdown(
        "We watch your configured tickers and keep **partnership- and strategic-type** deals "
        "(collaborations, JVs, licenses, M&A). **Financing** filings (credit, indentures, notes) stay out."
    )
    st.markdown("---")

    col_refresh, col_opts = st.columns([1, 5])
    with col_opts:
        force_submissions_refresh = st.checkbox(
            "Re-fetch SEC filing index for every watchlist ticker",
            value=False,
            key="partnerships_force_submissions",
            help="Slower (~1 HTTP per ticker). Use when you need the newest 8-K list before the submissions cache expires (default 1 hour).",
        )
    with col_refresh:
        if st.button("Refresh", key="partnerships_refresh", help="Fetch latest 8-K filings from SEC"):
            with st.spinner("Fetching SEC EDGAR data..."):
                try:
                    _, refresh_warnings = refresh_edgar_data(
                        limit=50,
                        force_submissions_refresh=force_submissions_refresh,
                    )
                    _cached_get_partnership_events.clear()
                    if refresh_warnings:
                        st.warning(
                            "Partial refresh — some watchlist symbols were skipped:\n\n"
                            + "\n".join(f"- {w}" for w in refresh_warnings)
                        )
                    st.success("Data refreshed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    cap_lo_b = FILER_CAP_USD_MIN / 1e9
    cap_hi_b = FILER_CAP_USD_MAX / 1e9
    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        cap_filter = st.selectbox(
            "Cap filter (filer)",
            options=[
                "target_band",
                "target_band_or_unknown",
                "all",
            ],
            index=0,
            format_func=lambda x: {
                "target_band": f"In band (${cap_lo_b:.1f}B–${cap_hi_b:.0f}B)",
                "target_band_or_unknown": "In band + unknown cap",
                "all": "All (dim outside band)",
            }[x],
            key="partnerships_cap_filter",
            help="**In band** matches the thesis range. **All** shows every row; filers outside the band use muted text.",
        )
    with fcol2:
        show_other = st.checkbox(
            'Show "Other" (ambiguous Item 1.01)',
            value=False,
            key="partnerships_show_other",
        )
    with fcol3:
        st.caption("Sort: **Interest hit** → **score** → **filing date**.")

    st.caption(
        "Megacap-heavy watchlists often yield an empty **In band** view. Switch to **All (dim outside band)** "
        "to see filings while keeping out-of-band rows visually de-emphasized."
    )

    try:
        with st.spinner("Loading partnership events…"):
            events = _cached_get_partnership_events(50)
    except Exception as e:
        st.error(f"Error loading partnership events: {e}")
        return

    if not events:
        st.info(
            "No partnership events yet. Click **Refresh** to fetch 8-K Item 1.01 filings "
            "from the SEC for your watched companies. The first run may take a minute."
        )
        return

    if cap_filter in ("target_band", "target_band_or_unknown"):
        with st.spinner("Loading market caps…"):
            events = hydrate_partnership_market_caps(events)
        _cached_get_partnership_events.clear()
    elif cap_filter == "all" and partnership_events_caps_deferred():
        st.caption(
            "Market caps are **deferred** for speed (cap column shows **—**). "
            "Band filters load caps automatically."
        )
        if st.button("Load market caps", key="partnerships_load_caps"):
            with st.spinner("Loading market caps…"):
                events = hydrate_partnership_market_caps(events)
            _cached_get_partnership_events.clear()
            st.rerun()

    filtered = list(events)
    if not show_other:
        filtered = [e for e in filtered if (e.get("relevance_type") or "") == "partnership"]
    if cap_filter == "target_band":
        filtered = [e for e in filtered if e.get("filer_in_cap_band") is True]
    elif cap_filter == "target_band_or_unknown":
        filtered = [e for e in filtered if e.get("filer_in_cap_band") is not False]

    filtered.sort(
        key=lambda e: (
            bool(e.get("interest_hit")),
            e.get("signal_score") or 0,
            e.get("filing_date") or "",
        ),
        reverse=True,
    )

    if not filtered:
        st.warning(
            "No rows match your filters. Widen the cap filter, enable **Other**, or click **Refresh**."
        )
        return

    rows = []
    dim_row_indices: set[int] = set()
    for row_idx, ev in enumerate(filtered):
        filer = (
            f"{ev.get('filer_ticker', '')} "
            f"({ev.get('filer_name', '')[:28]}{'…' if len(ev.get('filer_name', '') or '') > 28 else ''})"
        )
        filing_date = ev.get("filing_date") or ""
        relevance = ev.get("relevance_type") or "other"
        type_label = "Partnership" if relevance == "partnership" else "Other"
        counterparties = ev.get("counterparties") or []
        cp_display = ", ".join(c.get("name", "") for c in counterparties) if counterparties else "—"
        sec_url = ev.get("sec_url") or ""
        score = int(ev.get("signal_score") or 0)
        reasons = ev.get("signal_reasons") or []
        hit_mark = "HIT" if ev.get("interest_hit") else "—"
        r2 = "; ".join(reasons[:2]) if reasons else "—"
        if len(reasons) > 2:
            r2 += "…"
        signal_cell = f"{score:>3} · {hit_mark}\n{r2}"
        excerpt = (ev.get("display_excerpt") or ev.get("snippet") or "").strip() or "—"
        if len(excerpt) > 120:
            excerpt = excerpt[:119] + "…"
        cap_cell = _gft_fmt_usd_cap(ev.get("filer_market_cap"))
        band = ev.get("filer_in_cap_band")
        if band is True:
            cap_cell = f"{cap_cell} · in band"
        elif band is False:
            cap_cell = f"{cap_cell} · outside"
        elif cap_filter == "all":
            cap_cell = f"{cap_cell} · ?"

        if cap_filter == "all" and band is False:
            dim_row_indices.add(row_idx)

        rows.append({
            "Filing date": filing_date,
            "Filer": filer,
            "Cap": cap_cell,
            "Type": type_label,
            "Signal": signal_cell,
            "Excerpt": excerpt,
            "Counterparties": cp_display,
            "Link": sec_url,
        })

    df = pd.DataFrame(rows)
    styler = _gft_partnerships_styler(df, dim_row_indices if dim_row_indices else None)
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Filing date": st.column_config.TextColumn("Filing date", width="small"),
            "Filer": st.column_config.TextColumn("Filer", width="medium"),
            "Cap": st.column_config.TextColumn("Market cap", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Signal": st.column_config.TextColumn(
                "Signal",
                width="medium",
                help="Line 1: score and interest hit. Line 2: top reasons (full list in Inspect below).",
            ),
            "Excerpt": st.column_config.TextColumn("Excerpt", width="large"),
            "Counterparties": st.column_config.TextColumn("Counterparties", width="large"),
            "Link": st.column_config.LinkColumn("SEC", display_text="View", width="small"),
        },
    )

    labels = [f"{e.get('filer_ticker')} · {e.get('filing_date')} · {e.get('accession_number', '')[-8:]}" for e in filtered]
    choice = st.selectbox(
        "Inspect filing (full excerpt)",
        options=list(range(len(filtered))),
        format_func=lambda i: labels[i],
        key="partnerships_inspect_idx",
    )
    if choice is not None and 0 <= choice < len(filtered):
        ev = filtered[choice]
        full_ex = (ev.get("display_excerpt") or ev.get("snippet") or "").strip()
        if full_ex:
            st.text_area(
                "Excerpt (from 8-K body)",
                value=full_ex,
                height=160,
                disabled=True,
                key=f"partnerships_excerpt_{choice}",
            )
        reasons_full = ev.get("signal_reasons") or []
        if reasons_full:
            st.caption("**Why (full):** " + " · ".join(reasons_full))
        st.caption(ev.get("sec_url") or "")

    with st.expander("About this data"):
        st.caption(
            "Data from SEC EDGAR. **Partnership** vs **Other** uses keyword scoring on filing text; "
            "**Financing** is dropped. **Interest** uses your `partnerships_config` names and aliases "
            "(word-boundary style matching for 4+ character names to cut false positives). "
            "**Score** (0–100) weights interest hits, partnership-type language, counterparty extraction, "
            f"and whether the filer’s market cap (yfinance, best-effort) sits in **${cap_lo_b:.1f}B–${cap_hi_b:.0f}B**. "
            "**Unknown cap** is labeled in the Cap column; use **In band + unknown cap** to keep those rows visible. "
            "**All (dim outside band)** shows every passing row but mutes filers clearly outside the cap band."
        )


def thirteenf_page():
    """Display 13F Institutional Holdings page: 13f.info-style data, compare quarters, by-CUSIP, overlap."""
    st.title("13F Institutional Holdings")
    st.markdown(
        "Quarterly 13F filings for selected institutions. View holdings (13f.info-style), "
        "compare two quarters to see adds/subtracts, find who holds a given CUSIP, or see which holdings are common across funds."
    )
    st.markdown("---")

    institution_options = [inst["name"] for inst in THIRTEENF_INSTITUTIONS]
    name_to_cik = {inst["name"]: inst["cik"] for inst in THIRTEENF_INSTITUTIONS}

    # Build quarter list: last 8 quarters (e.g. 2025 Q4 down to 2024 Q1)
    today = date.today()
    quarter_choices = []
    y, q = today.year, (today.month - 1) // 3 + 1
    for _ in range(8):
        quarter_choices.append((y, q, f"{y} Q{q}"))
        q -= 1
        if q < 1:
            q = 4
            y -= 1

    # Controls
    selected_names = st.multiselect(
        "Institutions",
        options=institution_options,
        default=institution_options[: min(5, len(institution_options))],
        key="thirteenf_institutions",
    )
    selected_ciks = [name_to_cik[n] for n in selected_names]
    quarter_labels = [q[2] for q in quarter_choices]
    quarter_by_label = {q[2]: (q[0], q[1]) for q in quarter_choices}

    if not selected_names:
        st.info("Select at least one institution.")
        return

    # Section 1: Single filing (13f.info-style)
    st.markdown("### Single filing")
    inst_single = st.selectbox("Institution", selected_names, key="thirteenf_single_inst")
    q_single = st.selectbox("Quarter", quarter_labels, key="thirteenf_single_q")
    cik_single = name_to_cik[inst_single]
    year_single, qtr_single = quarter_by_label[q_single]
    try:
        data_single = _cached_get_13f_holdings_by_quarter(cik_single, year_single, qtr_single)
        if data_single:
            st.markdown(f"**{data_single.get('filer_name', inst_single)}** — {q_single}")
            st.caption(
                f"Holdings as of {data_single.get('period_end', 'N/A')} | "
                f"Value ($000): {data_single.get('value_thousands', 0):,.0f} | "
                f"Num holdings: {data_single.get('num_holdings', 0)} | "
                f"Date filed: {data_single.get('filing_date', 'N/A')} | "
                f"Form: {data_single.get('form_type', 'N/A')}"
            )
            if data_single.get("sec_url"):
                st.markdown(f"[View on SEC]({data_single['sec_url']})")
            rows = []
            for h in data_single.get("holdings", []):
                rows.append({
                    "Sym": h.get("sym", "—"),
                    "Issuer Name": (h.get("issuer_name") or "")[:60],
                    "CUSIP": h.get("cusip", ""),
                    "Value ($000)": h.get("value_thousands", 0),
                    "%": h.get("pct", 0),
                    "Shares": h.get("shares", 0),
                    "Principal": h.get("principal_type", "SH"),
                    "Option": h.get("option_type", "") or "—",
                })
            if rows:
                df1 = pd.DataFrame(rows)
                st.dataframe(_gft_tabular_styler(df1), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No 13F data for {inst_single} {q_single}.")
    except Exception as e:
        st.error(f"Error loading single filing: {e}")

    st.markdown("---")

    # Section 2: Compare two quarters
    with st.expander("Compare two quarters (adds/subtracts)", expanded=True):
        inst_compare = st.selectbox("Institution", selected_names, key="thirteenf_compare_inst")
        q_a = st.selectbox("Period A", quarter_labels, key="thirteenf_q_a")
        q_b = st.selectbox("Period B", quarter_labels, key="thirteenf_q_b")
        cik_c = name_to_cik[inst_compare]
        filings_c = _cached_get_13f_filings_for_institution(cik_c)
        acc_a = next((f["accession_number"] for f in filings_c if f["year"] == quarter_by_label[q_a][0] and f["quarter"] == quarter_by_label[q_a][1]), None)
        acc_b = next((f["accession_number"] for f in filings_c if f["year"] == quarter_by_label[q_b][0] and f["quarter"] == quarter_by_label[q_b][1]), None)
        if acc_a and acc_b and st.button("Compare", key="thirteenf_do_compare"):
            try:
                compare_data = _cached_get_13f_compare(cik_c, acc_a, acc_b)
                if compare_data:
                    st.caption(f"**{compare_data.get('filer_name', '')}** — Value ($000): {compare_data.get('value_a', 0):,.0f} → {compare_data.get('value_b', 0):,.0f}")
                    rows_c = []
                    for r in compare_data.get("rows", []):
                        rows_c.append({
                            "CUSIP": r.get("cusip", ""),
                            "Issuer Name": (r.get("issuer_name") or "")[:50],
                            "Opt": r.get("option_type", "") or "—",
                            "Shares A": r.get("shares_a", 0),
                            "Shares B": r.get("shares_b", 0),
                            "Diff Sh": r.get("diff_shares", 0),
                            "Chg %": r.get("chg_pct_shares", 0),
                            "Val A ($000)": r.get("value_a", 0),
                            "Val B ($000)": r.get("value_b", 0),
                            "Diff Val": r.get("diff_value", 0),
                        })
                    if rows_c:
                        st.dataframe(
                            _gft_tabular_styler(pd.DataFrame(rows_c)),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.warning("Could not load compare data.")
            except Exception as e:
                st.error(f"Error: {e}")
        elif not acc_a or not acc_b:
            st.caption("Select different quarters that have filings.")

    # Section 3: By CUSIP (who holds)
    with st.expander("By CUSIP (who holds this security)"):
        cusip_input = st.text_input("CUSIP", placeholder="e.g. 037833100", key="thirteenf_cusip").strip().upper()
        q_cusip = st.selectbox("Quarter", quarter_labels, key="thirteenf_cusip_q")
        if cusip_input:
            year_c, qtr_c = quarter_by_label[q_cusip]
            try:
                holders = _cached_get_holders_by_cusip(cusip_input, tuple(selected_ciks), year_c, qtr_c)
                if holders:
                    rows_h = [{"Institution": h["filer_name"], "Shares": h["shares"], "Value ($000)": h["value_thousands"], "%": h["pct"]} for h in holders]
                    st.dataframe(
                        _gft_tabular_styler(pd.DataFrame(rows_h)),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No tracked institution holds this CUSIP in that quarter.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Section 4: Overlap
    with st.expander("Overlap (holdings common to selected funds)"):
        overlap_names = st.multiselect("Institutions to compare (2+)", selected_names, default=selected_names[:2], key="thirteenf_overlap")
        q_overlap = st.selectbox("Quarter", quarter_labels, key="thirteenf_overlap_q")
        if len(overlap_names) >= 2:
            year_o, qtr_o = quarter_by_label[q_overlap]
            overlap_ciks = [name_to_cik[n] for n in overlap_names]
            try:
                overlap_list = _cached_get_overlap_holdings(tuple(overlap_ciks), year_o, qtr_o)
                if overlap_list:
                    rows_o = [{"CUSIP": o["cusip"], "Issuer Name": (o.get("issuer_name") or "")[:60]} for o in overlap_list]
                    st.dataframe(
                        _gft_tabular_styler(pd.DataFrame(rows_o)),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No common holdings for these institutions in that quarter.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.caption("Select at least 2 institutions.")

    st.markdown("---")
    
    # Section 5: Smart Money Intelligence (Agentic AI)
    st.markdown("### Smart Money Intelligence (agentic AI)")
    st.caption("Leverage LLMs to analyze 13F shifts, extract the macro theses behind the trades, and identify consensus vs. divergence across funds.")
    
    ai_tab1, ai_tab2 = st.tabs(["Fund profiler", "Consensus engine"])
    
    with ai_tab1:
        st.markdown("**Skill A: Generate a Fund Posture Report**")
        st.write("Select a fund and quarter to generate a narrative on their risk appetite, sector rotation, and top trade thesis (synthesized from recent news).")
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            agent_inst = st.selectbox("Fund to Profile", selected_names, key="agent_inst")
            agent_q = st.selectbox("Quarter", quarter_labels, key="agent_q")
        with col_f2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_agent_a = st.button("Generate Profile", use_container_width=True, type="primary")
            
        if run_agent_a:
            agent_cik = name_to_cik[agent_inst]
            year_a, qtr_a = quarter_by_label[agent_q]
            with st.spinner(f"Agents are analyzing the 13F and recent news for {agent_inst}... This may take ~15-30s."):
                from agents.thirteenf_agent import analyze_fund_posture
                try:
                    report = analyze_fund_posture(agent_cik, year_a, qtr_a)
                    if report is None:
                        st.error("Failed to generate report. (Check API Keys or Data Availability)")
                    else:
                        st.success("Synthesis Complete.")
                        st.markdown(f"**Risk Appetite:** {report.get('risk_appetite', 'N/A')}")
                        st.markdown(f"**Sector Rotation Logic:** {report.get('sector_rotation_logic', 'N/A')}")
                        st.info(f"**Top Thesis:** {report.get('top_thesis', 'N/A')}")
                except Exception as e:
                    st.error(f"Error accessing Agent: {e}")
                    
    with ai_tab2:
        st.markdown("**Skill B: Executive Consensus & Battlegrounds**")
        st.write("Aggregates the shifts of all currently selected funds to extract what the Smart Money universally likes, and where they fundamentally disagree.")
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            agent_c_q = st.selectbox("Quarter to Analyze", quarter_labels, key="agent_c_q")
        with col_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_agent_b = st.button("Extract Consensus", use_container_width=True, type="primary")
            
        if run_agent_b:
            if len(selected_ciks) < 2:
                st.warning("Please select at least 2 funds in the main filter above to run consensus.")
            else:
                year_c, qtr_c = quarter_by_label[agent_c_q]
                with st.spinner(f"Agents are analyzing {len(selected_ciks)} funds... This will take some time."):
                    from agents.thirteenf_agent import analyze_smart_money_consensus
                    try:
                        brief = analyze_smart_money_consensus(selected_ciks, year_c, qtr_c)
                        if brief is None:
                            st.error("Failed to aggregate data. (Check API Keys)")
                        else:
                            st.success("Consensus Extraction Complete.")
                            st.markdown(f"**Overall Sentiment (Smart Money Barometer):** {brief.get('overall_sentiment', 'N/A')}")
                            
                            st.markdown("#### Consensus buys")
                            st.write(brief.get('consensus_buys', 'N/A'))
                            
                            st.markdown("#### Battlegrounds")
                            st.write(brief.get('battlegrounds', 'N/A'))
                    except Exception as e:
                        st.error(f"Error accessing Agent: {e}")

    with st.expander("About this data"):
        st.caption(
            "Data from SEC EDGAR 13F-HR filings. Holdings are reported by CUSIP; ticker (Sym) is not provided by the SEC. "
            "Compare view shows quarter-over-quarter changes. Overlap shows securities held by all selected institutions in the chosen quarter."
        )


def diagnose_macro_health(df, metric_id):
    """
    Diagnose the health of a macroeconomic indicator based on the latest data.
    Returns:
        status (str): "Healthy", "Teetering", "Unhealthy", or "No Data"
        value_str (str): Formatted string of the current value (e.g., "+3.4% YoY" or "4.1%")
    """
    if df is None or df.empty:
        return "No Data", "N/A"

    try:
        latest_val = df["value"].iloc[-1]

        def get_yoy():
            if len(df) < 5:
                return None
            one_yr_ago = df.index[-1] - pd.DateOffset(years=1)
            closest_idx = df.index.get_indexer([one_yr_ago], method="nearest")[0]
            old_val = df["value"].iloc[closest_idx]
            if old_val == 0:
                return None
            return ((latest_val - old_val) / old_val) * 100

        if metric_id == "gdp":
            yoy = get_yoy()
            if yoy is None:
                return "No Data", "N/A"
            val_str = f"{yoy:+.1f}% YoY"
            if yoy >= 3.0:
                return "Healthy", val_str
            if yoy >= 1.0:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "cpi":
            yoy = get_yoy()
            if yoy is None:
                return "No Data", "N/A"
            val_str = f"{yoy:+.1f}% YoY"
            if yoy <= 2.5:
                return "Healthy", val_str
            if yoy <= 3.5:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "unemployment":
            val_str = f"{latest_val:.1f}%"
            if latest_val <= 4.0:
                return "Healthy", val_str
            if latest_val <= 4.5:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "treasury_10y":
            val_str = f"{latest_val:.2f}%"
            if 2.0 <= latest_val <= 4.0:
                return "Healthy", val_str
            if (4.1 <= latest_val <= 5.0) or (1.5 <= latest_val < 2.0):
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "m2":
            yoy = get_yoy()
            if yoy is None:
                return "No Data", "N/A"
            val_str = f"{yoy:+.1f}% YoY"
            if yoy >= 2.0:
                return "Healthy", val_str
            if yoy >= 0.0:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "pmi":
            val_str = f"{latest_val:.1f}"
            if latest_val >= 50.0:
                return "Healthy", val_str
            if latest_val >= 45.0:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "retail_sales":
            yoy = get_yoy()
            if yoy is None:
                return "No Data", "N/A"
            val_str = f"{yoy:+.1f}% YoY"
            if yoy >= 3.0:
                return "Healthy", val_str
            if yoy >= 1.0:
                return "Teetering", val_str
            return "Unhealthy", val_str

        if metric_id == "consumer_sentiment":
            val_str = f"{latest_val:.1f}"
            if latest_val >= 70.0:
                return "Healthy", val_str
            if latest_val >= 60.0:
                return "Teetering", val_str
            return "Unhealthy", val_str

        return "No Data", "N/A"
    except Exception as e:
        print(f"Error diagnosing macro health for {metric_id}: {e}")
        return "No Data", "N/A"

def macro_dashboard_page():
    """Display the Macro Dashboard page."""
    import plotly.express as px

    st.title("Macro Dashboard")
    st.markdown("---")
    
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh macro data"):
            st.cache_data.clear()
            st.rerun()
            
    st.markdown("A tracking dashboard for the most important macroeconomic indicators utilized by macro funds to assess liquidity, economic capacity, and the business cycle.")
    
    # Define metrics
    metrics = [
        {
            "id": "gdp",
            "title": "Nominal GDP (Billions)",
            "primer": "Gross Domestic Product (GDP) measures the total value of goods produced. It is the broadest measure of economic activity. Two consecutive quarters of decline signal a technical recession.",
            "color": "#1f77b4"
        },
        {
            "id": "cpi",
            "title": "Consumer Price Index (CPI)",
            "primer": "CPI tracks inflation across a basket of goods. The Federal Reserve targets ~2% YoY. High inflation often leads to higher interest rates, which cools the market and shrinks valuations.",
            "color": "#ff7f0e"
        },
        {
            "id": "unemployment",
            "title": "Unemployment Rate (%)",
            "primer": "Rising unemployment signals an economic slowdown. **The Sahm Rule** suggests a recession is near if the 3-month average rises 0.5% above its 12-month low.",
            "color": "#d62728"
        },
        {
            "id": "treasury_10y",
            "title": "10-Year Treasury Yield (%)",
            "primer": "The 10-Year Yield is the benchmark for global borrowing costs. Watch the **Yield Curve**: if short-term yields (e.g., 2-Year) exceed the 10-Year, this 'inversion' is a historically reliable recession indicator.",
            "color": "#2ca02c"
        },
        {
            "id": "m2",
            "title": "M2 Money Supply (Billions)",
            "primer": "M2 measures the total money in circulation and bank accounts. Expanding M2 boosts liquidity and asset prices. A contracting M2 signals a liquidity crunch and potential deleveraging.",
            "color": "#9467bd"
        },
        {
            "id": "pmi",
            "title": "Purchasing Managers' Index (PMI)",
            "primer": "A leading indicator of economic health derived from manufacturing/service sector surveys. A reading **above 50** indicates expansion; **below 50** indicates contraction.",
            "color": "#8c564b"
        },
        {
            "id": "retail_sales",
            "title": "Total Retail Sales (Millions)",
            "primer": "A real-time barometer of consumer spending momentum. Crucial for understanding if consumer demand is keeping up with economic capacity.",
            "color": "#e377c2"
        },
        {
            "id": "consumer_sentiment",
            "title": "Consumer Sentiment (U. Michigan)",
            "primer": "Measures how optimistic consumers feel about their finances and the economy. Since consumer spending is ~70% of US GDP, this sentiment drives future spending.",
            "color": "#7f7f7f"
        }
    ]
    
    st.markdown("---")
    
    # Pre-fetch data and calculate statuses
    metrics_data = {}
    health_counts = {"Healthy": 0, "Teetering": 0, "Unhealthy": 0}
    
    for m in metrics:
        df = _cached_fetch_macro_indicator(m["id"])
        status, val_str = diagnose_macro_health(df, m["id"])
        metrics_data[m["id"]] = {
            "df": df,
            "status": status,
            "val_str": val_str,
        }
        if status in health_counts:
            health_counts[status] += 1
            
    total_valid = sum(health_counts.values())
    
    # Render Thermometer
    if total_valid > 0:
        st.subheader("Economic temperature")
        st.markdown("An aggregate diagnosis of the macroeconomic metrics below to determine the broad health of the business cycle.")
        
        # Calculate percentages for the thermometer segments
        p_healthy = (health_counts['Healthy'] / total_valid) * 100
        p_teetering = (health_counts['Teetering'] / total_valid) * 100
        p_unhealthy = (health_counts['Unhealthy'] / total_valid) * 100
        
        # Custom HTML Thermometer Bar
        st.markdown(f"""
        <div style="width: 100%; display: flex; height: 35px; border-radius: 17px; overflow: hidden; margin-bottom: 15px; background-color: #333;">
            <div style="width: {p_healthy}%; background-color: #2ca02c; display: flex; align-items: center; justify-content: center; color: white; font-size: 16px; font-weight: bold;">
                {health_counts['Healthy'] if health_counts['Healthy'] > 0 else ''}
            </div>
            <div style="width: {p_teetering}%; background-color: #ffc107; display: flex; align-items: center; justify-content: center; color: black; font-size: 16px; font-weight: bold;">
                {health_counts['Teetering'] if health_counts['Teetering'] > 0 else ''}
            </div>
            <div style="width: {p_unhealthy}%; background-color: #d62728; display: flex; align-items: center; justify-content: center; color: white; font-size: 16px; font-weight: bold;">
                {health_counts['Unhealthy'] if health_counts['Unhealthy'] > 0 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Healthy:** {health_counts['Healthy']}")
        with col2:
            st.markdown(f"**Teetering:** {health_counts['Teetering']}")
        with col3:
            st.markdown(f"**Unhealthy:** {health_counts['Unhealthy']}")
            
        st.markdown("---")
    
    # Render charts in a 2-column layout
    for i in range(0, len(metrics), 2):
        col1, col2 = st.columns(2)
        
        # Left column
        metric1 = metrics[i]
        md1 = metrics_data[metric1["id"]]
        with col1:
            st.subheader(f"{metric1['title']} — {md1['val_str']} · {md1['status']}")
            df1 = md1["df"]
            if df1 is not None and not df1.empty:
                fig1 = px.line(df1, y="value", title="")
                fig1.update_traces(line_color=metric1["color"])
                fig1.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info(f"Data not available for {metric1['title']}")
                
            with st.expander("How to read this & tipping points"):
                st.write(metric1["primer"])
                
        # Right column
        if i + 1 < len(metrics):
            metric2 = metrics[i+1]
            md2 = metrics_data[metric2["id"]]
            with col2:
                st.subheader(f"{metric2['title']} — {md2['val_str']} · {md2['status']}")
                df2 = md2["df"]
                if df2 is not None and not df2.empty:
                    fig2 = px.line(df2, y="value", title="")
                    fig2.update_traces(line_color=metric2["color"])
                    fig2.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Value",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(f"Data not available for {metric2['title']}")
                    
                with st.expander("How to read this & tipping points"):
                    st.write(metric2["primer"])
        
        st.markdown("<br>", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
        if initialize_database():
            st.session_state.db_initialized = True
    
    # Sidebar navigation
    st.sidebar.title("Gavin Financial Terminal")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Portfolio & Taxes", "Market Analysis", "Macro Dashboard", "IPO Vintage Tracker", "Partnerships", "13F Holdings"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Personal financial intelligence platform for portfolio tracking, "
        "tax optimization, and market analysis."
    )
    
    # Route to appropriate page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Portfolio & Taxes":
        portfolio_taxes_page()
    elif page == "Market Analysis":
        market_analysis_page()
    elif page == "Macro Dashboard":
        macro_dashboard_page()
    elif page == "IPO Vintage Tracker":
        ipo_tracker_page()
    elif page == "Partnerships":
        partnerships_page()
    elif page == "13F Holdings":
        thirteenf_page()


if __name__ == "__main__":
    main()
