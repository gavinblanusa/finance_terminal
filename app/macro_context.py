"""
Macro context snapshot (GMM/BTMM-lite): cross-asset movers via yfinance and
optional Treasury/rates via FRED when FRED_API_KEY is set.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

# Enough calendar days for ~20+ trading closes (vol uses last 21 closes → 20 daily returns).
MOVER_HISTORY_PERIOD = "3mo"
# Symbols where daily-return σ is misleading (implied vol index, etc.): omit σ and Δ/σ.
SKIP_CHANGE_SIGMA_SYMBOLS = frozenset({"^VIX"})

# Yahoo symbols: (symbol, short label for UI)
MACRO_MOVERS: List[tuple[str, str]] = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("^DJI", "Dow"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
    ("EURUSD=X", "EUR/USD"),
    ("DX-Y.NYB", "DXY"),
    ("GC=F", "Gold"),
    ("CL=F", "WTI Crude"),
]

# FRED series id -> display title (order: curve-ish then policy)
FRED_RATES: Dict[str, str] = {
    "DGS3MO": "3M T-Bill (%)",
    "DGS2": "2Y Treasury (%)",
    "DGS5": "5Y Treasury (%)",
    "DGS10": "10Y Treasury (%)",
    "DGS30": "30Y Treasury (%)",
    "EFFR": "Fed Funds Effective (%)",
    "SOFR": "SOFR (%)",
    "T10Y2Y": "10Y–2Y spread (%)",
}


@dataclass
class MacroMoverRow:
    symbol: str
    label: str
    last: float | None
    previous_close: float | None
    pct_change: float | None
    error: str | None = None
    realized_vol_20d: float | None = None
    change_over_sigma: float | None = None


@dataclass
class FredRateRow:
    series_id: str
    title: str
    value: float | None
    as_of: str | None
    error: str | None = None


@dataclass
class MacroContextResult:
    movers: List[MacroMoverRow] = field(default_factory=list)
    rates: List[FredRateRow] = field(default_factory=list)
    fred_configured: bool = False
    errors: List[str] = field(default_factory=list)


def _pct_change(last: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return (last - prev) / prev * 100.0


def _daily_pct_returns(closes: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(closes)):
        prev_c = closes[i - 1]
        if prev_c == 0:
            return []
        out.append((closes[i] - prev_c) / prev_c * 100.0)
    return out


def realized_vol_20d_from_closes(close_series: pd.Series) -> float | None:
    """
    Sample stdev of the last 20 daily percentage changes from the last 21 closes.
    Returns None if insufficient data or degenerate series.
    """
    vals = [float(x) for x in close_series.dropna().tolist()]
    if len(vals) < 21:
        return None
    window = vals[-21:]
    rets = _daily_pct_returns(window)
    if len(rets) < 20:
        return None
    last_20 = rets[-20:]
    if len(last_20) < 2:
        return None
    try:
        v = statistics.stdev(last_20)
    except statistics.StatisticsError:
        return None
    if v is None or v < 1e-12:
        return None
    return float(v)


def _vol_and_z(
    close: pd.Series,
    last: float,
    prev: float,
    pct_change: float | None,
    skip_sigma: bool,
) -> tuple[float | None, float | None]:
    if skip_sigma or pct_change is None:
        return None, None
    vol = realized_vol_20d_from_closes(close)
    if vol is None or vol < 1e-12:
        return None, None
    z = pct_change / vol
    return vol, z


def fetch_macro_movers() -> List[MacroMoverRow]:
    rows: List[MacroMoverRow] = []
    for sym, lbl in MACRO_MOVERS:
        try:
            hist = yf.Ticker(sym).history(
                period=MOVER_HISTORY_PERIOD, interval="1d", auto_adjust=True
            )
            if hist is None or hist.empty or "Close" not in hist.columns:
                rows.append(
                    MacroMoverRow(
                        sym,
                        lbl,
                        None,
                        None,
                        None,
                        "No data",
                    )
                )
                continue
            close = hist["Close"].dropna()
            if len(close) < 2:
                rows.append(
                    MacroMoverRow(
                        sym,
                        lbl,
                        float(close.iloc[-1]) if len(close) else None,
                        None,
                        None,
                        "Insufficient history",
                    )
                )
                continue
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            pct = _pct_change(last, prev)
            skip_sig = sym in SKIP_CHANGE_SIGMA_SYMBOLS
            vol, z = _vol_and_z(close, last, prev, pct, skip_sig)
            rows.append(
                MacroMoverRow(
                    sym,
                    lbl,
                    last,
                    prev,
                    pct,
                    None,
                    vol,
                    z,
                )
            )
        except Exception as e:
            rows.append(
                MacroMoverRow(sym, lbl, None, None, None, str(e)),
            )
    return rows


def fetch_fred_rates() -> tuple[List[FredRateRow], bool]:
    if not FRED_API_KEY:
        return [], False

    rows: List[FredRateRow] = []
    base = "https://api.stlouisfed.org/fred/series/observations"
    for series_id, title in FRED_RATES.items():
        try:
            r = requests.get(
                base,
                params={
                    "series_id": series_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            obs = data.get("observations") or []
            if not obs:
                rows.append(
                    FredRateRow(series_id, title, None, None, "No observations"),
                )
                continue
            o = obs[0]
            val_raw = o.get("value")
            if val_raw in (None, "."):
                rows.append(
                    FredRateRow(series_id, title, None, o.get("date"), "Missing value"),
                )
                continue
            rows.append(
                FredRateRow(
                    series_id,
                    title,
                    float(val_raw),
                    o.get("date"),
                    None,
                )
            )
        except Exception as e:
            rows.append(
                FredRateRow(series_id, title, None, None, str(e)),
            )

    return rows, True


def build_macro_context() -> MacroContextResult:
    movers = fetch_macro_movers()
    rates, fred_ok = fetch_fred_rates()
    errors = [m.error for m in movers if m.error]
    errors.extend(r.error for r in rates if r.error)
    return MacroContextResult(
        movers=movers,
        rates=rates,
        fred_configured=fred_ok,
        errors=errors,
    )


def macro_context_to_dataframes(result: MacroContextResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serialize for Streamlit dataframe display."""
    mrows = []
    for m in result.movers:
        mrows.append(
            {
                "Label": m.label,
                "Symbol": m.symbol,
                "Last": m.last,
                "Prev close": m.previous_close,
                "Change %": m.pct_change,
                "σ 20d %": m.realized_vol_20d,
                "Δ/σ": m.change_over_sigma,
                "Note": m.error or "",
            }
        )
    movers_df = pd.DataFrame(mrows)

    rrows = []
    for r in result.rates:
        rrows.append(
            {
                "Series": r.series_id,
                "Description": r.title,
                "Value": r.value,
                "As of": r.as_of or "",
                "Note": r.error or "",
            }
        )
    rates_df = pd.DataFrame(rrows)
    return movers_df, rates_df
