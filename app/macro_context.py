"""
Macro context snapshot (GMM/BTMM-lite): cross-asset movers via yfinance and
optional Treasury/rates via FRED when FRED_API_KEY is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

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

# FRED series id -> display title
FRED_RATES: Dict[str, str] = {
    "DGS10": "10Y Treasury (%)",
    "DGS2": "2Y Treasury (%)",
    "DGS3MO": "3M T-Bill (%)",
    "EFFR": "Fed Funds Effective (%)",
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


def fetch_macro_movers() -> List[MacroMoverRow]:
    rows: List[MacroMoverRow] = []
    for sym, lbl in MACRO_MOVERS:
        try:
            hist = yf.Ticker(sym).history(period="15d", interval="1d", auto_adjust=True)
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
            rows.append(
                MacroMoverRow(
                    sym,
                    lbl,
                    last,
                    prev,
                    _pct_change(last, prev),
                    None,
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
