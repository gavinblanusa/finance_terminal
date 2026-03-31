"""
Fixed income / credit context strip (SRCH/YAS-lite): ETF and yield proxies via Yahoo.

Not TRACE or live bond inventory—liquid proxies for rate and spread tone only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import yfinance as yf


@dataclass
class FIProxyRow:
    symbol: str
    label: str
    last: float | None
    previous_close: float | None
    pct_change: float | None
    error: str | None = None


# Yahoo symbols for dashboard strip (duration / credit tone)
FI_PROXIES: List[tuple[str, str]] = [
    ("^TNX", "US 10Y yield"),
    ("HYG", "HYG · HY credit ETF"),
    ("LQD", "LQD · IG credit ETF"),
    ("TLT", "TLT · long Treasury ETF"),
    ("IEF", "IEF · 7–10Y Treasury ETF"),
]


def _pct_change(last: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return (last - prev) / prev * 100.0


def build_fi_context_strip() -> tuple[List[FIProxyRow], List[str]]:
    """Returns rows + aggregate error strings (non-fatal per row)."""
    errors: List[str] = []
    rows: List[FIProxyRow] = []
    for sym, lbl in FI_PROXIES:
        try:
            hist = yf.Ticker(sym).history(period="15d", interval="1d", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                rows.append(FIProxyRow(sym, lbl, None, None, None, "No data"))
                continue
            close = hist["Close"].dropna()
            if len(close) < 2:
                rows.append(
                    FIProxyRow(
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
                FIProxyRow(sym, lbl, last, prev, _pct_change(last, prev), None),
            )
        except Exception as e:
            err = str(e)
            errors.append(f"{sym}: {err}")
            rows.append(FIProxyRow(sym, lbl, None, None, None, err))
    return rows, errors


def fi_rows_to_records(rows: List[FIProxyRow]) -> List[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "Label": r.label,
                "Symbol": r.symbol,
                "Last": r.last,
                "Prev": r.previous_close,
                "Change %": r.pct_change,
                "Note": r.error or "",
            }
        )
    return out
