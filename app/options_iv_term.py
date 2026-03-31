"""
ATM implied volatility term structure from listed options (yfinance).

OVME/BVOL-lite: not a dealer vol surface—Yahoo option-chain IV at the strike
nearest spot, per expiry. Educational use only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class IVTermPoint:
    expiry: str
    dte: int
    iv_atm: Optional[float]
    strike: Optional[float]
    source: str


@dataclass
class IVTermStructureResult:
    ticker: str
    spot_used: Optional[float]
    points: List[IVTermPoint] = field(default_factory=list)
    data_warnings: List[str] = field(default_factory=list)


def _resolve_spot(tk: yf.Ticker, spot_override: Optional[float] = None) -> Optional[float]:
    if spot_override is not None and spot_override > 0:
        return float(spot_override)
    try:
        fi = tk.fast_info
        if fi is not None:
            if isinstance(fi, dict):
                lp = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            else:
                lp = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
            if lp is not None and float(lp) > 0:
                return float(lp)
    except Exception:
        pass
    try:
        hist = tk.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            v = float(hist["Close"].iloc[-1])
            if v > 0:
                return v
    except Exception:
        pass
    return None


def build_iv_term_structure(
    ticker: str,
    max_expirations: int = 12,
    spot_override: Optional[float] = None,
) -> IVTermStructureResult:
    sym = (ticker or "").upper().strip()
    warnings: List[str] = []
    if not sym:
        return IVTermStructureResult(ticker="", spot_used=None, data_warnings=["Empty ticker"])

    tk = yf.Ticker(sym)
    try:
        exps = list(tk.options)
    except Exception as e:
        return IVTermStructureResult(
            ticker=sym,
            spot_used=None,
            data_warnings=[f"No options metadata: {e}"],
        )

    if not exps:
        return IVTermStructureResult(
            ticker=sym,
            spot_used=None,
            data_warnings=["No listed expiries (symbol may be unsupported or illiquid)."],
        )

    spot = _resolve_spot(tk, spot_override)
    if spot is None or spot <= 0:
        warnings.append("Spot not resolved; picking ATM strike from first chain by median strike.")

    today = datetime.now(timezone.utc).date()
    rows: List[IVTermPoint] = []

    for exp_str in exps[: max(1, max_expirations)]:
        try:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_dt - today).days
        if dte < 0:
            continue

        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            warnings.append(f"Skip {exp_str}: option_chain failed.")
            continue

        calls = chain.calls
        puts = chain.puts
        if calls is None or calls.empty or "strike" not in calls.columns:
            warnings.append(f"Skip {exp_str}: no calls.")
            continue

        strikes = calls["strike"].astype(float)
        if spot and spot > 0:
            idx = int((strikes - spot).abs().argmin())
        else:
            idx = int(strikes.sub(strikes.median()).abs().argmin())

        row_c = calls.iloc[idx]
        strike_sel = float(row_c["strike"])
        iv_c = row_c.get("impliedVolatility")
        iv_c_f = float(iv_c) if pd.notna(iv_c) else None

        iv_p_f: Optional[float] = None
        if puts is not None and not puts.empty and "strike" in puts.columns:
            pm = puts[puts["strike"].astype(float) == strike_sel]
            if not pm.empty:
                iv_p = pm.iloc[0].get("impliedVolatility")
                iv_p_f = float(iv_p) if pd.notna(iv_p) else None

        iv_atm: Optional[float] = None
        src = "avg"
        if iv_c_f is not None and iv_p_f is not None:
            iv_atm = (iv_c_f + iv_p_f) / 2.0
            src = "avg"
        elif iv_c_f is not None:
            iv_atm = iv_c_f
            src = "call"
        elif iv_p_f is not None:
            iv_atm = iv_p_f
            src = "put"

        rows.append(
            IVTermPoint(
                expiry=exp_str,
                dte=dte,
                iv_atm=iv_atm,
                strike=strike_sel,
                source=src,
            )
        )

    if not rows:
        warnings.append("No IV points extracted.")
    return IVTermStructureResult(
        ticker=sym,
        spot_used=spot,
        points=rows,
        data_warnings=warnings,
    )
