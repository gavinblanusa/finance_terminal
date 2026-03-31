"""
Pre-trade cost / impact estimates (TRA-lite).

Uses a simple square-root participation model (same spirit as Almgren–Chriss
optimal execution literature) with an explicit illustrative coefficient — not
a broker TCA product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

# Illustrative temporary impact multiplier (tune only for relative comparisons).
# Real TCA systems calibrate this to venue and stock microstructure.
_SQRT_IMPACT_COEFF = 0.35

DEFAULT_ADV_DAYS = 20
DEFAULT_VOL_DAYS = 60


@dataclass
class TCAEstimateResult:
    ticker: str
    side: str
    shares: float
    notional_usd: float
    adv_shares: float
    adv_dollar: float
    participation_rate: float
    daily_volatility: float
    annualized_volatility: float
    estimated_impact_frac: float
    estimated_impact_bps: float
    estimated_impact_usd: float
    price_ref: float
    data_warnings: List[str] = field(default_factory=list)


def estimate_trade_impact(
    ticker: str,
    shares: float,
    side: str,
    fetch_ohlcv_fn: Callable[..., Optional[pd.DataFrame]],
    period_years: int = 2,
    adv_days: int = DEFAULT_ADV_DAYS,
    vol_days: int = DEFAULT_VOL_DAYS,
) -> Optional[TCAEstimateResult]:
    """
    fetch_ohlcv_fn(ticker, period_years) -> OHLCV DataFrame with Close, Volume.
    """
    warnings: List[str] = []
    t = (ticker or "").upper().strip()
    if not t or shares <= 0:
        return None

    df = fetch_ohlcv_fn(t, period_years)
    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        warnings.append("No OHLCV (or missing Volume) for TCA.")
        return TCAEstimateResult(
            ticker=t,
            side=side,
            shares=float(shares),
            notional_usd=0.0,
            adv_shares=0.0,
            adv_dollar=0.0,
            participation_rate=0.0,
            daily_volatility=0.0,
            annualized_volatility=0.0,
            estimated_impact_frac=0.0,
            estimated_impact_bps=0.0,
            estimated_impact_usd=0.0,
            price_ref=0.0,
            data_warnings=warnings,
        )

    close = df["Close"].astype(float).dropna()
    vol = df["Volume"].astype(float).reindex(close.index).fillna(0)
    if len(close) < max(adv_days, vol_days) + 5:
        warnings.append("Short history — ADV/vol may be noisy.")

    price_ref = float(close.iloc[-1])
    recent_v = vol.tail(adv_days)
    adv_shares = float(recent_v.mean()) if len(recent_v) else 0.0
    adv_dollar = adv_shares * price_ref if adv_shares > 0 else 0.0

    notional = float(shares) * price_ref
    participation = (float(shares) / adv_shares) if adv_shares > 0 else float("inf")

    if adv_shares <= 0:
        warnings.append("Average daily volume is zero — cannot compute participation.")

    r = close.pct_change().dropna().tail(vol_days)
    daily_vol = float(r.std()) if len(r) > 1 else 0.0
    ann_vol = daily_vol * np.sqrt(252) if daily_vol > 0 else 0.0

    # Square-root law: impact scales ~ sigma * sqrt(participation), capped for stability.
    part_capped = min(max(participation, 0.0), 3.0) if np.isfinite(participation) else 3.0
    impact_frac = _SQRT_IMPACT_COEFF * daily_vol * np.sqrt(part_capped)
    impact_frac = float(min(max(impact_frac, 0.0), 0.25))
    impact_bps = impact_frac * 10000.0
    impact_usd = notional * impact_frac

    if participation > 0.25:
        warnings.append("High participation vs ADV — model is rough; expect wide error bands.")
    if ann_vol > 0.8:
        warnings.append("Very high realized vol — impact band especially uncertain.")

    return TCAEstimateResult(
        ticker=t,
        side=(side or "buy").lower(),
        shares=float(shares),
        notional_usd=notional,
        adv_shares=adv_shares,
        adv_dollar=adv_dollar,
        participation_rate=float(participation) if np.isfinite(participation) else 0.0,
        daily_volatility=daily_vol,
        annualized_volatility=ann_vol,
        estimated_impact_frac=impact_frac,
        estimated_impact_bps=impact_bps,
        estimated_impact_usd=impact_usd,
        price_ref=price_ref,
        data_warnings=warnings,
    )
