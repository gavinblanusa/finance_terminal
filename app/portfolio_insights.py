"""
PORT-lite: sector/industry exposure, concentration, and value-weighted beta vs SPY
using existing OHLCV and company profile caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
# ~6 months of trading days for beta stability without huge fetch
MIN_OVERLAP_DAYS = 120


@dataclass
class PortfolioInsights:
    sector_weights: Dict[str, float]  # sector -> pct of portfolio value (0-100)
    industry_weights: Dict[str, float]
    top1_pct: float
    top5_pct: float
    herfindahl: float
    portfolio_beta: Optional[float]
    per_ticker_beta: Dict[str, float]
    beta_weights_used: Dict[str, float]  # normalized weights among names included in beta
    data_warnings: List[str] = field(default_factory=list)


def _normalize_weights(values: List[Tuple[str, float]]) -> Dict[str, float]:
    total = sum(v for _, v in values if v and v > 0)
    if total <= 0:
        return {}
    return {t: float(v) / float(total) for t, v in values if v and v > 0}


def compute_concentration(
    positions: List[Tuple[str, float]],
) -> Tuple[float, float, float]:
    """Returns (top1_pct, top5_pct, herfindahl) as fractions 0-100 for pcts."""
    if not positions:
        return 0.0, 0.0, 0.0
    vals = sorted((v for _, v in positions if v > 0), reverse=True)
    if not vals:
        return 0.0, 0.0, 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0, 0.0, 0.0
    w = [v / total for v in vals]
    top1 = w[0] * 100.0
    top5 = sum(w[:5]) * 100.0
    hhi = sum(x * x for x in w)
    return top1, top5, hhi


def build_sector_industry_weights(
    positions: List[Tuple[str, float]],
    profile_getter: Any,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """
    profile_getter(ticker) -> dict with sector, industry or None.
    Returns sector_weights, industry_weights as % of portfolio (sum ~100), warnings.
    """
    warnings: List[str] = []
    total_val = sum(v for _, v in positions if v and v > 0)
    if total_val <= 0:
        return {}, {}, warnings

    sector_vals: Dict[str, float] = {}
    industry_vals: Dict[str, float] = {}
    unknown_sector_ct = 0

    for ticker, val in positions:
        if not val or val <= 0:
            continue
        prof = profile_getter(ticker)
        sector = (prof or {}).get("sector") or ""
        industry = (prof or {}).get("industry") or ""
        if not str(sector).strip():
            sector = "Unknown"
            unknown_sector_ct += 1
        if not str(industry).strip():
            industry = "Unknown"
        sector_vals[sector] = sector_vals.get(sector, 0.0) + float(val)
        industry_vals[industry] = industry_vals.get(industry, 0.0) + float(val)

    if unknown_sector_ct:
        warnings.append(f"{unknown_sector_ct} ticker(s) missing sector (grouped as Unknown).")

    def to_pct(d: Dict[str, float]) -> Dict[str, float]:
        return {k: (v / total_val) * 100.0 for k, v in sorted(d.items(), key=lambda x: -x[1])}

    return to_pct(sector_vals), to_pct(industry_vals), warnings


def _returns_from_ohlcv(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].astype(float).dropna()
    if len(s) < MIN_OVERLAP_DAYS + 5:
        return None
    r = s.pct_change().dropna()
    return r


def compute_value_weighted_beta(
    positions: List[Tuple[str, float]],
    fetch_ohlcv_fn: Any,
    benchmark: str = BENCHMARK,
) -> Tuple[Optional[float], Dict[str, float], Dict[str, float], List[str]]:
    """
    fetch_ohlcv_fn(ticker, period_years) -> DataFrame or None.
    Returns portfolio_beta, per_ticker_beta, weights_used (renormalized), warnings.
    """
    warnings: List[str] = []
    weights = _normalize_weights(positions)
    if not weights:
        return None, {}, {}, warnings

    bm_df = fetch_ohlcv_fn(benchmark, 2)
    r_m = _returns_from_ohlcv(bm_df)
    if r_m is None or r_m.var() == 0 or np.isnan(r_m.var()):
        warnings.append(f"No usable return series for benchmark {benchmark}")
        return None, {}, {}, warnings

    betas: Dict[str, float] = {}
    included: List[str] = []

    for ticker, w in weights.items():
        df = fetch_ohlcv_fn(ticker, 2)
        r_i = _returns_from_ohlcv(df)
        if r_i is None:
            warnings.append(f"Skipping beta for {ticker}: insufficient OHLCV")
            continue
        aligned = pd.concat([r_i, r_m], axis=1, join="inner").dropna()
        aligned.columns = ["ri", "rm"]
        if len(aligned) < MIN_OVERLAP_DAYS:
            warnings.append(f"Skipping beta for {ticker}: overlap {len(aligned)} < {MIN_OVERLAP_DAYS}")
            continue
        cov = aligned["ri"].cov(aligned["rm"])
        var_m = aligned["rm"].var()
        if var_m == 0 or np.isnan(var_m):
            warnings.append(f"Skipping beta for {ticker}: zero benchmark variance")
            continue
        beta = float(cov / var_m)
        if np.isnan(beta):
            warnings.append(f"Skipping beta for {ticker}: NaN beta")
            continue
        betas[ticker] = beta
        included.append(ticker)

    if not included:
        return None, {}, {}, warnings

    sub_w = {t: weights[t] for t in included}
    sw = sum(sub_w.values())
    if sw <= 0:
        return None, betas, {}, warnings
    norm = {t: sub_w[t] / sw for t in included}
    port_beta = float(sum(norm[t] * betas[t] for t in included))
    return port_beta, betas, norm, warnings


def build_portfolio_insights(
    positions: List[Dict[str, Any]],
    get_company_profile_fn: Any,
    fetch_ohlcv_fn: Any,
) -> PortfolioInsights:
    """
    positions: list of dicts with 'ticker' and 'current_value' (from dashboard cache).
    """
    pairs: List[Tuple[str, float]] = []
    for p in positions:
        t = (p.get("ticker") or "").upper().strip()
        v = float(p.get("current_value") or 0)
        if t and v > 0:
            pairs.append((t, v))

    if not pairs:
        return PortfolioInsights(
            sector_weights={},
            industry_weights={},
            top1_pct=0.0,
            top5_pct=0.0,
            herfindahl=0.0,
            portfolio_beta=None,
            per_ticker_beta={},
            beta_weights_used={},
            data_warnings=[],
        )

    def getter(tk: str):
        return get_company_profile_fn(tk)

    sector_w, industry_w, w_prof = build_sector_industry_weights(pairs, getter)
    top1, top5, hhi = compute_concentration(pairs)
    pb, betas, bweights, w_beta = compute_value_weighted_beta(pairs, fetch_ohlcv_fn)

    all_w = list(w_prof)
    all_w.extend(w_beta)
    return PortfolioInsights(
        sector_weights=sector_w,
        industry_weights=industry_w,
        top1_pct=top1,
        top5_pct=top5,
        herfindahl=hhi,
        portfolio_beta=pb,
        per_ticker_beta=betas,
        beta_weights_used=bweights,
        data_warnings=all_w,
    )
