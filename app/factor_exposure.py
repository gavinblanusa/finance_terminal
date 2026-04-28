"""
Fama-French 5-factor exposure (PORT depth): Ken French daily factors + per-ticker
OLS vs portfolio value-weighted factor betas.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

# Match portfolio_insights beta stability window
MIN_OVERLAP_DAYS = 120

# Minimum overlapping attribution days before we surface a chart (noisy below this)
ATTR_MIN_DAYS = 5

FF5_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_CSV_NAME = "F-F_Research_Data_5_Factors_2x3_daily.csv"
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / ".market_cache"
_CACHE_TTL = timedelta(hours=24)


def _ensure_cache() -> None:
    _CACHE_DIR.mkdir(exist_ok=True)


def _ff_cache_path() -> Path:
    return _CACHE_DIR / "ff5_factors_daily.csv"


def _download_ff5_daily() -> Optional[pd.DataFrame]:
    try:
        r = requests.get(FF5_ZIP_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(FF5_CSV_NAME) as f:
                raw = f.read()
    except Exception:
        return None

    df = pd.read_csv(io.BytesIO(raw), skiprows=4)
    # First column is YYYYMMDD (unnamed or empty name)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date_raw"})
    df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for c in FACTOR_COLS + ["RF"]:
        if c not in df.columns:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df.dropna(subset=FACTOR_COLS + ["RF"])
    _ensure_cache()
    out = df[[*FACTOR_COLS, "RF"]]
    try:
        out.to_csv(_ff_cache_path(), index=True, date_format="%Y-%m-%d")
    except Exception:
        pass
    return out


def load_ff5_factors(force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load Fama-French 5 daily factors + RF. Values are decimal daily returns
    (e.g. 0.0079 for 0.79%). Returns (df, warnings).
    """
    warnings: List[str] = []
    path = _ff_cache_path()
    _ensure_cache()

    if not force_refresh and path.exists():
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age < _CACHE_TTL:
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty and all(c in df.columns for c in FACTOR_COLS + ["RF"]):
                    return df, warnings
            except Exception:
                warnings.append("FF factor cache unreadable; will try download.")

    df = _download_ff5_daily()
    if df is None or df.empty:
        warnings.append("Could not load Fama-French 5 factors (network or parse error).")
        return None, warnings

    return df, warnings


def _returns_from_ohlcv(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].astype(float).dropna()
    if len(s) < MIN_OVERLAP_DAYS + 5:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(s.index).tz_localize(None).normalize())
    s = pd.Series(s.values, index=idx).sort_index()
    r = s.pct_change().dropna()
    return r


def _regress_one_ticker(
    stock_r: pd.Series,
    factors: pd.DataFrame,
) -> Tuple[Optional[Dict[str, float]], Optional[float], int]:
    """
    Align stock returns with factors on calendar dates.
    y = stock_r - RF (excess return). X = FACTOR_COLS.
    Returns (betas dict, r2, n_obs) or (None, None, 0).
    """
    aligned = pd.concat(
        [stock_r, factors[FACTOR_COLS + ["RF"]]],
        axis=1,
        join="inner",
    ).dropna()
    aligned.columns = ["stock_r", *FACTOR_COLS, "RF"]
    if len(aligned) < MIN_OVERLAP_DAYS:
        return None, None, len(aligned)

    y = (aligned["stock_r"] - aligned["RF"]).values
    X = aligned[FACTOR_COLS].values
    X_design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 if ss_tot == 0 and ss_res < 1e-12 else (1.0 - ss_res / ss_tot if ss_tot else 0.0)
    betas = {FACTOR_COLS[i]: float(coef[i + 1]) for i in range(len(FACTOR_COLS))}
    betas["_alpha"] = float(coef[0])
    return betas, r2, len(aligned)


@dataclass
class FactorExposureResult:
    """Portfolio and per-name Fama-French 5 loadings."""

    portfolio_factor_betas: Dict[str, float]
    per_ticker_betas: Dict[str, Dict[str, float]]
    per_ticker_r2: Dict[str, float]
    per_ticker_n_obs: Dict[str, int]
    factor_names: List[str]
    as_of: Optional[str]
    regression_end: Optional[str]
    regression_start: Optional[str]
    data_warnings: List[str] = field(default_factory=list)
    factors_available: bool = False


def build_factor_exposure(
    positions: List[Dict[str, Any]],
    fetch_ohlcv_fn: Callable[..., Optional[pd.DataFrame]],
    period_years: int = 3,
) -> FactorExposureResult:
    """
    positions: list of dicts with 'ticker' and 'current_value'.
    fetch_ohlcv_fn(ticker, period_years) -> OHLCV DataFrame (same as market_data.fetch_ohlcv).
    """
    warnings: List[str] = []
    pairs: List[Tuple[str, float]] = []
    for p in positions:
        t = (p.get("ticker") or "").upper().strip()
        v = float(p.get("current_value") or 0)
        if t and v > 0:
            pairs.append((t, v))

    empty = FactorExposureResult(
        portfolio_factor_betas={},
        per_ticker_betas={},
        per_ticker_r2={},
        per_ticker_n_obs={},
        factor_names=list(FACTOR_COLS),
        as_of=None,
        regression_end=None,
        regression_start=None,
        data_warnings=warnings,
        factors_available=False,
    )

    if not pairs:
        return empty

    total = sum(v for _, v in pairs)
    if total <= 0:
        return empty

    weights = {t: v / total for t, v in pairs}

    ff, w_ff = load_ff5_factors()
    warnings.extend(w_ff)
    if ff is None or ff.empty:
        return FactorExposureResult(
            portfolio_factor_betas={},
            per_ticker_betas={},
            per_ticker_r2={},
            per_ticker_n_obs={},
            factor_names=list(FACTOR_COLS),
            as_of=None,
            regression_end=None,
            regression_start=None,
            data_warnings=warnings,
            factors_available=False,
        )

    # Restrict factors to recent window (speed + match stock data)
    end_d = ff.index.max()
    start_d = end_d - pd.Timedelta(days=365 * period_years + 30)
    ff_win = ff.loc[(ff.index >= start_d) & (ff.index <= end_d)]

    per_betas: Dict[str, Dict[str, float]] = {}
    per_r2: Dict[str, float] = {}
    per_n: Dict[str, int] = {}
    included_weights: Dict[str, float] = {}

    for ticker, w in weights.items():
        df = fetch_ohlcv_fn(ticker, period_years)
        r = _returns_from_ohlcv(df)
        if r is None:
            warnings.append(f"{ticker}: insufficient OHLCV for factor regression.")
            continue
        betas, r2, n_obs = _regress_one_ticker(r, ff_win)
        if betas is None or r2 is None:
            warnings.append(f"{ticker}: overlap with factors < {MIN_OVERLAP_DAYS} days.")
            continue
        # drop alpha from portfolio aggregation
        clean = {k: v for k, v in betas.items() if k != "_alpha"}
        per_betas[ticker] = clean
        per_r2[ticker] = r2
        per_n[ticker] = n_obs
        if r2 < 0.05:
            warnings.append(f"{ticker}: low R² ({r2:.2f}) — factor fit noisy.")
        included_weights[ticker] = w

    sw = sum(included_weights.values())
    port: Dict[str, float] = {f: 0.0 for f in FACTOR_COLS}
    if sw > 0:
        for t, w in included_weights.items():
            nw = w / sw
            for f in FACTOR_COLS:
                port[f] += nw * per_betas[t][f]

    reg_end = ff_win.index.max().strftime("%Y-%m-%d") if len(ff_win) else None
    reg_start = ff_win.index.min().strftime("%Y-%m-%d") if len(ff_win) else None

    if not included_weights:
        warnings.append("No tickers produced a factor regression.")

    return FactorExposureResult(
        portfolio_factor_betas=port,
        per_ticker_betas=per_betas,
        per_ticker_r2=per_r2,
        per_ticker_n_obs=per_n,
        factor_names=list(FACTOR_COLS),
        as_of=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        regression_end=reg_end,
        regression_start=reg_start,
        data_warnings=warnings,
        factors_available=True,
    )


def _fetch_years_for_span(need_start: pd.Timestamp) -> int:
    """OHLCV lookback: cover estimation + attribution through today."""
    now_ts = pd.Timestamp.now().normalize()
    span_days = max(0, int((now_ts - need_start.normalize()).days))
    return max(3, min(50, span_days // 365 + 2))


def resolve_attribution_window(
    ff: Optional[pd.DataFrame],
    preset: str,
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
) -> Tuple[Optional[date], Optional[date], List[str]]:
    """
    Map UI/API preset to inclusive calendar dates on the Ken French trading calendar.
    Returns (start, end, warnings); (None, None, errors) if unusable.
    """
    warnings: List[str] = []
    if ff is None or ff.empty:
        return None, None, ["No factor calendar available."]
    idx = ff.index.sort_values()
    last = idx.max()
    preset_l = (preset or "21").strip().lower()

    if preset_l == "custom":
        if custom_start is None or custom_end is None:
            return None, None, ["Custom attribution requires start and end dates."]
        if custom_start > custom_end:
            return None, None, ["Attribution start is after end date."]
        first_d = idx.min().date()
        last_d = idx.max().date()
        cs = max(custom_start, first_d)
        ce = min(custom_end, last_d)
        if cs > ce:
            return None, None, ["Attribution window does not overlap the factor calendar."]
        if cs > custom_start or ce < custom_end:
            warnings.append("Attribution window clamped to factor calendar bounds.")
        return cs, ce, warnings

    if len(idx) < ATTR_MIN_DAYS:
        return None, None, ["Factor history too short for attribution."]

    if preset_l == "mtd":
        month_start = last.replace(day=1)
        mtd = idx[idx >= month_start]
        if len(mtd) < ATTR_MIN_DAYS:
            warnings.append("MTD window is short; attribution is noisy.")
        return mtd[0].date(), mtd[-1].date(), warnings

    if preset_l == "63":
        n = 63
    else:
        n = 21
        if preset_l != "21":
            warnings.append(f"Unknown preset {preset!r}; using last {n} trading days.")

    take = idx[-min(n, len(idx)) :]
    if len(take) < n:
        warnings.append(f"Using {len(take)} trading days (requested {n}).")
    return take[0].date(), take[-1].date(), warnings


@dataclass
class FactorAttributionResult:
    """Cumulative factor attribution vs value-weighted portfolio (estimation before window)."""

    attribution_start: Optional[str]
    attribution_end: Optional[str]
    estimation_start: Optional[str]
    estimation_end: Optional[str]
    n_attribution_days: int
    portfolio_alpha: float
    portfolio_factor_betas: Dict[str, float]
    cumulative_excess_return: float
    cumulative_factor_contributions: Dict[str, float]
    cumulative_alpha_component: float
    cumulative_residual: float
    data_warnings: List[str] = field(default_factory=list)
    available: bool = False


def build_factor_attribution(
    positions: List[Dict[str, Any]],
    fetch_ohlcv_fn: Callable[..., Optional[pd.DataFrame]],
    attr_start: date,
    attr_end: date,
    period_years: int = 3,
) -> FactorAttributionResult:
    """
    Estimate β and α on Ken French dates strictly before ``attr_start``; attribute
    portfolio daily excess returns over [attr_start, attr_end] to Σ β_k F_{k,t} + α,
    with residual = realized excess minus that explained component (constant weights).
    """
    warnings: List[str] = []
    empty = FactorAttributionResult(
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
        data_warnings=warnings,
        available=False,
    )

    pairs: List[Tuple[str, float]] = []
    for p in positions:
        t = (p.get("ticker") or "").upper().strip()
        v = float(p.get("current_value") or 0)
        if t and v > 0:
            pairs.append((t, v))
    if not pairs:
        warnings.append("No positions for attribution.")
        return empty

    total = sum(v for _, v in pairs)
    if total <= 0:
        return empty
    weights = {t: v / total for t, v in pairs}

    ff, w_ff = load_ff5_factors()
    warnings.extend(w_ff)
    if ff is None or ff.empty:
        warnings.append("Fama-French factors unavailable.")
        return empty

    ts_a0 = pd.Timestamp(attr_start)
    ts_a1 = pd.Timestamp(attr_end)
    if ts_a0 > ts_a1:
        warnings.append("Attribution start after end.")
        return empty

    prior = ff.index[ff.index < ts_a0]
    if len(prior) == 0:
        warnings.append("No factor dates before attribution start (cannot estimate loadings).")
        return empty
    est_end = prior.max()
    est_start = est_end - pd.Timedelta(days=365 * int(period_years) + 30)
    ff_est = ff.loc[(ff.index >= est_start) & (ff.index <= est_end)]

    ff_attr = ff.loc[(ff.index >= ts_a0) & (ff.index <= ts_a1)]
    if ff_attr.empty:
        warnings.append("No factor rows in attribution window.")
        return empty

    need_start = min(ff_est.index.min(), ff_attr.index.min())
    fetch_years = _fetch_years_for_span(need_start)

    per_betas: Dict[str, Dict[str, float]] = {}
    per_alpha: Dict[str, float] = {}
    per_r2: Dict[str, float] = {}
    per_n: Dict[str, int] = {}
    included_weights: Dict[str, float] = {}
    returns_map: Dict[str, pd.Series] = {}

    for ticker, w in weights.items():
        df = fetch_ohlcv_fn(ticker, fetch_years)
        r = _returns_from_ohlcv(df)
        if r is None:
            warnings.append(f"{ticker}: insufficient OHLCV for attribution regression.")
            continue
        returns_map[ticker] = r
        betas, r2, n_obs = _regress_one_ticker(r, ff_est)
        if betas is None or r2 is None:
            warnings.append(f"{ticker}: overlap with estimation factors < {MIN_OVERLAP_DAYS} days.")
            continue
        per_alpha[ticker] = float(betas.get("_alpha", 0.0))
        clean = {k: v for k, v in betas.items() if k != "_alpha"}
        per_betas[ticker] = clean
        per_r2[ticker] = float(r2)
        per_n[ticker] = int(n_obs)
        if r2 < 0.05:
            warnings.append(f"{ticker}: low R² ({r2:.2f}) — loadings noisy.")
        included_weights[ticker] = w

    sw = sum(included_weights.values())
    if sw <= 0 or not included_weights:
        warnings.append("No tickers produced an estimation-window regression.")
        return FactorAttributionResult(
            attribution_start=attr_start.isoformat(),
            attribution_end=attr_end.isoformat(),
            estimation_start=ff_est.index.min().strftime("%Y-%m-%d") if len(ff_est) else None,
            estimation_end=est_end.strftime("%Y-%m-%d") if est_end is not None else None,
            n_attribution_days=0,
            portfolio_alpha=0.0,
            portfolio_factor_betas={},
            cumulative_excess_return=0.0,
            cumulative_factor_contributions={},
            cumulative_alpha_component=0.0,
            cumulative_residual=0.0,
            data_warnings=warnings,
            available=False,
        )

    port_beta: Dict[str, float] = {f: 0.0 for f in FACTOR_COLS}
    port_alpha = 0.0
    for t, w in included_weights.items():
        nw = w / sw
        port_alpha += nw * per_alpha[t]
        for f in FACTOR_COLS:
            port_beta[f] += nw * per_betas[t][f]

    common_dates: List[pd.Timestamp] = []
    for d in ff_attr.index:
        if all(d in returns_map[t].index for t in included_weights):
            common_dates.append(d)

    if len(common_dates) < ATTR_MIN_DAYS:
        warnings.append(
            f"Only {len(common_dates)} days with full book returns in window "
            f"(need {ATTR_MIN_DAYS}+ for a stable strip)."
        )
        return FactorAttributionResult(
            attribution_start=attr_start.isoformat(),
            attribution_end=attr_end.isoformat(),
            estimation_start=ff_est.index.min().strftime("%Y-%m-%d") if len(ff_est) else None,
            estimation_end=est_end.strftime("%Y-%m-%d"),
            n_attribution_days=len(common_dates),
            portfolio_alpha=port_alpha,
            portfolio_factor_betas=dict(port_beta),
            cumulative_excess_return=0.0,
            cumulative_factor_contributions={f: 0.0 for f in FACTOR_COLS},
            cumulative_alpha_component=0.0,
            cumulative_residual=0.0,
            data_warnings=warnings,
            available=False,
        )

    excess_list: List[float] = []
    explained_list: List[float] = []
    for d in common_dates:
        r_p = sum((included_weights[t] / sw) * float(returns_map[t].loc[d]) for t in included_weights)
        rf = float(ff.loc[d, "RF"])
        excess_list.append(r_p - rf)
        fac_part = sum(port_beta[f] * float(ff.loc[d, f]) for f in FACTOR_COLS)
        explained_list.append(port_alpha + fac_part)

    n_days = len(common_dates)
    residual_list = [excess_list[i] - explained_list[i] for i in range(n_days)]
    cum_fac: Dict[str, float] = {}
    for f in FACTOR_COLS:
        cum_fac[f] = port_beta[f] * sum(float(ff.loc[d, f]) for d in common_dates)
    cum_alpha_comp = port_alpha * n_days
    cum_excess = sum(excess_list)
    cum_residual = sum(residual_list)

    return FactorAttributionResult(
        attribution_start=attr_start.isoformat(),
        attribution_end=attr_end.isoformat(),
        estimation_start=ff_est.index.min().strftime("%Y-%m-%d") if len(ff_est) else None,
        estimation_end=est_end.strftime("%Y-%m-%d"),
        n_attribution_days=n_days,
        portfolio_alpha=port_alpha,
        portfolio_factor_betas=dict(port_beta),
        cumulative_excess_return=cum_excess,
        cumulative_factor_contributions=cum_fac,
        cumulative_alpha_component=cum_alpha_comp,
        cumulative_residual=cum_residual,
        data_warnings=warnings,
        available=True,
    )
