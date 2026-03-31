"""
Fama-French 5-factor exposure (PORT depth): Ken French daily factors + per-ticker
OLS vs portfolio value-weighted factor betas.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Match portfolio_insights beta stability window
MIN_OVERLAP_DAYS = 120

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

    y = (aligned["stock_r"] - aligned["RF"]).values.reshape(-1, 1)
    X = aligned[FACTOR_COLS].values
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y.ravel())
    r2 = float(reg.score(X, y.ravel()))
    betas = {FACTOR_COLS[i]: float(reg.coef_[i]) for i in range(len(FACTOR_COLS))}
    betas["_alpha"] = float(reg.intercept_)
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
