"""
S&P 500 sector SPDR proxies: price returns and relative performance vs SPY.
Educational: delayed quotes, not a fundamental sector model.
"""

from __future__ import annotations

import pandas as pd

SPDR_SECTORS: list[tuple[str, str]] = [
    ("XLE", "Energy"),
    ("XLF", "Financials"),
    ("XLK", "Tech"),
    ("XLV", "Health care"),
    ("XLY", "Cons. disc."),
    ("XLP", "Cons. staples"),
    ("XLI", "Industrials"),
    ("XLU", "Utilities"),
    ("XLB", "Materials"),
    ("XLRE", "Real estate"),
    ("XLC", "Comm. services"),
]

BENCHMARK = "SPY"
PERIOD_1M = "1mo"
PERIOD_PAIR = "3mo"

SECTOR_BENCHMARK_CHOICES: list[str] = sorted(
    {BENCHMARK, *[s for s, _ in SPDR_SECTORS]}
)
_ALLOWED = frozenset(SECTOR_BENCHMARK_CHOICES)


def _ret_first_last(c: pd.Series) -> float | None:
    c = c.dropna()
    if c.shape[0] < 2:
        return None
    a, b = float(c.iloc[0]), float(c.iloc[-1])
    if a == 0:
        return None
    return (b / a - 1.0) * 100.0


def _ret_last_bars(c: pd.Series, bars: int) -> float | None:
    c = c.dropna()
    if c.shape[0] < bars + 1:
        return _ret_first_last(c) if c.shape[0] >= 2 else None
    a, b = float(c.iloc[-(bars + 1)]), float(c.iloc[-1])
    if a == 0:
        return None
    return (b / a - 1.0) * 100.0


def _history_closes(ticker: str) -> pd.Series:
    import yfinance as yf

    h = yf.Ticker(ticker).history(period=PERIOD_1M, interval="1d", auto_adjust=True)
    if h is None or h.empty or "Close" not in h.columns:
        return pd.Series(dtype=float)
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s = s.tz_localize(None) if s.index.tz is not None else s
    return s


def build_spdr_momentum_table() -> tuple[pd.DataFrame, str | None]:
    """
    Return table: Symbol, Sector, 1d %, 5d %, 1m %, rel 1m vs SPY, Rank, and optional error.
    """
    err: str | None = None
    spy = _history_closes(BENCHMARK)
    rs1m = _ret_first_last(spy)
    rows: list[dict] = []
    for sym, name in SPDR_SECTORS:
        c = _history_closes(sym)
        r1d = _ret_last_bars(c, 1)
        r5d = _ret_last_bars(c, 5)
        r1m = _ret_first_last(c)
        rrel = (r1m - rs1m) if (r1m is not None and rs1m is not None) else None
        rows.append(
            {
                "Symbol": sym,
                "Sector": name,
                "1d %": r1d,
                "5d %": r5d,
                "1m %": r1m,
                "rel 1m vs SPY (pp)": rrel,
            }
        )
    dfx = pd.DataFrame(rows)
    if dfx.empty:
        return dfx, "No sector data (check network / Yahoo)."
    dfx = dfx.sort_values("rel 1m vs SPY (pp)", ascending=False, na_position="last")
    dfx.insert(0, "Rank", range(1, len(dfx) + 1))
    dfx = dfx.reset_index(drop=True)
    return dfx, err


def build_pair_ratio_frame(sym_a: str, sym_b: str, period: str = PERIOD_PAIR) -> tuple[pd.DataFrame | None, str | None]:
    """Cumulative (sym_a / sym_b) rebased to 1.0 at first aligned date."""
    import yfinance as yf

    a, b = sym_a.strip().upper(), sym_b.strip().upper()
    if a not in _ALLOWED or b not in _ALLOWED or a == b:
        return None, "Choose two distinct symbols from the sector list (and SPY)."
    for s in (a, b):
        if not s or len(s) > 6 or not s.replace("-", "").isalnum():
            return None, "Invalid symbol."

    ha = yf.Ticker(a).history(period=period, interval="1d", auto_adjust=True)
    hb = yf.Ticker(b).history(period=period, interval="1d", auto_adjust=True)
    if ha is None or ha.empty or hb is None or hb.empty:
        return None, "No history."
    ca, cb = ha["Close"].copy(), hb["Close"].copy()
    for s in (ca, cb):
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    m = pd.DataFrame({a: ca, b: cb}).dropna().sort_index()
    if m.empty:
        return None, "No overlapping dates."
    r = m[a] / m[b]
    r = r / r.iloc[0]
    return pd.DataFrame({f"{a} / {b} (rebased)": r}), None
