"""
OpenBB adapter for Gavin Financial Terminal.

Provides a single data layer over OpenBB (OHLCV, quote, profile, fundamentals, etc.).
Returns None on any exception or when OpenBB is not installed so callers can fall back
to existing yfinance / api_clients / direct API paths.

Lazy load, env, and provider chaining live in openbb_fetch.py.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from openbb_fetch import USE_OPENBB, _get_obb, run_provider_chain
from openbb_provider_registry import OPENBB_PROVIDER_CHAINS

# Re-export for tests or tools that introspect the adapter module
__all__ = [
    "USE_OPENBB",
    "fetch_ohlcv_openbb",
    "fetch_quote_openbb",
    "fetch_profile_openbb",
    "fetch_fundamentals_openbb",
    "fetch_news_openbb",
    "fetch_historical_price_openbb",
    "fetch_macro_data_openbb",
]

_MACRO_FRED_SERIES_IDS: Dict[str, str] = {
    "gdp": "GDP",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "treasury_10y": "DGS10",
    "m2": "WM2NS",
    "pmi": "M0204AM356PCEN",
    "retail_sales": "RSAFS",
    "consumer_sentiment": "UMCSENT",
}


def _macro_fred_ob_result_to_df(raw: Any, series_id: str) -> Optional["Any"]:
    """Normalize OpenBB FRED series OBBject to DataFrame: datetime index, column 'value'."""
    try:
        import pandas as pd

        df = raw.to_df()
        if df is None or df.empty:
            return None
        out = df.copy()
        if "date" in out.columns:
            out = out.set_index("date")
        elif "Date" in out.columns:
            out = out.set_index("Date")
        out.index = pd.to_datetime(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index.name = "date"
        val_col = None
        for c in ("value", series_id, series_id.upper(), "close"):
            if c in out.columns:
                val_col = c
                break
        if val_col is None:
            for c in out.columns:
                if out[c].dtype.kind in "fiu" or str(out[c].dtype).startswith("float"):
                    val_col = c
                    break
        if val_col is None:
            return None
        out = out[[val_col]].rename(columns={val_col: "value"})
        return out
    except Exception:
        return None


def _fetch_macro_via_pandas_datareader(metric: str) -> Optional["Any"]:
    """FRED series via pandas_datareader (legacy path; no OpenBB)."""
    try:
        import pandas_datareader.data as web

        if metric not in _MACRO_FRED_SERIES_IDS:
            return None
        sid = _MACRO_FRED_SERIES_IDS[metric]
        df = web.DataReader(sid, "fred", "2000-01-01")
        df = df.rename(columns={sid: "value"})
        df.index.name = "date"
        if metric == "treasury_10y":
            df = df.ffill()
        return df.dropna()
    except Exception as e:
        print(f"pandas_datareader failed for {metric}: {e}")
        return None


def _valid_float(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float):
        return v == v  # exclude nan
    return True


def fetch_ohlcv_openbb(
    ticker: str,
    start_date: date,
    end_date: date,
) -> Optional["Any"]:
    """
    Fetch OHLCV from OpenBB equity.price.historical.

    Returns DataFrame with columns Open, High, Low, Close, Volume and datetime index
    (timezone-naive), or None on failure or when OpenBB is unavailable.
    """
    try:
        import pandas as pd

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        def invoke(obb: Any, provider: str) -> Any:
            return obb.equity.price.historical(
                symbol=ticker,
                start_date=start_str,
                end_date=end_str,
                provider=provider,
            )

        res = run_provider_chain(
            "equity.price.historical",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.price.historical"],
            invoke,
        )
        if not res.ok or res.data is None:
            return None

        df = res.data.to_df()
        if df is None or df.empty:
            return None
        col_map = {
            c: c.capitalize()
            for c in df.columns
            if isinstance(c, str) and c.lower() in ("open", "high", "low", "close", "volume")
        }
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df = df.set_index("date")
        elif "Date" in df.columns:
            df = df.set_index("Date")
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for n in needed:
            if n not in df.columns and n.lower() in df.columns:
                df[n] = df[n.lower()]
        df = df[[c for c in needed if c in df.columns]]
        if len(df.columns) < 5:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.dropna(inplace=True)
        if len(df) >= 50:
            return df
        return None
    except Exception as e:
        print(f"[OpenBB] fetch_ohlcv_openbb error for {ticker}: {e}")
        return None


def _quote_df_to_decimal(df: Any) -> Optional[Decimal]:
    if df is None or df.empty:
        return None
    for col in ("last_price", "close", "Close", "lastPrice"):
        if col in df.columns:
            val = df[col].iloc[0]
            if val is not None and getattr(val, "__float__", None):
                return Decimal(str(round(float(val), 2)))
    for c in df.columns:
        val = df[c].iloc[0]
        if isinstance(val, (int, float)) and val > 0:
            return Decimal(str(round(float(val), 2)))
    return None


def fetch_quote_openbb(ticker: str) -> Optional[Decimal]:
    """
    Fetch latest price (quote) from OpenBB equity.price.quote.

    Returns price as Decimal or None on failure or when OpenBB is unavailable.
    """
    try:

        def invoke(obb: Any, provider: str) -> Any:
            return obb.equity.price.quote(symbol=ticker, provider=provider)

        res = run_provider_chain(
            "equity.price.quote",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.price.quote"],
            invoke,
        )
        if not res.ok or res.data is None:
            return None
        df = res.data.to_df()
        return _quote_df_to_decimal(df)
    except Exception as e:
        print(f"[OpenBB] fetch_quote_openbb error for {ticker}: {e}")
        return None


def _profile_df_to_dict(df: Any, ticker: str) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    row = df.iloc[0]

    def get_val(*keys):
        for k in keys:
            if k in row.index:
                v = row[k]
                if _valid_float(v) and v is not None:
                    return v
            k2 = k.replace("_", "")
            for c in row.index:
                if c and c.replace("_", "").lower() == k2.lower():
                    return row[c]
        return None

    sector = get_val("sector", "Sector")
    industry = get_val("industry", "Industry")
    description = get_val("description", "Description", "long_description")
    website = get_val("website", "Website", "url")
    employees = get_val("employees", "full_time_employees", "Full Time Employees")
    ceo = get_val("ceo", "ceo_name", "CEO")
    return {
        "ticker": ticker.upper().strip(),
        "sector": str(sector) if sector is not None else None,
        "industry": str(industry) if industry is not None else None,
        "description": str(description) if description is not None else None,
        "website": str(website) if website is not None else None,
        "employees": int(employees)
        if employees is not None and isinstance(employees, (int, float)) and _valid_float(employees)
        else (int(employees) if isinstance(employees, str) and employees.isdigit() else None),
        "ceo": str(ceo) if ceo is not None else None,
        "data_source": "openbb",
    }


def fetch_profile_openbb(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company profile from OpenBB equity.profile.
    Returns dict with sector, industry, description, website, employees, ceo, data_source; or None.
    """
    try:

        def invoke(obb: Any, provider: str) -> Any:
            return obb.equity.profile(symbol=ticker, provider=provider)

        res = run_provider_chain(
            "equity.profile",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.profile"],
            invoke,
        )
        if not res.ok or res.data is None:
            return None
        df = res.data.to_df()
        return _profile_df_to_dict(df, ticker)
    except Exception as e:
        print(f"[OpenBB] fetch_profile_openbb error for {ticker}: {e}")
        return None


def _merge_ratios_row(result: Dict[str, Any], df: Any) -> None:
    if df is None or df.empty:
        return
    row = df.iloc[0]
    for our_key, keys in [
        ("gross_margin", ["gross_profit_margin", "grossProfitMargin"]),
        ("operating_margin", ["operating_profit_margin", "operatingProfitMargin"]),
        ("net_margin", ["net_profit_margin", "netProfitMargin"]),
        ("roe", ["return_on_equity", "returnOnEquity"]),
        ("roa", ["return_on_assets", "returnOnAssets"]),
    ]:
        for k in keys:
            if k not in row.index:
                continue
            v = row[k]
            if v is None or not isinstance(v, (int, float)) or not _valid_float(v):
                continue
            if "margin" in our_key and abs(v) > 1 and abs(v) <= 100:
                v = v / 100.0
            result[our_key] = float(v)
            break


def _normalize_income_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


# Order matters: pick total/operating revenue before a bare "revenue" column that might be ambiguous.
_INCOME_REVENUE_COLUMN_ORDER: tuple[str, ...] = (
    "total_revenue",
    "totalrevenue",
    "total_revenues",
    "operating_revenue",
    "operatingrevenue",
    "operating_revenues",
    "revenue",
    "revenues",
    "net_sales",
    "netsales",
    "sales",
)


def _is_bad_income_revenue_column(norm: str) -> bool:
    """Columns whose name contains 'revenue' but are not quarterly total sales (e.g. COGS)."""
    bad = (
        "cost_of",
        "costof",
        "growth",
        "per_share",
        "pershare",
        "estimate",
        "margin",
        "deferred",
        "unearned",
        "yield",
        "cagr",
        "ratio",
    )
    return any(b in norm for b in bad)


def _pick_income_revenue_column(columns: Any) -> Optional[str]:
    """Choose the income-statement column to sum for trailing-four-quarter revenue."""
    col_list = [str(c) for c in columns]
    norms = {c: _normalize_income_column_name(c) for c in col_list}
    for target in _INCOME_REVENUE_COLUMN_ORDER:
        for orig, norm in norms.items():
            if norm != target or _is_bad_income_revenue_column(norm):
                continue
            return orig
    for orig, norm in norms.items():
        if "revenue" not in norm or _is_bad_income_revenue_column(norm):
            continue
        return orig
    return None


def _merge_income_ttm(result: Dict[str, Any], df: Any) -> None:
    if df is None or df.empty:
        return
    import pandas as pd

    rev_col = _pick_income_revenue_column(df.columns)
    if rev_col is None:
        return
    series = pd.to_numeric(df[rev_col], errors="coerce").dropna()
    if series.empty:
        return
    total = float(series.sum())
    if total > 0:
        result["revenue_ttm"] = total


def fetch_fundamentals_openbb(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamentals/ratios from OpenBB equity.fundamental.
    Returns dict with revenue_ttm, gross_margin, operating_margin, net_margin, roe, roa, data_source; or None.
    """
    if _get_obb() is None:
        return None
    try:
        result: Dict[str, Any] = {}

        def invoke_ratios(obb: Any, provider: str) -> Any:
            return obb.equity.fundamental.ratios(symbol=ticker, provider=provider)

        res_r = run_provider_chain(
            "equity.fundamental.ratios",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.fundamental.ratios"],
            invoke_ratios,
        )
        if res_r.ok and res_r.data is not None:
            _merge_ratios_row(result, res_r.data.to_df())

        def invoke_income(obb: Any, provider: str) -> Any:
            return obb.equity.fundamental.income(
                symbol=ticker, period="quarter", limit=4, provider=provider
            )

        res_i = run_provider_chain(
            "equity.fundamental.income",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.fundamental.income"],
            invoke_income,
        )
        if res_i.ok and res_i.data is not None:
            _merge_income_ttm(result, res_i.data.to_df())

        if not result:
            return None
        result["ticker"] = ticker.upper().strip()
        result["data_source"] = "openbb"
        return result
    except Exception as e:
        print(f"[OpenBB] fetch_fundamentals_openbb error for {ticker}: {e}")
        return None


def fetch_news_openbb(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch company news from OpenBB. Returns list of dicts with headline, url, source, datetime.
    """
    if _get_obb() is None:
        return []
    try:
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=14)

        def invoke(obb: Any, provider: str) -> Any:
            return obb.news.company(
                symbol=ticker,
                start_date=from_date.isoformat(),
                end_date=to_date.isoformat(),
                limit=limit,
                provider=provider,
            )

        res = run_provider_chain(
            "news.company",
            ticker,
            OPENBB_PROVIDER_CHAINS["news.company"],
            invoke,
        )
        if not res.ok or res.data is None:
            return []
        df = res.data.to_df()
        if df is None or df.empty:
            return []
        out = []
        for _, row in df.head(limit).iterrows():
            headline = None
            for k in ("title", "headline", "headlines"):
                if k in row.index and row[k]:
                    headline = str(row[k])
                    break
            if not headline:
                headline = "No title"
            url = None
            for k in ("url", "link", "article_url"):
                if k in row.index and row[k]:
                    url = str(row[k])
                    break
            source = None
            for k in ("source", "publisher", "site"):
                if k in row.index and row[k]:
                    source = str(row[k])
                    break
            dt_val = None
            for k in ("date", "published_at", "datetime", "publishedDate"):
                if k in row.index and row[k] is not None:
                    dt_val = row[k]
                    break
            out.append(
                {
                    "headline": headline,
                    "url": url or "",
                    "source": source or "",
                    "datetime": dt_val,
                }
            )
        return out
    except Exception as e:
        print(f"[OpenBB] fetch_news_openbb error for {ticker}: {e}")
        return []


def fetch_historical_price_openbb(ticker: str, target_date: date) -> Optional[float]:
    """
    Fetch closing price on or near target_date from OpenBB. Used for IPO vintage price.
    """
    try:
        from datetime import datetime, timedelta
        import pandas as pd
        start_date = target_date - timedelta(days=7)
        end_date = target_date + timedelta(days=7)

        def invoke(obb: Any, provider: str) -> Any:
            return obb.equity.price.historical(
                symbol=ticker,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                provider=provider,
            )

        res = run_provider_chain(
            "equity.price.historical_point",
            ticker,
            OPENBB_PROVIDER_CHAINS["equity.price.historical_point"],
            invoke,
        )
        if not res.ok or res.data is None:
            return None
        df = res.data.to_df()
        if df is None or df.empty:
            return None
            
        close_col = None
        for c in df.columns:
            if str(c).lower() == "close":
                close_col = c
                break
        if close_col is None:
            return None
            
        if df.index.tz is not None:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        target_dt = datetime.combine(target_date, datetime.min.time())
        valid_dates = df[df.index <= target_dt]
        
        if valid_dates.empty:
            val = df[close_col].dropna().iloc[0]
        else:
            val = valid_dates[close_col].dropna().iloc[-1]
            
        if val is not None and float(val) > 0:
            return float(val)
        return None
    except Exception as e:
        print(f"[OpenBB] fetch_historical_price_openbb error for {ticker}: {e}")
        return None


def fetch_macro_data_openbb(metric: str) -> Optional["Any"]:
    """
    Fetch macroeconomic FRED series for dashboard macro indicators.

    When ``USE_OPENBB`` is true and ``FRED_API_KEY`` is set in the environment, uses
    ``obb.economy.fred_series`` (OpenBB maps ``FRED_API_KEY`` to ``fred_api_key``).
    Otherwise, or on failure, falls back to **pandas_datareader**.

    metric: 'gdp', 'cpi', 'unemployment', 'treasury_10y', 'm2', 'pmi',
    'retail_sales', 'consumer_sentiment'

    Returns a DataFrame with a datetime index and a 'value' column, or None.
    """
    if metric not in _MACRO_FRED_SERIES_IDS:
        return None
    series_id = _MACRO_FRED_SERIES_IDS[metric]
    fred_key = (os.environ.get("FRED_API_KEY") or "").strip()

    if USE_OPENBB and fred_key:

        def invoke(obb: Any, provider: str) -> Any:
            return obb.economy.fred_series(
                symbol=series_id,
                start_date="2000-01-01",
                provider=provider,
            )

        try:
            res = run_provider_chain(
                f"economy.fred_series.{metric}",
                series_id,
                OPENBB_PROVIDER_CHAINS["economy.fred_series"],
                invoke,
            )
            if res.ok and res.data is not None:
                df = _macro_fred_ob_result_to_df(res.data, series_id)
                if df is not None and not df.empty:
                    if metric == "treasury_10y":
                        df = df.ffill()
                    return df.dropna()
        except Exception as e:
            print(f"[OpenBB] fetch_macro_data_openbb error for {metric}: {e}")

    return _fetch_macro_via_pandas_datareader(metric)
