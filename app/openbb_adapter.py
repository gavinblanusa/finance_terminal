"""
OpenBB adapter for Gavin Financial Terminal.

Provides a single data layer over OpenBB (OHLCV, quote, profile, fundamentals, etc.).
Returns None on any exception or when OpenBB is not installed so callers can fall back
to existing yfinance / api_clients / direct API paths.

Ensure project .env is loaded before first use (e.g. in main.py via python-dotenv).
OpenBB reads API keys from process environment; POLYGON_API_KEY is set from MASSIVE_API_KEY if needed.
"""

import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

def _valid_float(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float):
        return v == v  # exclude nan
    return True

# Project root (app lives in Invest/app/)
_ROOT = Path(__file__).resolve().parent.parent
# Load .env so OpenBB can see API keys (if not already loaded by app)
try:
    from dotenv import load_dotenv
    _env_path = _ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# Map MASSIVE_API_KEY to POLYGON_API_KEY for OpenBB (OpenBB expects POLYGON_API_KEY)
if os.environ.get("MASSIVE_API_KEY") and not os.environ.get("POLYGON_API_KEY"):
    os.environ["POLYGON_API_KEY"] = os.environ["MASSIVE_API_KEY"]

# Feature flag: set USE_OPENBB=false to disable OpenBB and use fallbacks only
USE_OPENBB = os.environ.get("USE_OPENBB", "true").lower() in ("true", "1", "yes")

_obb = None


def _get_obb():
    """Lazy load OpenBB to avoid import errors when OpenBB is not installed."""
    global _obb
    if _obb is not None:
        return _obb
    if not USE_OPENBB:
        return None
    try:
        from openbb import obb
        _obb = obb
        return _obb
    except ImportError:
        return None


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
    obb = _get_obb()
    if obb is None:
        return None
    try:
        import pandas as pd
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        # Try polygon first, then yfinance
        for provider in ("polygon", "yfinance", "fmp"):
            try:
                result = obb.equity.price.historical(
                    symbol=ticker,
                    start_date=start_str,
                    end_date=end_str,
                    provider=provider,
                )
                if result is None or not getattr(result, "results", None):
                    continue
                df = result.to_df()
                if df is None or df.empty:
                    continue
                # OpenBB returns date, open, high, low, close, volume (lowercase)
                # Normalize to market_data cache format: index=date, columns Open, High, Low, Close, Volume
                col_map = {c: c.capitalize() for c in df.columns if isinstance(c, str) and c.lower() in ("open", "high", "low", "close", "volume")}
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
                    continue
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.dropna(inplace=True)
                if len(df) >= 50:
                    return df
            except Exception:
                continue
        return None
    except Exception as e:
        print(f"[OpenBB] fetch_ohlcv_openbb error for {ticker}: {e}")
        return None


def fetch_quote_openbb(ticker: str) -> Optional[Decimal]:
    """
    Fetch latest price (quote) from OpenBB equity.price.quote.

    Returns price as Decimal or None on failure or when OpenBB is unavailable.
    """
    obb = _get_obb()
    if obb is None:
        return None
    try:
        result = obb.equity.price.quote(symbol=ticker)
        if result is None or not getattr(result, "results", None):
            return None
        df = result.to_df()
        if df is None or df.empty:
            return None
        # Quote typically has last_price, close, or similar
        for col in ("last_price", "close", "Close", "lastPrice"):
            if col in df.columns:
                val = df[col].iloc[0]
                if val is not None and getattr(val, "__float__", None):
                    return Decimal(str(round(float(val), 2)))
        # Try first numeric column
        for c in df.columns:
            val = df[c].iloc[0]
            if isinstance(val, (int, float)) and val > 0:
                return Decimal(str(round(float(val), 2)))
        return None
    except Exception as e:
        print(f"[OpenBB] fetch_quote_openbb error for {ticker}: {e}")
        return None


def fetch_profile_openbb(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company profile from OpenBB equity.profile.
    Returns dict with sector, industry, description, website, employees, ceo, data_source; or None.
    """
    obb = _get_obb()
    if obb is None:
        return None
    try:
        result = obb.equity.profile(symbol=ticker)
        if result is None or not getattr(result, "results", None):
            return None
        df = result.to_df()
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        # Map OpenBB fields (snake_case or camelCase) to our shape
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
            "employees": int(employees) if employees is not None and isinstance(employees, (int, float)) and _valid_float(employees) else (int(employees) if isinstance(employees, str) and employees.isdigit() else None),
            "ceo": str(ceo) if ceo is not None else None,
            "data_source": "openbb",
        }
    except Exception as e:
        print(f"[OpenBB] fetch_profile_openbb error for {ticker}: {e}")
        return None


def fetch_fundamentals_openbb(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamentals/ratios from OpenBB equity.fundamental.
    Returns dict with revenue_ttm, gross_margin, operating_margin, net_margin, roe, roa, data_source; or None.
    """
    obb = _get_obb()
    if obb is None:
        return None
    try:
        result = {}
        # Ratios (if available)
        try:
            r = obb.equity.fundamental.ratios(symbol=ticker)
            if r and getattr(r, "results", None):
                df = r.to_df()
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    for our_key, keys in [
                        ("gross_margin", ["gross_profit_margin", "grossProfitMargin"]),
                        ("operating_margin", ["operating_profit_margin", "operatingProfitMargin"]),
                        ("net_margin", ["net_profit_margin", "netProfitMargin"]),
                        ("roe", ["return_on_equity", "returnOnEquity"]),
                        ("roa", ["return_on_assets", "returnOnAssets"]),
                    ]:
                        for k in keys:
                            if k in row.index:
                                v = row[k]
                                if v is not None and isinstance(v, (int, float)) and _valid_float(v):
                                    if "margin" in our_key and abs(v) > 1 and abs(v) <= 100:
                                        v = v / 100.0
                                    result[our_key] = float(v)
                                    break
        except Exception:
            pass
        # Revenue from income statement
        try:
            inc = obb.equity.fundamental.income(symbol=ticker, period="quarter", limit=4)
            if inc and getattr(inc, "results", None):
                df = inc.to_df()
                if df is not None and not df.empty:
                    rev_col = None
                    for c in df.columns:
                        if c and "revenue" in str(c).lower():
                            rev_col = c
                            break
                    if rev_col is not None:
                        total = df[rev_col].sum()
                        if total and total > 0:
                            result["revenue_ttm"] = float(total)
        except Exception:
            pass
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
    obb = _get_obb()
    if obb is None:
        return []
    try:
        from datetime import datetime as dt, timedelta
        to_date = dt.now().date()
        from_date = to_date - timedelta(days=14)
        result = obb.news.company(symbol=ticker, start_date=from_date.isoformat(), end_date=to_date.isoformat(), limit=limit)
        if result is None or not getattr(result, "results", None):
            return []
        df = result.to_df()
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
            out.append({
                "headline": headline,
                "url": url or "",
                "source": source or "",
                "datetime": dt_val,
            })
        return out
    except Exception as e:
        print(f"[OpenBB] fetch_news_openbb error for {ticker}: {e}")
        return []


def fetch_historical_price_openbb(ticker: str, target_date: date) -> Optional[float]:
    """
    Fetch closing price on or near target_date from OpenBB. Used for IPO vintage price.
    """
    obb = _get_obb()
    if obb is None:
        return None
    try:
        start = target_date
        end = target_date
        result = obb.equity.price.historical(symbol=ticker, start_date=start.isoformat(), end_date=end.isoformat())
        if result is None or not getattr(result, "results", None):
            return None
        df = result.to_df()
        if df is None or df.empty:
            return None
        close_col = None
        for c in df.columns:
            if str(c).lower() == "close":
                close_col = c
                break
        if close_col is None:
            return None
        val = df[close_col].dropna().iloc[-1]
        if val is not None and float(val) > 0:
            return float(val)
        return None
    except Exception as e:
        print(f"[OpenBB] fetch_historical_price_openbb error for {ticker}: {e}")
        return None
