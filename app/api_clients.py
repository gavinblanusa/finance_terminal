"""
API Clients - Polygon, Twelve Data, EODHD backup data sources.

Centralized module for fetching stock prices and OHLCV data from backup APIs
when Yahoo Finance is unavailable or rate-limited.

Note: OpenBB (openbb_adapter) is preferred when available; this module remains
as fallback when OpenBB is not installed or returns no data.
"""

import os
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# API Keys
MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '')
EODHD_API_KEY = os.getenv('EODHD_API_KEY', '')

# API Base URLs
POLYGON_BASE_URL = "https://api.polygon.io"
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
EODHD_BASE_URL = "https://eodhd.com/api"

# Rate limit tracking
_last_polygon_call = 0.0
_last_twelvedata_call = 0.0
POLYGON_MIN_INTERVAL = 12.0  # 5 calls/min = 12s between calls
TWELVEDATA_MIN_INTERVAL = 8.0  # 8 credits/min, space calls

# EODHD daily limit (free tier: 20 calls/day)
EODHD_DAILY_LIMIT = 20
_eodhd_call_count = 0
_eodhd_reset_date: Optional[date] = None
# Project root (caches live at Invest/ root)
_ROOT = Path(__file__).resolve().parent.parent
EODHD_COUNTER_PATH = _ROOT / ".eodhd_counter.json"


def _to_eodhd_ticker(ticker: str) -> str:
    """Convert ticker to EODHD format (e.g., AAPL -> AAPL.US)."""
    ticker = ticker.upper().strip()
    # EODHD uses TICKER.US for US stocks
    # Handle special cases - Polygon/Twelve Data use BRK-B, EODHD may use BRK-B.US
    if '.' in ticker and not ticker.endswith('.US'):
        # e.g. BRK.B -> BRK-B.US
        ticker = ticker.replace('.', '-')
    if not ticker.endswith('.US') and not ticker.endswith('.') and '.' not in ticker:
        ticker = f"{ticker}.US"
    return ticker


def _to_polygon_ticker(ticker: str) -> str:
    """Convert ticker for Polygon API (uses BRK.B, BRK-B format)."""
    ticker = ticker.upper().strip()
    # Polygon typically uses BRK.B for Berkshire B
    corrections = {'BRK-B': 'BRK.B', 'BRKA': 'BRK.A', 'BRK-A': 'BRK.A'}
    return corrections.get(ticker, ticker)


def _wait_polygon_rate_limit():
    """Wait if needed to respect Polygon rate limit (5 calls/min)."""
    global _last_polygon_call
    if _last_polygon_call > 0:
        elapsed = time.time() - _last_polygon_call
        if elapsed < POLYGON_MIN_INTERVAL:
            wait_time = POLYGON_MIN_INTERVAL - elapsed
            print(f"[Polygon] Waiting {wait_time:.1f}s for rate limit...")
            time.sleep(wait_time)
    _last_polygon_call = time.time()


def _wait_twelvedata_rate_limit():
    """Wait if needed to respect Twelve Data rate limit."""
    global _last_twelvedata_call
    if _last_twelvedata_call > 0:
        elapsed = time.time() - _last_twelvedata_call
        if elapsed < TWELVEDATA_MIN_INTERVAL:
            wait_time = TWELVEDATA_MIN_INTERVAL - elapsed
            print(f"[Twelve Data] Waiting {wait_time:.1f}s for rate limit...")
            time.sleep(wait_time)
    _last_twelvedata_call = time.time()


def _can_use_eodhd() -> bool:
    """Check if EODHD can be used (under daily limit)."""
    global _eodhd_call_count, _eodhd_reset_date
    today = date.today()
    if _eodhd_reset_date != today:
        _eodhd_call_count = 0
        _eodhd_reset_date = today
    return _eodhd_call_count < EODHD_DAILY_LIMIT


def _record_eodhd_call():
    """Record an EODHD API call."""
    global _eodhd_call_count
    _eodhd_call_count += 1


# -----------------------------------------------------------------------------
# Single Price Fetching
# -----------------------------------------------------------------------------

def fetch_price_polygon(ticker: str) -> Optional[Decimal]:
    """
    Fetch previous close price from Polygon (Massive) API.
    Free tier: 5 calls/min.
    """
    if not MASSIVE_API_KEY:
        return None
    try:
        _wait_polygon_rate_limit()
        poly_ticker = _to_polygon_ticker(ticker)
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{poly_ticker}/prev"
        params = {"apiKey": MASSIVE_API_KEY, "adjusted": "true"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get("resultsCount", 0) > 0:
            results = data.get("results", [])
            if results:
                close_price = results[0].get("c")
                if close_price and float(close_price) > 0:
                    price = Decimal(str(round(float(close_price), 2)))
                    print(f"✓ Polygon: {ticker} = ${price}")
                    return price
        elif data.get("error"):
            print(f"[Polygon] {ticker}: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"[Polygon] Exception for {ticker}: {e}")
    return None


def fetch_price_twelvedata(ticker: str) -> Optional[Decimal]:
    """
    Fetch latest price from Twelve Data API.
    Free tier: 8 credits/min, 800/day. Price endpoint = 1 credit.
    """
    if not TWELVEDATA_API_KEY:
        return None
    try:
        _wait_twelvedata_rate_limit()
        url = f"{TWELVEDATA_BASE_URL}/price"
        params = {"symbol": ticker.upper(), "apikey": TWELVEDATA_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get("price"):
            price_val = float(data["price"])
            if price_val > 0:
                price = Decimal(str(round(price_val, 2)))
                print(f"✓ Twelve Data: {ticker} = ${price}")
                return price
        elif data.get("code"):
            print(f"[Twelve Data] {ticker}: {data.get('message', data.get('code'))}")
    except Exception as e:
        print(f"[Twelve Data] Exception for {ticker}: {e}")
    return None


def fetch_price_eodhd(ticker: str) -> Optional[Decimal]:
    """
    Fetch last close price from EODHD API.
    Free tier: 20 calls/day.
    """
    if not EODHD_API_KEY or not _can_use_eodhd():
        return None
    try:
        eodhd_ticker = _to_eodhd_ticker(ticker)
        url = f"{EODHD_BASE_URL}/eod/{eodhd_ticker}"
        params = {"api_token": EODHD_API_KEY, "fmt": "json", "filter": "last_close"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            text = response.text.strip()
            if text and text.replace('.', '').replace('-', '').isdigit():
                price_val = float(text)
                if price_val > 0:
                    _record_eodhd_call()
                    price = Decimal(str(round(price_val, 2)))
                    print(f"✓ EODHD: {ticker} = ${price}")
                    return price
            elif text.startswith('{'):
                data = response.json()
                if isinstance(data, (int, float)) and data > 0:
                    _record_eodhd_call()
                    return Decimal(str(round(float(data), 2)))
                close = data.get("close") if isinstance(data, dict) else None
                if close is not None and float(close) > 0:
                    _record_eodhd_call()
                    return Decimal(str(round(float(close), 2)))
    except Exception as e:
        print(f"[EODHD] Exception for {ticker}: {e}")
    return None


# -----------------------------------------------------------------------------
# OHLCV History Fetching
# -----------------------------------------------------------------------------

def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has Open, High, Low, Close, Volume columns and Date index."""
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()
    df = df[required].copy()
    df.dropna(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_ohlcv_polygon(ticker: str, from_date: date, to_date: date) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV history from Polygon aggregates API.
    Returns DataFrame with Open, High, Low, Close, Volume indexed by Date.
    """
    if not MASSIVE_API_KEY:
        return None
    try:
        _wait_polygon_rate_limit()
        poly_ticker = _to_polygon_ticker(ticker)
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{poly_ticker}/range/1/day/{from_date}/{to_date}"
        params = {"apiKey": MASSIVE_API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if response.status_code != 200 or data.get("resultsCount", 0) == 0:
            return None
        results = data.get("results", [])
        if not results:
            return None
        rows = []
        for r in results:
            ts_ms = r.get("t")
            if ts_ms:
                dt = datetime.fromtimestamp(ts_ms / 1000.0)
                rows.append({
                    "Date": dt,
                    "Open": r.get("o"),
                    "High": r.get("h"),
                    "Low": r.get("l"),
                    "Close": r.get("c"),
                    "Volume": r.get("v", 0)
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df.set_index("Date", inplace=True)
        return _normalize_ohlcv_df(df)
    except Exception as e:
        print(f"[Polygon OHLCV] Exception for {ticker}: {e}")
    return None


def fetch_ohlcv_twelvedata(ticker: str, days: int = 730) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV history from Twelve Data time_series API.
    """
    if not TWELVEDATA_API_KEY:
        return None
    try:
        _wait_twelvedata_rate_limit()
        url = f"{TWELVEDATA_BASE_URL}/time_series"
        params = {
            "symbol": ticker.upper(),
            "interval": "1day",
            "outputsize": min(days, 5000),
            "apikey": TWELVEDATA_API_KEY,
            "format": "JSON"
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            return None
        values = data.get("values", [])
        if not values:
            return None
        rows = []
        for v in values:
            dt_str = v.get("datetime")
            if not dt_str:
                continue
            dt = pd.to_datetime(dt_str)
            rows.append({
                "Date": dt,
                "Open": float(v.get("open", 0)),
                "High": float(v.get("high", 0)),
                "Low": float(v.get("low", 0)),
                "Close": float(v.get("close", 0)),
                "Volume": float(v.get("volume", 0))
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return _normalize_ohlcv_df(df)
    except Exception as e:
        print(f"[Twelve Data OHLCV] Exception for {ticker}: {e}")
    return None


def fetch_ohlcv_eodhd(ticker: str, from_date: date, to_date: date) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV history from EODHD EOD API.
    Free tier: 1 year history for EOD. May truncate if range > 1 year.
    """
    if not EODHD_API_KEY or not _can_use_eodhd():
        return None
    try:
        eodhd_ticker = _to_eodhd_ticker(ticker)
        url = f"{EODHD_BASE_URL}/eod/{eodhd_ticker}"
        params = {
            "api_token": EODHD_API_KEY,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "period": "d",
            "fmt": "json"
        }
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None
        data = response.json()
        _record_eodhd_call()
        if not data:
            return None
        if isinstance(data, dict) and "date" in data:
            data = [data]
        rows = []
        for row in (data if isinstance(data, list) else []):
            dt_str = row.get("date")
            if not dt_str:
                continue
            dt = pd.to_datetime(dt_str)
            close = row.get("close") or row.get("adjusted_close")
            if close is None:
                continue
            rows.append({
                "Date": dt,
                "Open": float(row.get("open", close)),
                "High": float(row.get("high", close)),
                "Low": float(row.get("low", close)),
                "Close": float(close),
                "Volume": float(row.get("volume", 0))
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return _normalize_ohlcv_df(df)
    except Exception as e:
        print(f"[EODHD OHLCV] Exception for {ticker}: {e}")
    return None


# -----------------------------------------------------------------------------
# Historical Close (specific date) - for IPO vintage, etc.
# -----------------------------------------------------------------------------

def fetch_historical_close_polygon(ticker: str, target_date: date) -> Optional[float]:
    """Fetch closing price on or near target_date from Polygon."""
    if not MASSIVE_API_KEY:
        return None
    try:
        _wait_polygon_rate_limit()
        poly_ticker = _to_polygon_ticker(ticker)
        start = (target_date - timedelta(days=7)).isoformat()
        end = (target_date + timedelta(days=7)).isoformat()
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{poly_ticker}/range/1/day/{start}/{end}"
        params = {"apiKey": MASSIVE_API_KEY, "adjusted": "true", "sort": "asc"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code != 200 or data.get("resultsCount", 0) == 0:
            return None
        results = data.get("results", [])
        target_ts = datetime.combine(target_date, datetime.min.time()).timestamp() * 1000
        best = None
        best_diff = float('inf')
        for r in results:
            ts = r.get("t")
            c = r.get("c")
            if ts and c and float(c) > 0:
                diff = abs(ts - target_ts)
                if ts <= target_ts + 86400000:
                    if diff < best_diff:
                        best_diff = diff
                        best = float(c)
        return round(best, 2) if best else None
    except Exception as e:
        print(f"[Polygon historical] Exception for {ticker}: {e}")
    return None


def fetch_historical_close_twelvedata(ticker: str, target_date: date) -> Optional[float]:
    """Fetch closing price on or near target_date from Twelve Data."""
    if not TWELVEDATA_API_KEY:
        return None
    try:
        _wait_twelvedata_rate_limit()
        start = (target_date - timedelta(days=7)).isoformat()
        end = (target_date + timedelta(days=7)).isoformat()
        url = f"{TWELVEDATA_BASE_URL}/time_series"
        params = {
            "symbol": ticker.upper(),
            "interval": "1day",
            "start_date": start,
            "end_date": end,
            "apikey": TWELVEDATA_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            return None
        values = data.get("values", [])
        target_dt = datetime.combine(target_date, datetime.min.time())
        best = None
        best_diff = timedelta(days=999)
        for v in values:
            dt_str = v.get("datetime")
            close = v.get("close")
            if not dt_str or not close:
                continue
            try:
                bar_dt = pd.to_datetime(dt_str).to_pydatetime()
                if hasattr(bar_dt, 'replace') and hasattr(bar_dt, 'tzinfo'):
                    if bar_dt.tzinfo:
                        bar_dt = bar_dt.replace(tzinfo=None)
                diff = abs((bar_dt - target_dt).days)
                if bar_dt <= target_dt + timedelta(days=1) and diff < abs(best_diff.days):
                    best_diff = timedelta(days=diff)
                    best = float(close)
            except Exception:
                continue
        return round(best, 2) if best else None
    except Exception as e:
        print(f"[Twelve Data historical] Exception for {ticker}: {e}")
    return None


def fetch_historical_close_eodhd(ticker: str, target_date: date) -> Optional[float]:
    """Fetch closing price on or near target_date from EODHD."""
    if not EODHD_API_KEY or not _can_use_eodhd():
        return None
    try:
        eodhd_ticker = _to_eodhd_ticker(ticker)
        start = (target_date - timedelta(days=14)).isoformat()
        end = (target_date + timedelta(days=7)).isoformat()
        url = f"{EODHD_BASE_URL}/eod/{eodhd_ticker}"
        params = {
            "api_token": EODHD_API_KEY,
            "from": start,
            "to": end,
            "period": "d",
            "fmt": "json"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        _record_eodhd_call()
        if not data:
            return None
        if isinstance(data, dict) and "date" in data:
            data = [data]
        target_str = target_date.isoformat()
        best = None
        best_date = None
        for row in (data if isinstance(data, list) else []):
            dt_str = row.get("date")
            close = row.get("close") or row.get("adjusted_close")
            if not dt_str or close is None or float(close) <= 0:
                continue
            if dt_str <= target_str:
                if best_date is None or dt_str > best_date:
                    best_date = dt_str
                    best = float(close)
        return round(best, 2) if best else None
    except Exception as e:
        print(f"[EODHD historical] Exception for {ticker}: {e}")
    return None


# -----------------------------------------------------------------------------
# Twelve Data Quote - for ticker info fallback (basic fields)
# -----------------------------------------------------------------------------

def fetch_quote_twelvedata(ticker: str) -> Optional[dict]:
    """
    Fetch quote from Twelve Data (close, volume, 52w high/low, etc.).
    Returns dict with keys: close, open, high, low, volume, previous_close, percent_change.
    """
    if not TWELVEDATA_API_KEY:
        return None
    try:
        _wait_twelvedata_rate_limit()
        url = f"{TWELVEDATA_BASE_URL}/quote"
        params = {"symbol": ticker.upper(), "apikey": TWELVEDATA_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get("symbol"):
            return {
                "close": float(data.get("close", 0)) if data.get("close") else None,
                "open": float(data.get("open", 0)) if data.get("open") else None,
                "high": float(data.get("high", 0)) if data.get("high") else None,
                "low": float(data.get("low", 0)) if data.get("low") else None,
                "volume": float(data.get("volume", 0)) if data.get("volume") else None,
                "previous_close": float(data.get("previous_close", 0)) if data.get("previous_close") else None,
                "percent_change": float(data.get("percent_change", 0)) if data.get("percent_change") else None,
            }
    except Exception as e:
        print(f"[Twelve Data quote] Exception for {ticker}: {e}")
    return None
