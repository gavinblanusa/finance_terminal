"""
IPO Service - IPO Calendar & Vintage Performance Analysis.

This module provides:
- Finnhub API integration for upcoming IPO calendar
- Caching mechanism to avoid rate limits
- Vintage performance analysis (1, 2, 3-year returns)
- IPO anniversary detection for vintage alerts
"""

import os
import json
import requests
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from decimal import Decimal
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Finnhub API configuration
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Cache configuration
# Project root (caches live at Invest/ root)
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".ipo_cache"
IPO_CACHE_EXPIRY_HOURS = 6  # Cache IPO calendar for 6 hours


@dataclass
class IPOEntry:
    """Represents an upcoming or historical IPO entry."""
    ticker: str
    name: str
    exchange: str
    ipo_date: date
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    ipo_price: Optional[float] = None
    shares_offered: Optional[int] = None
    total_shares: Optional[int] = None
    status: str = "expected"  # expected, priced, withdrawn


@dataclass
class VintagePerformance:
    """Performance metrics at IPO vintages (1, 2, 3 years)."""
    ticker: str
    ipo_date: date
    ipo_price: float
    current_price: float
    year_1_return: Optional[float] = None  # Percentage return at 1 year
    year_2_return: Optional[float] = None  # Percentage return at 2 years
    year_3_return: Optional[float] = None  # Percentage return at 3 years
    year_1_status: str = "Pending"  # Pending, Calculated
    year_2_status: str = "Pending"
    year_3_status: str = "Pending"
    total_return: Optional[float] = None  # Current total return


def _get_cache_path(cache_name: str) -> Path:
    """Get cache file path."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{cache_name}.json"


def _is_cache_valid(cache_path: Path, expiry_hours: int = IPO_CACHE_EXPIRY_HOURS) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
        expiry_delta = timedelta(hours=expiry_hours)
        
        return datetime.now() - cached_time < expiry_delta
    except (json.JSONDecodeError, KeyError):
        return False


def _save_to_cache(cache_name: str, data: dict):
    """Save data to cache."""
    cache_path = _get_cache_path(cache_name)
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)


def _load_from_cache(cache_name: str) -> Optional[dict]:
    """Load data from cache."""
    cache_path = _get_cache_path(cache_name)
    
    if not _is_cache_valid(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        return cache_data.get('data')
    except Exception as e:
        print(f"Cache read error for {cache_name}: {e}")
        return None


def _get_finnhub_ipo_list(days_ahead: int) -> List[IPOEntry]:
    """
    Fetch upcoming IPO list from Finnhub API (existing logic).
    Uses same cache key and behavior: cache hit returns parsed list;
    no API key or request error returns mock data.
    """
    cache_name = f"ipo_calendar_{days_ahead}d"
    cached_data = _load_from_cache(cache_name)
    if cached_data:
        print(f"[Cache Hit] Loaded IPO calendar from cache")
        return _parse_ipo_data(cached_data)
    if not FINNHUB_API_KEY:
        print("[Warning] FINNHUB_API_KEY not set, using mock data")
        return _get_mock_ipo_data()
    start_date = date.today()
    end_date = start_date + timedelta(days=days_ahead)
    try:
        url = f"{FINNHUB_BASE_URL}/calendar/ipo"
        params = {
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'token': FINNHUB_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        _save_to_cache(cache_name, data)
        return _parse_ipo_data(data)
    except requests.RequestException as e:
        print(f"Error fetching IPO calendar: {e}")
        return _get_mock_ipo_data()


def _ipo_entry_to_dict(entry: IPOEntry) -> dict:
    """Serialize IPOEntry to a JSON-serializable dict (for NASDAQ cache)."""
    return {
        'ticker': entry.ticker,
        'name': entry.name,
        'exchange': entry.exchange,
        'ipo_date': entry.ipo_date.isoformat(),
        'price_range_low': entry.price_range_low,
        'price_range_high': entry.price_range_high,
        'ipo_price': entry.ipo_price,
        'shares_offered': entry.shares_offered,
        'total_shares': entry.total_shares,
        'status': entry.status,
    }


def _ipo_entry_from_dict(d: dict) -> IPOEntry:
    """Deserialize dict (from cache or merge) back to IPOEntry."""
    ipo_date_str = d.get('ipo_date', '')
    ipo_date = datetime.strptime(ipo_date_str, '%Y-%m-%d').date() if ipo_date_str else date.today()
    return IPOEntry(
        ticker=d.get('ticker', 'N/A'),
        name=d.get('name', 'Unknown Company'),
        exchange=d.get('exchange', 'N/A'),
        ipo_date=ipo_date,
        price_range_low=d.get('price_range_low'),
        price_range_high=d.get('price_range_high'),
        ipo_price=d.get('ipo_price'),
        shares_offered=d.get('shares_offered'),
        total_shares=d.get('total_shares'),
        status=d.get('status', 'expected'),
    )


def _fetch_nasdaq_ipo_list(days_ahead: int) -> List[IPOEntry]:
    """
    Fetch upcoming IPO list from NASDAQ via finance_calendars.
    Uses separate cache key ipo_calendar_{days_ahead}d_nasdaq.
    On any exception returns [] so Finnhub-only result is still used.
    """
    cache_name = f"ipo_calendar_{days_ahead}d_nasdaq"
    cached = _load_from_cache(cache_name)
    if cached is not None and isinstance(cached.get('entries'), list):
        entries = cached.get('entries', [])
        try:
            out = [_ipo_entry_from_dict(e) for e in entries]
            out.sort(key=lambda x: x.ipo_date)
            return out
        except Exception as e:
            print(f"Error deserializing NASDAQ IPO cache: {e}")
            # Fall through to fetch

    try:
        from finance_calendars import finance_calendars as fc
        today = date.today()
        end_cap = today + timedelta(days=days_ahead)
        all_entries: List[IPOEntry] = []
        # Current month and next month (NASDAQ API is month-based)
        month_start_cur = datetime(today.year, today.month, 1)
        if today.month == 12:
            month_start_next = datetime(today.year + 1, 1, 1)
        else:
            month_start_next = datetime(today.year, today.month + 1, 1)
        for month_start in (month_start_cur, month_start_next):
            if month_start.date() > end_cap:
                continue
            try:
                raw = fc.get_upcoming_ipos_by_month(month_start)
            except Exception as e:
                print(f"NASDAQ get_upcoming_ipos_by_month({month_start}) error: {e}")
                raw = []
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                continue
            # DataFrame: iterate rows
            if hasattr(raw, 'itertuples') or hasattr(raw, 'iterrows'):
                items = list(raw.to_dict('records')) if hasattr(raw, 'to_dict') else []
            elif isinstance(raw, list):
                items = raw
            else:
                items = []
            for item in items:
                entry = _parse_nasdaq_ipo_item(item, end_cap)
                if entry and entry.ipo_date <= end_cap:
                    all_entries.append(entry)
        # Dedupe by (normalized key), prefer entry with more filled fields
        def _filled(e: IPOEntry) -> int:
            return sum(1 for x in (e.price_range_low, e.price_range_high, e.shares_offered, e.total_shares) if x is not None)
        by_key: Dict[Tuple[str, date], IPOEntry] = {}
        for e in all_entries:
            k = _normalize_ipo_key(e)
            if k not in by_key or _filled(e) > _filled(by_key[k]):
                by_key[k] = e
        all_entries = sorted(by_key.values(), key=lambda x: x.ipo_date)
        _save_to_cache(cache_name, {'entries': [_ipo_entry_to_dict(e) for e in all_entries]})
        return all_entries
    except Exception as e:
        print(f"Error fetching NASDAQ IPO calendar: {e}")
        return []


def _normalize_ipo_key(entry: IPOEntry) -> Tuple[str, date]:
    """Key for deduplication: (normalized company name, ipo_date) so same company from different sources merges."""
    name = (entry.name or '').strip().lower() if entry.name else ''
    return (name or 'unknown', entry.ipo_date)


def _parse_nasdaq_ipo_item(item, end_cap: date) -> Optional[IPOEntry]:
    """Map one item from finance_calendars (DataFrame row as dict or object) to IPOEntry. Returns None if no valid date."""
    try:
        if hasattr(item, 'get'):
            d = item
        elif hasattr(item, 'keys'):
            d = dict(item)
        else:
            d = {}
        # finance_calendars DataFrame columns: companyName, proposedExchange, proposedSharePrice, sharesOffered, expectedPriceDate, dollarValueOfSharesOffered
        name = d.get('companyName') or d.get('CompanyName') or d.get('name') or d.get('company') or 'Unknown Company'
        symbol = d.get('symbol') or d.get('Symbol') or d.get('ticker') or 'N/A'
        exchange = d.get('proposedExchange') or d.get('exchange') or d.get('Exchange') or 'NASDAQ'
        raw_date = d.get('expectedPriceDate') or d.get('ExpectedPriceDate') or d.get('date') or d.get('ipoDate') or d.get('offerDate')
        if raw_date is None or (isinstance(raw_date, float) and pd.isna(raw_date)):
            return None
        if isinstance(raw_date, (date, datetime)):
            ipo_date = raw_date.date() if isinstance(raw_date, datetime) else raw_date
        else:
            s = str(raw_date).strip()
            if not s:
                return None
            # Support MM/DD/YYYY (NASDAQ) and YYYY-MM-DD
            if '/' in s:
                ipo_date = datetime.strptime(s[:10] if len(s) >= 10 else s, '%m/%d/%Y').date()
            else:
                ipo_date = datetime.strptime(s[:10], '%Y-%m-%d').date()
        if ipo_date > end_cap:
            return None
        low = d.get('priceRangeLow') or d.get('priceMin') or d.get('share_price_lowest') or d.get('proposedSharePrice')
        high = d.get('priceRangeHigh') or d.get('priceMax') or d.get('share_price_highest') or d.get('proposedSharePrice')
        shares_raw = d.get('numberOfShares') or d.get('sharesOffered') or d.get('share_count')
        if shares_raw is not None and not (isinstance(shares_raw, float) and pd.isna(shares_raw)):
            try:
                shares_offered = int(str(shares_raw).replace(',', ''))
            except (ValueError, TypeError):
                shares_offered = None
        else:
            shares_offered = None
        status = (d.get('status') or 'expected').lower()
        if status not in ('expected', 'priced', 'withdrawn'):
            status = 'expected'
        return IPOEntry(
            ticker=symbol,
            name=name,
            exchange=exchange,
            ipo_date=ipo_date,
            price_range_low=float(low) if low is not None and not (isinstance(low, float) and pd.isna(low)) else None,
            price_range_high=float(high) if high is not None and not (isinstance(high, float) and pd.isna(high)) else None,
            shares_offered=shares_offered,
            total_shares=None,
            status=status,
        )
    except Exception as e:
        print(f"Error parsing NASDAQ IPO item: {e}")
        return None


def _merge_ipo_entries(finnhub_list: List[IPOEntry], nasdaq_list: List[IPOEntry]) -> List[IPOEntry]:
    """
    Merge two IPO lists and deduplicate by (normalized ticker or name, ipo_date).
    When duplicates exist, prefer the entry with more non-null fields. Return sorted by ipo_date.
    """
    def filled_count(e: IPOEntry) -> int:
        return sum(1 for x in (e.price_range_low, e.price_range_high, e.shares_offered, e.total_shares) if x is not None)

    by_key: Dict[Tuple[str, date], IPOEntry] = {}
    for e in finnhub_list + nasdaq_list:
        k = _normalize_ipo_key(e)
        if k not in by_key or filled_count(e) > filled_count(by_key[k]):
            by_key[k] = e
    merged = list(by_key.values())
    merged.sort(key=lambda x: x.ipo_date)
    return merged


def fetch_ipo_calendar(days_ahead: int = 30) -> List[IPOEntry]:
    """
    Fetch upcoming IPO calendar from Finnhub and NASDAQ, then merge.
    Uses caching per source; existing Finnhub behavior unchanged.
    """
    finnhub_list = _get_finnhub_ipo_list(days_ahead)
    try:
        nasdaq_list = _fetch_nasdaq_ipo_list(days_ahead)
    except Exception as e:
        print(f"NASDAQ IPO fetch failed: {e}")
        nasdaq_list = []
    merged = _merge_ipo_entries(finnhub_list, nasdaq_list)
    print(f"[IPO calendar] {len(merged)} total ({len(finnhub_list)} Finnhub, {len(nasdaq_list)} NASDAQ)")
    return merged


def _parse_ipo_data(data: dict) -> List[IPOEntry]:
    """Parse Finnhub IPO data into IPOEntry objects."""
    ipos = []
    
    ipo_calendar = data.get('ipoCalendar', [])
    
    for ipo in ipo_calendar:
        try:
            # Parse date
            ipo_date_str = ipo.get('date', '')
            if ipo_date_str:
                ipo_date = datetime.strptime(ipo_date_str, '%Y-%m-%d').date()
            else:
                continue
            
            entry = IPOEntry(
                ticker=ipo.get('symbol', 'N/A'),
                name=ipo.get('name', 'Unknown Company'),
                exchange=ipo.get('exchange', 'N/A'),
                ipo_date=ipo_date,
                price_range_low=ipo.get('priceRangeLow'),
                price_range_high=ipo.get('priceRangeHigh'),
                shares_offered=ipo.get('numberOfShares'),
                total_shares=ipo.get('totalSharesValue'),
                status=ipo.get('status', 'expected')
            )
            ipos.append(entry)
            
        except Exception as e:
            print(f"Error parsing IPO entry: {e}")
            continue
    
    # Sort by date
    ipos.sort(key=lambda x: x.ipo_date)
    
    return ipos


def _get_mock_ipo_data() -> List[IPOEntry]:
    """Return mock IPO data for testing when API key is not available."""
    today = date.today()
    
    return [
        IPOEntry(
            ticker="DEMO1",
            name="Demo Tech Corp",
            exchange="NASDAQ",
            ipo_date=today + timedelta(days=5),
            price_range_low=18.0,
            price_range_high=22.0,
            shares_offered=10000000,
            status="expected"
        ),
        IPOEntry(
            ticker="DEMO2",
            name="Sample AI Inc",
            exchange="NYSE",
            ipo_date=today + timedelta(days=12),
            price_range_low=25.0,
            price_range_high=30.0,
            shares_offered=8000000,
            status="expected"
        ),
        IPOEntry(
            ticker="DEMO3",
            name="Example Health Ltd",
            exchange="NASDAQ",
            ipo_date=today + timedelta(days=20),
            price_range_low=12.0,
            price_range_high=15.0,
            shares_offered=15000000,
            status="expected"
        ),
    ]


def get_historical_price(ticker: str, target_date: date) -> Optional[float]:
    """
    Get the closing price for a stock on or near a specific date.
    
    Args:
        ticker: Stock ticker symbol
        target_date: The date to get the price for
        
    Returns:
        Closing price as float, or None if unavailable
    """
    try:
        from openbb_adapter import fetch_historical_price_openbb
        price = fetch_historical_price_openbb(ticker, target_date)
        if price is not None and price > 0:
            return price
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenBB] get_historical_price fallback for {ticker}: {e}")
    try:
        stock = yf.Ticker(ticker)
        start = target_date - timedelta(days=7)
        end = target_date + timedelta(days=7)
        hist = stock.history(start=start, end=end)

        if not hist.empty:
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            target_dt = datetime.combine(target_date, datetime.min.time())
            valid_dates = hist[hist.index <= target_dt]
            if valid_dates.empty:
                return float(hist['Close'].iloc[0])
            return float(valid_dates['Close'].iloc[-1])
    except Exception as e:
        print(f"Yahoo error fetching historical price for {ticker}: {e}")

    # Backup APIs
    try:
        from api_clients import (
            fetch_historical_close_polygon,
            fetch_historical_close_twelvedata,
            fetch_historical_close_eodhd,
        )
        price = fetch_historical_close_polygon(ticker, target_date)
        if price is not None:
            return price
        price = fetch_historical_close_twelvedata(ticker, target_date)
        if price is not None:
            return price
        price = fetch_historical_close_eodhd(ticker, target_date)
        if price is not None:
            return price
    except ImportError:
        pass

    return None


def get_current_price(ticker: str) -> Optional[float]:
    """
    Get current/latest price for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Current price as float, or None if unavailable
    """
    try:
        from openbb_adapter import fetch_quote_openbb
        price = fetch_quote_openbb(ticker)
        if price is not None and price > 0:
            return float(price)
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenBB] get_current_price fallback for {ticker}: {e}")
    try:
        stock = yf.Ticker(ticker)
        fast_info = stock.fast_info
        price = None
        if hasattr(fast_info, 'last_price') and fast_info.last_price:
            price = fast_info.last_price
        elif hasattr(fast_info, 'previous_close') and fast_info.previous_close:
            price = fast_info.previous_close
        if price and price > 0:
            return float(round(price, 2))
        hist = stock.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"Yahoo error fetching current price for {ticker}: {e}")

    # Backup APIs (return Decimal; convert to float)
    try:
        from api_clients import fetch_price_polygon, fetch_price_twelvedata, fetch_price_eodhd
        for fetcher in (fetch_price_polygon, fetch_price_twelvedata, fetch_price_eodhd):
            result = fetcher(ticker)
            if result is not None and float(result) > 0:
                return float(result)
    except ImportError:
        pass

    return None


def get_vintage_performance(ticker: str, ipo_date: date, ipo_price: Optional[float] = None) -> Optional[VintagePerformance]:
    """
    Calculate stock performance at 1, 2, and 3-year vintages from IPO.
    
    Args:
        ticker: Stock ticker symbol
        ipo_date: The IPO listing date
        ipo_price: IPO price (if known). If None, will try to fetch from history.
        
    Returns:
        VintagePerformance object with return calculations, or None if data unavailable
    """
    today = date.today()
    
    # Get IPO price if not provided
    if ipo_price is None:
        ipo_price = get_historical_price(ticker, ipo_date)
        if ipo_price is None:
            print(f"Could not determine IPO price for {ticker}")
            return None
    
    # Get current price
    current_price = get_current_price(ticker)
    if current_price is None:
        print(f"Could not get current price for {ticker}")
        return None
    
    # Calculate total return
    total_return = ((current_price - ipo_price) / ipo_price) * 100
    
    # Initialize vintage performance
    vintage = VintagePerformance(
        ticker=ticker,
        ipo_date=ipo_date,
        ipo_price=ipo_price,
        current_price=current_price,
        total_return=round(total_return, 2)
    )
    
    # Calculate 1-year performance
    year_1_date = ipo_date + timedelta(days=365)
    if today >= year_1_date:
        year_1_price = get_historical_price(ticker, year_1_date)
        if year_1_price:
            vintage.year_1_return = round(((year_1_price - ipo_price) / ipo_price) * 100, 2)
            vintage.year_1_status = "Calculated"
    else:
        vintage.year_1_status = "Pending"
    
    # Calculate 2-year performance
    year_2_date = ipo_date + timedelta(days=730)
    if today >= year_2_date:
        year_2_price = get_historical_price(ticker, year_2_date)
        if year_2_price:
            vintage.year_2_return = round(((year_2_price - ipo_price) / ipo_price) * 100, 2)
            vintage.year_2_status = "Calculated"
    else:
        vintage.year_2_status = "Pending"
    
    # Calculate 3-year performance
    year_3_date = ipo_date + timedelta(days=1095)
    if today >= year_3_date:
        year_3_price = get_historical_price(ticker, year_3_date)
        if year_3_price:
            vintage.year_3_return = round(((year_3_price - ipo_price) / ipo_price) * 100, 2)
            vintage.year_3_status = "Calculated"
    else:
        vintage.year_3_status = "Pending"
    
    return vintage


def get_ipo_price_history(ticker: str, ipo_date: date, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Get price history from IPO date for charting.
    
    Aligns data to "Day 0" being the IPO date for comparison charts.
    
    Args:
        ticker: Stock ticker symbol
        ipo_date: The IPO listing date
        days: Number of trading days to fetch
        
    Returns:
        DataFrame with columns ['Day', 'Close', 'Normalized'] or None if unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch data from IPO date
        end_date = ipo_date + timedelta(days=int(days * 1.5))  # Buffer for non-trading days
        
        hist = stock.history(start=ipo_date, end=end_date)
        
        if hist.empty or len(hist) < 5:
            return None
        
        # Take first 'days' trading days
        hist = hist.head(days)
        
        # Create Day 0 aligned index
        df = pd.DataFrame({
            'Day': range(len(hist)),
            'Close': hist['Close'].values,
            'Date': hist.index.values
        })
        
        # Normalize to Day 0 price (100 = IPO price)
        ipo_price = df['Close'].iloc[0]
        df['Normalized'] = (df['Close'] / ipo_price) * 100
        
        df['Ticker'] = ticker
        
        return df
        
    except Exception as e:
        print(f"Error fetching IPO price history for {ticker}: {e}")
        return None


def check_vintage_anniversaries(
    ipo_registries: List[dict],
    days_threshold: int = 10
) -> List[dict]:
    """
    Check for IPOs approaching 1, 2, or 3-year anniversaries.
    
    Args:
        ipo_registries: List of IPO registry entries (dicts with ticker, ipo_date)
        days_threshold: Alert if within ±this many days of anniversary
        
    Returns:
        List of dicts with anniversary alerts
    """
    alerts = []
    today = date.today()
    
    for ipo in ipo_registries:
        ticker = ipo.get('ticker')
        ipo_date = ipo.get('ipo_date')
        
        if not ticker or not ipo_date:
            continue
        
        # Convert to date if string
        if isinstance(ipo_date, str):
            ipo_date = datetime.strptime(ipo_date, '%Y-%m-%d').date()
        
        # Check each anniversary (1, 2, 3 years)
        for years in [1, 2, 3]:
            anniversary = ipo_date + timedelta(days=365 * years)
            days_diff = (anniversary - today).days
            
            # Check if within threshold (±days_threshold)
            if abs(days_diff) <= days_threshold:
                status = "upcoming" if days_diff > 0 else ("today" if days_diff == 0 else "recent")
                
                alerts.append({
                    'ticker': ticker,
                    'ipo_date': ipo_date,
                    'anniversary_years': years,
                    'anniversary_date': anniversary,
                    'days_diff': days_diff,
                    'status': status,
                    'company_name': ipo.get('company_name', 'Unknown')
                })
    
    # Sort by absolute days difference
    alerts.sort(key=lambda x: abs(x['days_diff']))
    
    return alerts


def clear_ipo_cache():
    """Clear all IPO-related cache files."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
        print("IPO cache cleared")

