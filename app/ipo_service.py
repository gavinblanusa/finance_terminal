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


def fetch_ipo_calendar(days_ahead: int = 30) -> List[IPOEntry]:
    """
    Fetch upcoming IPO calendar from Finnhub API.
    
    Uses caching to avoid hitting API rate limits.
    
    Args:
        days_ahead: Number of days to look ahead for IPOs
        
    Returns:
        List of IPOEntry objects for upcoming IPOs
    """
    cache_name = f"ipo_calendar_{days_ahead}d"
    
    # Try loading from cache
    cached_data = _load_from_cache(cache_name)
    if cached_data:
        print(f"[Cache Hit] Loaded IPO calendar from cache")
        return _parse_ipo_data(cached_data)
    
    if not FINNHUB_API_KEY:
        print("[Warning] FINNHUB_API_KEY not set, using mock data")
        return _get_mock_ipo_data()
    
    # Calculate date range
    start_date = date.today()
    end_date = start_date + timedelta(days=days_ahead)
    
    try:
        # Fetch from Finnhub API
        url = f"{FINNHUB_BASE_URL}/calendar/ipo"
        params = {
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'token': FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Cache the response
        _save_to_cache(cache_name, data)
        
        return _parse_ipo_data(data)
        
    except requests.RequestException as e:
        print(f"Error fetching IPO calendar: {e}")
        return _get_mock_ipo_data()


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

