"""
Market Data Service - Technical Intelligence Factory.

This module provides:
- Real-time market data fetching via yfinance
- Caching mechanism to avoid rate limits
- Technical indicator calculations (SMA, RSI, Bollinger Bands)
- Buy/Sell signal generation
- Historical valuation metrics via Financial Modeling Prep API
"""

import os
import json
import hashlib
import requests
import time
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Tuple, List, Literal
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Load environment variables from .env file
load_dotenv()

# Database imports (lazy loaded to avoid circular imports)
_db_module = None
_models_module = None

def _get_db_session():
    """Lazy load database session to avoid circular imports."""
    global _db_module
    if _db_module is None:
        from db import get_db_session
        _db_module = get_db_session
    return _db_module()

def _get_valuation_model():
    """Lazy load ValuationHistory model."""
    global _models_module
    if _models_module is None:
        from models import ValuationHistory
        _models_module = ValuationHistory
    return _models_module

# API Configuration
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_URL = "https://financialmodelingprep.com/stable"

# Alpha Vantage - check both possible env var names for compatibility
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '') or os.environ.get('ALPHA_API_KEY', '')
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Finnhub - 60 API calls/minute on free tier
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Alpha Vantage rate limit tracking (5 calls/minute on free tier)
# We need to space out API calls to avoid hitting the limit
_last_av_api_call = 0.0  # timestamp of last Alpha Vantage API call
AV_MIN_CALL_INTERVAL = 1.5  # minimum seconds between Alpha Vantage API calls

def _wait_for_av_rate_limit():
    """Wait if needed to respect Alpha Vantage rate limits."""
    global _last_av_api_call
    if _last_av_api_call > 0:
        elapsed = time.time() - _last_av_api_call
        if elapsed < AV_MIN_CALL_INTERVAL:
            wait_time = AV_MIN_CALL_INTERVAL - elapsed
            print(f"[Alpha Vantage] Waiting {wait_time:.1f}s to respect rate limit...")
            time.sleep(wait_time)
    _last_av_api_call = time.time()

# Debug: Print which API keys are configured (only first run)
_api_keys_printed = False
def _print_api_status():
    global _api_keys_printed
    if not _api_keys_printed:
        try:
            from api_clients import MASSIVE_API_KEY, TWELVEDATA_API_KEY, EODHD_API_KEY
            print(f"[API Config] Alpha Vantage: {'✓ Configured' if ALPHA_VANTAGE_API_KEY else '✗ Not set (add ALPHA_API_KEY to .env)'}")
            print(f"[API Config] FMP: {'✓ Configured' if FMP_API_KEY else '✗ Not set (add FMP_API_KEY to .env)'}")
            print(f"[API Config] Finnhub: {'✓ Configured' if FINNHUB_API_KEY else '✗ Not set (add FINNHUB_API_KEY to .env)'}")
            print(f"[API Config] Polygon/Massive: {'✓ Configured' if MASSIVE_API_KEY else '✗ Not set (add MASSIVE_API_KEY to .env)'}")
            print(f"[API Config] Twelve Data: {'✓ Configured' if TWELVEDATA_API_KEY else '✗ Not set (add TWELVEDATA_API_KEY to .env)'}")
            print(f"[API Config] EODHD: {'✓ Configured' if EODHD_API_KEY else '✗ Not set (add EODHD_API_KEY to .env)'}")
        except ImportError:
            print(f"[API Config] Alpha Vantage: {'✓ Configured' if ALPHA_VANTAGE_API_KEY else '✗ Not set (add ALPHA_API_KEY to .env)'}")
            print(f"[API Config] FMP: {'✓ Configured' if FMP_API_KEY else '✗ Not set (add FMP_API_KEY to .env)'}")
            print(f"[API Config] Finnhub: {'✓ Configured' if FINNHUB_API_KEY else '✗ Not set (add FINNHUB_API_KEY to .env)'}")
        _api_keys_printed = True


# Project root (caches live at Invest/ root)
_ROOT = Path(__file__).resolve().parent.parent
# Cache configuration
CACHE_DIR = _ROOT / ".market_cache"
CACHE_EXPIRY_HOURS = 4  # Cache OHLCV data for 4 hours during market hours
TICKER_INFO_CACHE_HOURS = 2  # Cache ticker info (P/E, earnings, etc.) for 2 hours


def _get_cache_path(ticker: str) -> Path:
    """Get cache file path for a ticker."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_ohlcv.json"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
        expiry_delta = timedelta(hours=CACHE_EXPIRY_HOURS)
        
        return datetime.now() - cached_time < expiry_delta
    except (json.JSONDecodeError, KeyError):
        return False


def _save_to_cache(ticker: str, df: pd.DataFrame):
    """Save DataFrame to cache."""
    cache_path = _get_cache_path(ticker)
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker.upper(),
        'data': df.reset_index().to_json(date_format='iso', orient='records')
    }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)


def _load_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from cache."""
    cache_path = _get_cache_path(ticker)
    
    if not _is_cache_valid(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        from io import StringIO
        df = pd.read_json(StringIO(cache_data['data']), orient='records')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Cache read error for {ticker}: {e}")
        return None


# =============================================================================
# Ticker Info Cache (P/E, PEG, earnings date, dividend yield, market cap)
# =============================================================================

def _get_ticker_info_cache_path(ticker: str) -> Path:
    """Get cache file path for ticker info data."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_info.json"


def _is_ticker_info_cache_valid(cache_path: Path) -> bool:
    """Check if ticker info cache is valid (not expired)."""
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
        expiry_delta = timedelta(hours=TICKER_INFO_CACHE_HOURS)
        
        return datetime.now() - cached_time < expiry_delta
    except (json.JSONDecodeError, KeyError):
        return False


def _save_ticker_info_to_cache(ticker: str, info_data: Dict):
    """Save ticker info data to cache."""
    cache_path = _get_ticker_info_cache_path(ticker)
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker.upper(),
        'data': info_data
    }
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"[Cache] Saved ticker info for {ticker}")
    except Exception as e:
        print(f"[Cache] Error saving ticker info for {ticker}: {e}")


def _load_ticker_info_from_cache(ticker: str) -> Optional[Dict]:
    """Load ticker info from cache if valid."""
    cache_path = _get_ticker_info_cache_path(ticker)
    
    if not _is_ticker_info_cache_valid(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        print(f"[Cache Hit] Loaded ticker info for {ticker} from cache")
        return cache_data.get('data')
    except Exception as e:
        print(f"[Cache] Error loading ticker info for {ticker}: {e}")
        return None


def _fetch_and_cache_ticker_info(ticker: str) -> Dict:
    """
    Fetch ticker info from yfinance and cache it.
    
    Returns dict with: pe_ratio, peg_ratio, dividend_yield, market_cap, 
                       earnings_date, days_to_earnings
    """
    # Check cache first
    cached_info = _load_ticker_info_from_cache(ticker)
    if cached_info is not None:
        # Recalculate days_to_earnings since it changes daily
        if cached_info.get('earnings_date'):
            try:
                earnings_dt = datetime.strptime(cached_info['earnings_date'], '%Y-%m-%d').date()
                cached_info['days_to_earnings'] = (earnings_dt - datetime.now().date()).days
            except:
                pass
        return cached_info
    
    # Fetch from yfinance
    info_data = {
        'pe_ratio': None,
        'forward_pe': None,
        'peg_ratio': None,
        'dividend_yield': None,
        'market_cap': None,
        'market_cap_category': None,
        'earnings_date': None,
        'days_to_earnings': None,
        'short_percent': None,
        'analyst_target': None,
        'current_price': None,
        'longName': None,
        'shortName': None,
    }
    
    yahoo_succeeded = False
    try:
        print(f"[Fetching] Downloading ticker info for {ticker} from yfinance...")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        info_data['longName'] = info.get('longName') or info.get('shortName')
        info_data['shortName'] = info.get('shortName') or info.get('longName')
        info_data['pe_ratio'] = info.get('trailingPE')
        
        # PEG ratio - try multiple sources
        peg_ratio = info.get('pegRatio')
        if peg_ratio is None:
            # Try to calculate PEG from forward P/E and growth
            forward_pe = info.get('forwardPE')
            earnings_growth = info.get('earningsGrowth')  # As decimal (0.25 = 25%)
            if forward_pe and earnings_growth and earnings_growth > 0:
                growth_pct = earnings_growth * 100
                peg_ratio = forward_pe / growth_pct
                print(f"[{ticker}] Calculated PEG: {forward_pe:.1f} / {growth_pct:.1f}% = {peg_ratio:.2f}")
        info_data['peg_ratio'] = peg_ratio
        
        # Handle dividend yield - multiple fallback methods
        yield_pct = None
        
        # Method 1: Try dividendYield field (returns as decimal, e.g., 0.0045 = 0.45%)
        dividend_yield = info.get('dividendYield')
        if dividend_yield is not None and dividend_yield > 0:
            yield_pct = dividend_yield * 100
            # Sanity check: yields above 20% are extremely rare
            if yield_pct > 20:
                print(f"[{ticker}] WARNING: dividendYield {yield_pct:.2f}% too high, trying fallback")
                yield_pct = None
        
        # Method 2: Calculate from dividendRate / price (more reliable)
        if yield_pct is None:
            dividend_rate = info.get('dividendRate')  # Annual dividend per share
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if dividend_rate and current_price and current_price > 0:
                yield_pct = (dividend_rate / current_price) * 100
                if yield_pct > 20:  # Sanity check
                    print(f"[{ticker}] WARNING: Calculated yield {yield_pct:.2f}% too high, skipping")
                    yield_pct = None
                else:
                    print(f"[{ticker}] Calculated dividend yield from rate: ${dividend_rate:.2f} / ${current_price:.2f} = {yield_pct:.2f}%")
        
        # Method 3: Try trailingAnnualDividendYield
        if yield_pct is None:
            trailing_yield = info.get('trailingAnnualDividendYield')
            if trailing_yield and trailing_yield > 0:
                yield_pct = trailing_yield * 100
                if yield_pct <= 20:
                    print(f"[{ticker}] Using trailingAnnualDividendYield: {yield_pct:.2f}%")
                else:
                    yield_pct = None
        
        info_data['dividend_yield'] = yield_pct
        
        # Market cap and category
        market_cap = info.get('marketCap')
        info_data['market_cap'] = market_cap
        if market_cap:
            if market_cap >= 200_000_000_000:  # $200B+
                info_data['market_cap_category'] = 'Mega'
            elif market_cap >= 10_000_000_000:  # $10B+
                info_data['market_cap_category'] = 'Large'
            elif market_cap >= 2_000_000_000:  # $2B+
                info_data['market_cap_category'] = 'Mid'
            elif market_cap >= 300_000_000:  # $300M+
                info_data['market_cap_category'] = 'Small'
            else:
                info_data['market_cap_category'] = 'Micro'
        
        # Forward P/E
        info_data['forward_pe'] = info.get('forwardPE')
        
        # Short interest (as percentage of float)
        short_percent = info.get('shortPercentOfFloat')
        if short_percent:
            info_data['short_percent'] = short_percent * 100  # Convert to percentage
        else:
            # Try alternative field
            shares_short = info.get('sharesShort')
            float_shares = info.get('floatShares')
            if shares_short and float_shares and float_shares > 0:
                info_data['short_percent'] = (shares_short / float_shares) * 100
        
        # Analyst target price (consensus)
        info_data['analyst_target'] = info.get('targetMeanPrice')

        # Current/last price for consistency
        info_data['current_price'] = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Get next earnings date
        try:
            calendar = stock.calendar
            if calendar is not None:
                earnings_date = None
                # calendar can be a dict or DataFrame depending on yfinance version
                if isinstance(calendar, dict):
                    earnings_dates = calendar.get('Earnings Date', [])
                    if earnings_dates:
                        if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                            earnings_date = earnings_dates[0]
                        else:
                            earnings_date = earnings_dates
                elif hasattr(calendar, 'loc'):
                    # DataFrame format
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date'].iloc[0] if hasattr(calendar.loc['Earnings Date'], 'iloc') else calendar.loc['Earnings Date']
                
                # Convert earnings date to string for caching
                if earnings_date is not None:
                    if hasattr(earnings_date, 'strftime'):
                        info_data['earnings_date'] = earnings_date.strftime('%Y-%m-%d')
                    elif hasattr(earnings_date, 'date'):
                        info_data['earnings_date'] = earnings_date.date().strftime('%Y-%m-%d')
                    elif isinstance(earnings_date, str):
                        info_data['earnings_date'] = earnings_date[:10]  # Take YYYY-MM-DD part
                    
                    # Calculate days to earnings
                    if info_data['earnings_date']:
                        try:
                            earnings_dt = datetime.strptime(info_data['earnings_date'], '%Y-%m-%d').date()
                            info_data['days_to_earnings'] = (earnings_dt - datetime.now().date()).days
                        except:
                            pass
        except Exception as e:
            print(f"[{ticker}] Error getting calendar: {e}")
        
        yahoo_succeeded = True
        # Save to cache
        _save_ticker_info_to_cache(ticker, info_data)
        
    except Exception as e:
        print(f"[{ticker}] Error fetching ticker info from Yahoo: {e}")

    # Minimal fallback when Yahoo fails: get at least a price from backup APIs
    if not yahoo_succeeded:
        try:
            from api_clients import (
                fetch_quote_twelvedata,
                fetch_price_polygon,
                fetch_price_twelvedata,
                fetch_price_eodhd,
            )
            backup_price = None
            quote = fetch_quote_twelvedata(ticker)
            if quote and quote.get('close') and quote['close'] > 0:
                backup_price = quote['close']
                info_data['current_price'] = backup_price
            if backup_price is None:
                for fetcher in (fetch_price_polygon, fetch_price_twelvedata, fetch_price_eodhd):
                    result = fetcher(ticker)
                    if result is not None and float(result) > 0:
                        info_data['current_price'] = float(result)
                        break
            if info_data.get('current_price') is not None:
                print(f"[Backup] Got ticker price for {ticker} from backup API")
                _save_ticker_info_to_cache(ticker, info_data)
        except ImportError:
            pass
    
    return info_data


def fetch_ohlcv(ticker: str, period_years: int | str = 2) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a ticker.
    
    Uses caching to avoid rate limits. Supports 2 years (default) up to 50 years
    or 'max' (full history via yfinance).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period_years: Number of years (e.g. 2, 50) or 'max' for full history
        
    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    ticker = ticker.upper().strip()
    use_max = period_years in (None, "max")

    # Try loading from cache first (skip if we need long history and cache is short or doesn't cover range)
    cached_df = _load_from_cache(ticker)
    if cached_df is not None:
        # If caller asked for long history but cache has < ~5 years, refetch to upgrade cache
        if use_max or (isinstance(period_years, int) and period_years > 5):
            if len(cached_df) < 1260:  # ~5 years of trading days
                cached_df = None  # force refetch
            else:
                # Ensure cache covers the requested range start (e.g. IPO stocks: cache might start in 2024 but we want 5Y)
                years = 50 if use_max else int(period_years)
                range_start = (datetime.now() - timedelta(days=years * 365)).date()
                try:
                    cache_start = pd.Timestamp(cached_df.index.min()).date()
                except Exception:
                    cache_start = range_start
                if cache_start > range_start:
                    cached_df = None  # cache doesn't go back far enough, refetch
                else:
                    print(f"[Cache Hit] Loaded {ticker} from cache")
                    return cached_df
        else:
            print(f"[Cache Hit] Loaded {ticker} from cache")
            return cached_df

    # Date range for non-max: backup APIs and yfinance start/end
    if use_max:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=50 * 365)
        start_date = start_dt.date()
        end_date = end_dt.date()
        days_needed = 50 * 365
    else:
        years = int(period_years)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=years * 365)
        start_date = start_dt.date()
        end_date = end_dt.date()
        days_needed = years * 365

    # Try OpenBB first (when available)
    df = None
    try:
        from openbb_adapter import fetch_ohlcv_openbb
        df = fetch_ohlcv_openbb(ticker, start_date, end_date)
        if df is not None and not df.empty and len(df) >= 50:
            # If we wanted long history, ensure OpenBB data goes back far enough (e.g. IPO stocks)
            want_long = use_max or (isinstance(period_years, int) and period_years >= 5)
            if want_long:
                try:
                    cache_start = pd.Timestamp(df.index.min()).date()
                    if cache_start > start_date:
                        df = None  # OpenBB returned truncated history; try yfinance period="max"
                except Exception:
                    pass
            if df is not None and not df.empty and len(df) >= 200:
                print(f"[OpenBB] Retrieved {len(df)} rows for {ticker}")
                _save_to_cache(ticker, df)
                return df
            if df is not None and len(df) < 200:
                print(f"Insufficient data for {ticker} ({len(df)} days, need 200+)")
            df = None
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenBB] fallback for {ticker}: {e}")
        df = None

    # Fetch from Yahoo Finance (use period="max" for long history so IPO-era stocks get full history)
    print(f"[Fetching] Downloading {ticker} data from Yahoo Finance...")
    try:
        stock = yf.Ticker(ticker)
        want_long_history = use_max or (isinstance(period_years, int) and period_years >= 5)
        if use_max or want_long_history:
            df = stock.history(period="max", interval="1d")
        else:
            df = stock.history(start=start_dt, end=end_dt, interval="1d")

        if not df.empty:
            # Clean up column names and data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            # Remove timezone from index (yfinance returns tz-aware timestamps)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            else:
                df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            if len(df) < 200:
                print(f"Insufficient data for {ticker} ({len(df)} days, need 200+)")
    except Exception as e:
        print(f"Yahoo Finance error for {ticker}: {e}")
        df = None

    # Backup APIs if Yahoo failed or returned insufficient data (< 50 rows)
    if df is None or df.empty or len(df) < 50:
        try:
            from api_clients import (
                fetch_ohlcv_polygon,
                fetch_ohlcv_twelvedata,
                fetch_ohlcv_eodhd,
            )
            backup_df = None
            if df is None or df.empty:
                print(f"[Backup] Trying Polygon for {ticker}...")
                backup_df = fetch_ohlcv_polygon(ticker, start_date, end_date)
            if backup_df is None or backup_df.empty:
                print(f"[Backup] Trying Twelve Data for {ticker}...")
                backup_df = fetch_ohlcv_twelvedata(ticker, days=days_needed)
            if backup_df is None or backup_df.empty:
                print(f"[Backup] Trying EODHD for {ticker}...")
                backup_df = fetch_ohlcv_eodhd(ticker, start_date, end_date)
            if backup_df is not None and not backup_df.empty:
                df = backup_df
                print(f"[Backup] Retrieved {len(df)} rows for {ticker}")
        except ImportError:
            pass

    if df is None or df.empty:
        print(f"No data returned for {ticker}")
        return None

    if len(df) < 200:
        print(f"Insufficient data for {ticker} ({len(df)} days, need 200+)")

    # Save to cache
    _save_to_cache(ticker, df)
    return df


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(
    series: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle_band = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return middle_band, upper_band, lower_band


# =============================================================================
# TradingView-Style Indicators
# =============================================================================

def calculate_stochastic_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic RSI oscillator.
    
    Stochastic RSI applies the Stochastic formula to RSI values instead of price.
    This creates a more sensitive oscillator that can identify overbought/oversold
    conditions earlier than regular RSI.
    
    Returns:
        Tuple of (stoch_rsi_k, stoch_rsi_d) - both scaled to -50 to +50 range
    """
    # First calculate RSI
    rsi = calculate_rsi(series, rsi_period)
    
    # Apply Stochastic formula to RSI
    rsi_min = rsi.rolling(window=stoch_period, min_periods=1).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=1).max()
    
    # Stochastic RSI (0-100 range initially)
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    stoch_rsi = stoch_rsi.fillna(50)  # Fill NaN with neutral value
    
    # Smooth %K
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k, min_periods=1).mean()
    
    # %D is SMA of %K
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, min_periods=1).mean()
    
    # Convert to -50 to +50 range (centered at 0)
    stoch_rsi_k = stoch_rsi_k - 50
    stoch_rsi_d = stoch_rsi_d - 50
    
    return stoch_rsi_k, stoch_rsi_d


def calculate_rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Calculate Rate of Change (ROC) indicator.
    
    ROC measures the percentage change between the current price and the price
    n periods ago. Positive values indicate upward momentum, negative indicate downward.
    
    Args:
        series: Price series (typically Close)
        period: Lookback period (default 12)
        
    Returns:
        ROC values as percentage (-100 to +100 typical range)
    """
    roc = ((series - series.shift(period)) / series.shift(period)) * 100
    return roc.fillna(0)


def calculate_momentum_oscillator(
    df: pd.DataFrame,
    stoch_weight: float = 0.6,
    roc_weight: float = 0.4,
    roc_period: int = 12,
    smooth_period: int = 3
) -> pd.Series:
    """
    Calculate a combined momentum oscillator (Stochastic RSI + ROC blend).
    
    This creates the top oscillator panel seen in the TradingView chart,
    combining the sensitivity of Stochastic RSI with the trend confirmation of ROC.
    
    Args:
        df: DataFrame with Close prices
        stoch_weight: Weight for Stochastic RSI component (default 0.6)
        roc_weight: Weight for ROC component (default 0.4)
        roc_period: ROC lookback period
        smooth_period: Final smoothing period
        
    Returns:
        Momentum oscillator scaled to approximately -50 to +50
    """
    close = df['Close']
    
    # Get Stochastic RSI (already in -50 to +50 range)
    stoch_k, _ = calculate_stochastic_rsi(close)
    
    # Get ROC and normalize to similar scale
    roc = calculate_rate_of_change(close, roc_period)
    # Clip ROC to reasonable range and scale
    roc_normalized = np.clip(roc, -50, 50)
    
    # Combine with weights
    combined = (stoch_k * stoch_weight) + (roc_normalized * roc_weight)
    
    # Smooth the result
    momentum = combined.rolling(window=smooth_period, min_periods=1).mean()
    
    return momentum


def calculate_secondary_oscillator(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate a secondary momentum oscillator (similar to MACD histogram style).
    
    This creates the bottom oscillator panel, showing momentum divergence
    between fast and slow EMAs.
    
    Args:
        df: DataFrame with Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        
    Returns:
        Tuple of (oscillator, signal_line) - scaled to approximately -40 to +40
    """
    close = df['Close']
    
    # Calculate EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram (MACD - Signal)
    histogram = macd - signal
    
    # Normalize to -40 to +40 range based on price scale
    # Use percentage of price for normalization
    price_scale = close.rolling(window=slow_period, min_periods=1).mean()
    oscillator = (histogram / price_scale) * 1000  # Scale factor
    
    # Clip to reasonable range
    oscillator = np.clip(oscillator, -40, 40)
    
    # Also normalize signal line
    signal_normalized = (signal - ema_slow) / price_scale * 500
    signal_normalized = np.clip(signal_normalized, -40, 40)
    
    return oscillator, signal_normalized


def calculate_supply_demand_zones(
    df: pd.DataFrame,
    period: int = 20,
    inner_std: float = 2.0,
    outer_std: float = 3.0
) -> Dict[str, pd.Series]:
    """
    Calculate supply/demand zones (extended Bollinger Bands).
    
    These create the pink shaded zones at price extremes, indicating
    areas where price is statistically likely to reverse.
    
    Args:
        df: DataFrame with Close prices
        period: Lookback period for bands
        inner_std: Inner band standard deviation (typical BB)
        outer_std: Outer band standard deviation (extreme zones)
        
    Returns:
        Dict with zone boundaries
    """
    close = df['Close']
    
    # Calculate middle band (SMA)
    middle = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    
    # Inner bands (regular Bollinger Bands)
    inner_upper = middle + (std * inner_std)
    inner_lower = middle - (std * inner_std)
    
    # Outer bands (extreme zones)
    outer_upper = middle + (std * outer_std)
    outer_lower = middle - (std * outer_std)
    
    return {
        'middle': middle,
        'inner_upper': inner_upper,
        'inner_lower': inner_lower,
        'outer_upper': outer_upper,
        'outer_lower': outer_lower
    }


def calculate_tradingview_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate TradingView-style trading signals with all indicators.
    
    Adds the following columns to the DataFrame:
    - TV_Momentum: Primary momentum oscillator (-50 to +50)
    - TV_Oscillator: Secondary oscillator (-40 to +40)
    - TV_Signal_Line: Signal line for secondary oscillator
    - TV_Zone_*: Supply/demand zone boundaries
    - TV_Support: Key moving average support line
    - TV_Signal: Trading signal (BUY, SELL, or empty)
    - TV_Signal_Strength: Signal strength (STRONG, MODERATE, WEAK)
    
    Signal Logic (based on TradingView chart analysis):
    - BUY: Price near lower zone + both oscillators negative but turning up
    - SELL: Price near upper zone + both oscillators positive but turning down
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added TradingView indicator and signal columns
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    close = df['Close']
    
    # === OSCILLATORS ===
    # Primary momentum oscillator (top panel)
    df['TV_Momentum'] = calculate_momentum_oscillator(df)
    
    # Secondary oscillator (bottom panel)
    df['TV_Oscillator'], df['TV_Signal_Line'] = calculate_secondary_oscillator(df)
    
    # === SUPPLY/DEMAND ZONES ===
    zones = calculate_supply_demand_zones(df)
    df['TV_Zone_Middle'] = zones['middle']
    df['TV_Zone_Inner_Upper'] = zones['inner_upper']
    df['TV_Zone_Inner_Lower'] = zones['inner_lower']
    df['TV_Zone_Outer_Upper'] = zones['outer_upper']
    df['TV_Zone_Outer_Lower'] = zones['outer_lower']
    
    # === KEY SUPPORT LINE ===
    # Use 50-period SMA as the "gold" support line
    df['TV_Support'] = calculate_sma(close, 50)
    
    # === SIGNAL GENERATION ===
    df['TV_Signal'] = ''
    df['TV_Signal_Strength'] = ''
    
    # Calculate momentum changes (for detecting turns)
    momentum_change = df['TV_Momentum'].diff()
    oscillator_change = df['TV_Oscillator'].diff()
    
    # Previous values for crossover detection
    prev_momentum = df['TV_Momentum'].shift(1)
    prev_oscillator = df['TV_Oscillator'].shift(1)
    
    # Price position relative to zones
    price_near_lower = close <= df['TV_Zone_Inner_Lower']
    price_near_upper = close >= df['TV_Zone_Inner_Upper']
    price_at_extreme_lower = close <= df['TV_Zone_Outer_Lower']
    price_at_extreme_upper = close >= df['TV_Zone_Outer_Upper']
    
    # Oscillator conditions
    momentum_oversold = df['TV_Momentum'] < -20
    momentum_overbought = df['TV_Momentum'] > 20
    momentum_turning_up = (momentum_change > 0) & (prev_momentum < 0)
    momentum_turning_down = (momentum_change < 0) & (prev_momentum > 0)
    
    oscillator_oversold = df['TV_Oscillator'] < -15
    oscillator_overbought = df['TV_Oscillator'] > 15
    oscillator_turning_up = (oscillator_change > 0) & (prev_oscillator < 0)
    oscillator_turning_down = (oscillator_change < 0) & (prev_oscillator > 0)
    
    # === BUY SIGNALS ===
    # Strong buy: Price at extreme lower + both oscillators oversold and turning up
    strong_buy = (
        price_at_extreme_lower &
        momentum_oversold &
        oscillator_oversold &
        (momentum_turning_up | oscillator_turning_up)
    )
    
    # Moderate buy: Price near lower band + momentum turning up
    moderate_buy = (
        price_near_lower &
        momentum_oversold &
        momentum_turning_up &
        ~strong_buy
    )
    
    # Weak buy: Oscillators both oversold and at least one turning up
    weak_buy = (
        (momentum_oversold | oscillator_oversold) &
        (momentum_turning_up | oscillator_turning_up) &
        ~strong_buy &
        ~moderate_buy
    )
    
    # === SELL SIGNALS ===
    # Strong sell: Price at extreme upper + both oscillators overbought and turning down
    strong_sell = (
        price_at_extreme_upper &
        momentum_overbought &
        oscillator_overbought &
        (momentum_turning_down | oscillator_turning_down)
    )
    
    # Moderate sell: Price near upper band + momentum turning down
    moderate_sell = (
        price_near_upper &
        momentum_overbought &
        momentum_turning_down &
        ~strong_sell
    )
    
    # Weak sell: Oscillators both overbought and at least one turning down
    weak_sell = (
        (momentum_overbought | oscillator_overbought) &
        (momentum_turning_down | oscillator_turning_down) &
        ~strong_sell &
        ~moderate_sell
    )
    
    # Apply signals (priority order)
    df.loc[weak_buy, 'TV_Signal'] = 'BUY'
    df.loc[weak_buy, 'TV_Signal_Strength'] = 'WEAK'
    
    df.loc[moderate_buy, 'TV_Signal'] = 'BUY'
    df.loc[moderate_buy, 'TV_Signal_Strength'] = 'MODERATE'
    
    df.loc[strong_buy, 'TV_Signal'] = 'BUY'
    df.loc[strong_buy, 'TV_Signal_Strength'] = 'STRONG'
    
    df.loc[weak_sell, 'TV_Signal'] = 'SELL'
    df.loc[weak_sell, 'TV_Signal_Strength'] = 'WEAK'
    
    df.loc[moderate_sell, 'TV_Signal'] = 'SELL'
    df.loc[moderate_sell, 'TV_Signal_Strength'] = 'MODERATE'
    
    df.loc[strong_sell, 'TV_Signal'] = 'SELL'
    df.loc[strong_sell, 'TV_Signal_Strength'] = 'STRONG'
    
    return df


def _get_tv_signals_cache_path(ticker: str, timeframe: str = '1W') -> Path:
    """Get cache file path for TradingView signals."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_tv_signals_{timeframe}.json"


def save_tv_signals_to_cache(ticker: str, df: pd.DataFrame, timeframe: str = '1W') -> bool:
    """
    Save TradingView signals to cache for a ticker.
    
    Saves the calculated signals so they can be retrieved without recalculation.
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with TradingView signals calculated
        timeframe: Timeframe label (e.g., '1W', '1D')
        
    Returns:
        True if saved successfully, False otherwise
    """
    cache_path = _get_tv_signals_cache_path(ticker, timeframe)
    
    try:
        # Extract only the signal-related columns to save
        signal_cols = ['TV_Momentum', 'TV_Oscillator', 'TV_Signal', 'TV_Signal_Strength',
                       'TV_Zone_Middle', 'TV_Zone_Inner_Upper', 'TV_Zone_Inner_Lower',
                       'TV_Zone_Outer_Upper', 'TV_Zone_Outer_Lower', 'TV_Support']
        
        # Only include columns that exist
        available_cols = [col for col in signal_cols if col in df.columns]
        
        if not available_cols:
            print(f"[TV Cache] No TradingView columns to save for {ticker}")
            return False
        
        # Create a copy with just the signal columns + OHLCV
        save_df = df[['Open', 'High', 'Low', 'Close', 'Volume'] + available_cols].copy()
        
        # Convert to JSON-serializable format
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker.upper(),
            'timeframe': timeframe,
            'data': save_df.reset_index().to_json(date_format='iso', orient='records')
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"[TV Cache] Saved TradingView signals for {ticker} ({timeframe})")
        return True
        
    except Exception as e:
        print(f"[TV Cache] Error saving for {ticker}: {e}")
        return False


def load_tv_signals_from_cache(ticker: str, timeframe: str = '1W') -> Optional[pd.DataFrame]:
    """
    Load TradingView signals from cache if available and fresh.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Timeframe label
        
    Returns:
        DataFrame with cached signals, or None if not available/stale
    """
    cache_path = _get_tv_signals_cache_path(ticker, timeframe)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is fresh (within 4 hours for intraday, 24 hours for weekly)
        cached_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
        max_age_hours = 24 if timeframe == '1W' else 4
        
        if datetime.now() - cached_time > timedelta(hours=max_age_hours):
            print(f"[TV Cache] Cache for {ticker} ({timeframe}) is stale")
            return None
        
        from io import StringIO
        df = pd.read_json(StringIO(cache_data['data']), orient='records')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        print(f"[TV Cache] Loaded TradingView signals for {ticker} ({timeframe}) from cache")
        return df
        
    except Exception as e:
        print(f"[TV Cache] Error loading for {ticker}: {e}")
        return None


def check_tv_signals_in_cache(ticker: str, timeframe: str = '1W') -> Dict:
    """
    Check if TradingView signals are cached for a ticker.
    
    Returns:
        Dict with 'has_data', 'timestamp', 'is_fresh'
    """
    cache_path = _get_tv_signals_cache_path(ticker, timeframe)
    
    if not cache_path.exists():
        return {'has_data': False, 'timestamp': None, 'is_fresh': False}
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
        max_age_hours = 24 if timeframe == '1W' else 4
        is_fresh = datetime.now() - cached_time < timedelta(hours=max_age_hours)
        
        return {
            'has_data': True,
            'timestamp': cached_time,
            'is_fresh': is_fresh
        }
        
    except Exception:
        return {'has_data': False, 'timestamp': None, 'is_fresh': False}


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators and generate trading signals.
    
    Adds the following columns to the DataFrame:
    - SMA_50: 50-day Simple Moving Average
    - SMA_200: 200-day Simple Moving Average
    - RSI_14: 14-day Relative Strength Index
    - BB_Middle: Bollinger Band middle (20-day SMA)
    - BB_Upper: Bollinger Band upper (2 std dev)
    - BB_Lower: Bollinger Band lower (2 std dev)
    - Signal: Trading signal (BUY, SELL, GOLDEN CROSS, or empty)
    
    Signal Logic:
    - BUY: Price < Lower Bollinger Band AND RSI < 35
    - SELL: Price > Upper Bollinger Band AND RSI > 65
    - GOLDEN CROSS: 50-day SMA crosses above 200-day SMA
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        
    Returns:
        DataFrame with added indicator and signal columns
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    close = df['Close']
    
    # Trend Indicators: SMAs
    df['SMA_50'] = calculate_sma(close, 50)
    df['SMA_200'] = calculate_sma(close, 200)
    
    # Momentum Indicator: RSI
    df['RSI_14'] = calculate_rsi(close, 14)
    
    # Volatility Indicators: Bollinger Bands
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(close, 20, 2.0)
    
    # Initialize Signal column
    df['Signal'] = ''
    
    # Golden Cross Detection: 50-day SMA crosses above 200-day SMA
    df['SMA_50_prev'] = df['SMA_50'].shift(1)
    df['SMA_200_prev'] = df['SMA_200'].shift(1)
    
    golden_cross = (
        (df['SMA_50'] > df['SMA_200']) & 
        (df['SMA_50_prev'] <= df['SMA_200_prev'])
    )
    
    # Death Cross Detection: 50-day SMA crosses below 200-day SMA
    death_cross = (
        (df['SMA_50'] < df['SMA_200']) & 
        (df['SMA_50_prev'] >= df['SMA_200_prev'])
    )
    
    # BUY Signal: Price below lower BB AND RSI oversold
    buy_signal = (close < df['BB_Lower']) & (df['RSI_14'] < 35)
    
    # SELL Signal: Price above upper BB AND RSI overbought  
    sell_signal = (close > df['BB_Upper']) & (df['RSI_14'] > 65)
    
    # Apply signals (priority: Golden Cross > Death Cross > Buy > Sell)
    df.loc[sell_signal, 'Signal'] = 'SELL'
    df.loc[buy_signal, 'Signal'] = 'BUY'
    df.loc[death_cross, 'Signal'] = 'DEATH CROSS'
    df.loc[golden_cross, 'Signal'] = 'GOLDEN CROSS'
    
    # Clean up temporary columns
    df.drop(columns=['SMA_50_prev', 'SMA_200_prev'], inplace=True)
    
    return df


def get_current_signal(df: pd.DataFrame) -> str:
    """
    Get the current/most recent signal for a ticker.
    
    Args:
        df: DataFrame with signals calculated
        
    Returns:
        Current signal status string
    """
    if df is None or df.empty or 'Signal' not in df.columns:
        return 'NEUTRAL'
    
    # Get the most recent signal (non-empty)
    recent_signals = df[df['Signal'] != '']['Signal']
    
    if recent_signals.empty:
        return 'NEUTRAL'
    
    # Return most recent signal
    return recent_signals.iloc[-1]


def _calculate_importance_score(
    rsi: Optional[float],
    signal: str,
    bb_pct: Optional[float],
    days_to_earnings: Optional[int],
    current_price: float,
    alert_price: Optional[float]
) -> int:
    """
    Calculate importance score for watchlist sorting.
    
    Technical signals get highest weight as per user preference.
    
    Args:
        rsi: Current RSI value (0-100)
        signal: Technical signal string
        bb_pct: Bollinger Band percentage (0-100, can exceed bounds)
        days_to_earnings: Days until next earnings report
        current_price: Current stock price
        alert_price: User's alert price target
        
    Returns:
        Importance score (0-100, higher = more important)
    """
    score = 0
    
    # Technical signals - HIGHEST WEIGHT (40 pts max)
    signal_scores = {
        'GOLDEN CROSS': 40,
        'DEATH CROSS': 40,
        'BUY': 30,
        'SELL': 30,
        'HOLD': 0
    }
    score += signal_scores.get(signal, 0)
    
    # RSI extremes (30 pts max)
    if rsi is not None:
        if rsi < 30 or rsi > 70:
            score += 30  # Extreme oversold/overbought
        elif rsi < 40 or rsi > 60:
            score += 15  # Approaching extremes
    
    # Bollinger Band breakout (15 pts max)
    if bb_pct is not None:
        if bb_pct < 0 or bb_pct > 100:
            score += 15  # Outside bands
        elif bb_pct < 10 or bb_pct > 90:
            score += 8   # Near bands
    
    # Earnings proximity (10 pts max)
    if days_to_earnings is not None:
        if days_to_earnings <= 7:
            score += 10
        elif days_to_earnings <= 14:
            score += 5
    
    # Alert proximity (5 pts max)
    if alert_price is not None and current_price > 0:
        pct_from_alert = abs(current_price - alert_price) / current_price * 100
        if pct_from_alert <= 5:
            score += 5
        elif pct_from_alert <= 10:
            score += 2
    
    return score


def get_ticker_summary(ticker: str, alert_price: Optional[float] = None) -> Optional[Dict]:
    """
    Get a comprehensive summary for a ticker including price, signals, and valuation.
    
    Enhanced version with:
    - 3-month price change (replacing daily)
    - 52-week high/low positioning
    - P/E and PEG ratios
    - Next earnings date
    - Dividend yield
    - Volume vs average
    - Bollinger Band % position
    - Importance score for watchlist sorting
    
    Args:
        ticker: Stock ticker symbol
        alert_price: Optional user alert price for importance calculation
        
    Returns:
        Dict with comprehensive ticker data, or None if data unavailable
    """
    df = fetch_ohlcv(ticker)
    
    if df is None or len(df) < 2:
        return None
    
    df = calculate_signals(df)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100
    
    # 3-month change (approximately 63 trading days)
    trading_days_3m = min(63, len(df) - 1)
    price_3m_ago = df['Close'].iloc[-trading_days_3m - 1] if len(df) > trading_days_3m else df['Close'].iloc[0]
    change_3m = current_price - price_3m_ago
    change_3m_pct = (change_3m / price_3m_ago) * 100 if price_3m_ago > 0 else 0
    
    # 52-week high/low (approximately 252 trading days)
    trading_days_52w = min(252, len(df))
    recent_data = df.tail(trading_days_52w)
    high_52w = recent_data['High'].max()
    low_52w = recent_data['Low'].min()
    pct_from_52w_high = ((current_price - high_52w) / high_52w) * 100 if high_52w > 0 else 0
    
    current_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else None
    current_signal = get_current_signal(df)
    
    # Bollinger Band % position (0 = at lower band, 100 = at upper band)
    bb_upper = df['BB_Upper'].iloc[-1] if 'BB_Upper' in df.columns else None
    bb_lower = df['BB_Lower'].iloc[-1] if 'BB_Lower' in df.columns else None
    bb_pct = None
    if bb_upper and bb_lower and bb_upper != bb_lower:
        bb_pct = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
    
    # Volume vs average (20-day average)
    current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else None
    avg_volume_20d = df['Volume'].tail(20).mean() if 'Volume' in df.columns else None
    vol_vs_avg = None
    if current_volume and avg_volume_20d and avg_volume_20d > 0:
        vol_vs_avg = current_volume / avg_volume_20d
    
    # Determine trend
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else None
    sma_200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns else None
    
    if sma_50 and sma_200:
        if sma_50 > sma_200:
            trend = 'BULLISH'
        elif sma_50 < sma_200:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
    else:
        trend = 'NEUTRAL'
    
    # Fetch additional data from yfinance (P/E, PEG, earnings date, dividend yield, etc.)
    # Uses caching to avoid hitting rate limits every page load
    ticker_info = _fetch_and_cache_ticker_info(ticker)
    
    pe_ratio = ticker_info.get('pe_ratio')
    forward_pe = ticker_info.get('forward_pe')
    peg_ratio = ticker_info.get('peg_ratio')
    dividend_yield = ticker_info.get('dividend_yield')
    market_cap = ticker_info.get('market_cap')
    market_cap_category = ticker_info.get('market_cap_category')
    earnings_date = ticker_info.get('earnings_date')
    days_to_earnings = ticker_info.get('days_to_earnings')
    short_percent = ticker_info.get('short_percent')
    analyst_target = ticker_info.get('analyst_target')
    
    # Calculate importance score
    importance_score = _calculate_importance_score(
        rsi=current_rsi,
        signal=current_signal,
        bb_pct=bb_pct,
        days_to_earnings=days_to_earnings,
        current_price=current_price,
        alert_price=alert_price
    )
    
    return {
        'ticker': ticker.upper(),
        'current_price': round(current_price, 2),
        'daily_change': round(daily_change, 2),
        'daily_change_pct': round(daily_change_pct, 2),
        'change_3m': round(change_3m, 2),
        'change_3m_pct': round(change_3m_pct, 2),
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'pct_from_52w_high': round(pct_from_52w_high, 2),
        'rsi': round(current_rsi, 2) if current_rsi else None,
        'signal': current_signal,
        'trend': trend,
        'sma_50': round(sma_50, 2) if sma_50 else None,
        'sma_200': round(sma_200, 2) if sma_200 else None,
        'bb_upper': round(bb_upper, 2) if bb_upper else None,
        'bb_lower': round(bb_lower, 2) if bb_lower else None,
        'bb_pct': round(bb_pct, 1) if bb_pct is not None else None,
        'vol_vs_avg': round(vol_vs_avg, 2) if vol_vs_avg else None,
        'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
        'forward_pe': round(forward_pe, 2) if forward_pe else None,
        'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
        'dividend_yield': round(dividend_yield, 2) if dividend_yield else None,
        'market_cap': market_cap,
        'market_cap_category': market_cap_category,
        'earnings_date': str(earnings_date) if earnings_date else None,
        'days_to_earnings': days_to_earnings,
        'short_percent': round(short_percent, 2) if short_percent else None,
        'analyst_target': round(analyst_target, 2) if analyst_target else None,
        'importance_score': importance_score,
    }


def clear_cache(ticker: Optional[str] = None):
    """
    Clear cache for a specific ticker or all tickers.
    
    Clears both OHLCV data cache and ticker info cache.
    
    Args:
        ticker: If provided, clear only this ticker's cache. Otherwise, clear all.
    """
    if ticker:
        # Clear OHLCV cache
        cache_path = _get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()
            print(f"OHLCV cache cleared for {ticker}")
        
        # Clear ticker info cache
        info_cache_path = _get_ticker_info_cache_path(ticker)
        if info_cache_path.exists():
            info_cache_path.unlink()
            print(f"Ticker info cache cleared for {ticker}")
    else:
        if CACHE_DIR.exists():
            # Clear all OHLCV caches
            for cache_file in CACHE_DIR.glob("*_ohlcv.json"):
                cache_file.unlink()
            # Clear all ticker info caches
            for cache_file in CACHE_DIR.glob("*_info.json"):
                cache_file.unlink()
            print("All caches cleared (OHLCV + ticker info)")


def _get_forward_growth_estimate(stock, ticker: str) -> Optional[float]:
    """
    Get forward-looking long-term EPS growth estimate (3-5 year CAGR).
    
    Tries multiple sources in order of preference:
    1. yfinance growth_estimates DataFrame (analyst consensus)
    2. yfinance info 'earningsGrowth' as fallback (less ideal - often trailing)
    
    Args:
        stock: yfinance Ticker object
        ticker: Stock ticker symbol for logging
        
    Returns:
        Long-term EPS growth rate as a decimal (0.25 = 25%), or None if unavailable
    """
    try:
        # Priority 1: Try to get analyst growth estimates from yfinance
        # This DataFrame contains forward-looking consensus estimates
        growth_est = stock.growth_estimates
        
        if growth_est is not None and not growth_est.empty:
            # The growth_estimates DataFrame has columns for different periods
            # and an index with the ticker symbol
            # Look for "Next 5 Years (per annum)" or similar long-term estimate
            
            # Check if ticker is in the index
            if ticker.upper() in growth_est.index:
                row = growth_est.loc[ticker.upper()]
                
                # Try to get 5-year growth first, then fall back to other periods
                for col_name in ['Next 5 Years (per annum)', 'Next 5Y', '5Y', 
                                 'Next Year', '+1Y', 'Current Year', '0Y']:
                    if col_name in row.index:
                        value = row[col_name]
                        if pd.notna(value) and value != 0:
                            # Value might be a percentage string like "25.50%" or a decimal
                            if isinstance(value, str):
                                value = float(value.replace('%', '')) / 100
                            print(f"[{ticker}] Forward growth estimate ({col_name}): {value*100:.1f}%")
                            return float(value)
            
            # Alternative: check if it's a different structure
            # Some versions have 'Growth' as a row index
            if 'Growth' in growth_est.index or 'Earnings Growth' in growth_est.index:
                idx = 'Growth' if 'Growth' in growth_est.index else 'Earnings Growth'
                for col in growth_est.columns:
                    if 'next 5' in str(col).lower() or '5 year' in str(col).lower():
                        value = growth_est.loc[idx, col]
                        if pd.notna(value) and value != 0:
                            if isinstance(value, str):
                                value = float(value.replace('%', '')) / 100
                            print(f"[{ticker}] Forward growth estimate: {value*100:.1f}%")
                            return float(value)
        
        # Priority 2: Try analyst earnings estimates to calculate implied growth
        earnings_est = stock.earnings_estimate
        if earnings_est is not None and not earnings_est.empty:
            # Try to calculate growth from current year to next year estimates
            try:
                if 'avg' in earnings_est.index:
                    cols = list(earnings_est.columns)
                    if len(cols) >= 2:
                        current_est = earnings_est.loc['avg', cols[0]]
                        next_est = earnings_est.loc['avg', cols[1]]
                        if pd.notna(current_est) and pd.notna(next_est) and current_est > 0:
                            implied_growth = (next_est - current_est) / current_est
                            if 0 < implied_growth < 2:  # Sanity check: 0-200% growth
                                print(f"[{ticker}] Implied forward growth from estimates: {implied_growth*100:.1f}%")
                                return implied_growth
            except Exception:
                pass
        
        print(f"[{ticker}] No forward growth estimates available from analysts")
        return None
        
    except Exception as e:
        print(f"[{ticker}] Error getting forward growth estimate: {e}")
        return None


def fetch_valuation_data(ticker: str, years: int = 2, skip_db: bool = False) -> Optional[Dict]:
    """
    Fetch P/E ratio and revenue growth data for valuation analysis.
    
    Priority:
    1. Check PostgreSQL database (if fresh and sufficient data)
    2. Alpha Vantage APIs
    3. Finnhub API
    4. FMP API
    5. yfinance fallback
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of data (2, 5, or 10)
        skip_db: If True, skip database check and fetch fresh data
        
    Returns:
        Dict with valuation data, or None if unavailable
    """
    ticker = ticker.upper().strip()
    print(f"\n[fetch_valuation_data] Starting for {ticker} with years={years}")
    
    # Priority 0: Check database first (unless skipped)
    # Only use DB if it has at least 90% of requested quarters, so we don't serve thin
    # history when a fresh API call could return more (e.g. "best of" multiple sources).
    # e.g., 5 years (20 quarters) → need at least 18 quarters in DB; else hit API for more.
    requested_quarters = years * 4
    min_quarters_needed = max(int(requested_quarters * 0.90), 6)  # At least 6 quarters minimum
    
    if not skip_db:
        db_data = load_valuation_from_db(ticker, min_quarters=min_quarters_needed)
        if db_data:
            print(f"[DB] Using cached data for {ticker}")
            # Still need to get current metrics from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_pe = info.get('trailingPE')
                forward_pe = info.get('forwardPE')
                
                # Get forward-looking growth estimate for PEG calculation
                forward_growth = _get_forward_growth_estimate(stock, ticker)
                earnings_growth = forward_growth if forward_growth else info.get('earningsGrowth')
                
                # Calculate PEG using industry standard: Forward P/E / Long-term EPS Growth Rate
                peg_ratio = None
                if forward_pe and forward_growth and forward_growth > 0:
                    growth_pct = forward_growth * 100
                    peg_ratio = forward_pe / growth_pct
                    print(f"[{ticker}] PEG calculated: {forward_pe:.1f} / {growth_pct:.1f}% = {peg_ratio:.2f}")
                elif forward_pe and earnings_growth and earnings_growth > 0:
                    # Fallback to trailing growth
                    growth_pct = earnings_growth * 100
                    peg_ratio = forward_pe / growth_pct
                    print(f"[{ticker}] PEG (fallback to trailing): {forward_pe:.1f} / {growth_pct:.1f}% = {peg_ratio:.2f}")
                
                # Limit to requested years
                pe_history = db_data['pe_history'][:years * 4]
                revenue_history = db_data['revenue_history'][:years * 4]
                
                return {
                    'ticker': ticker,
                    'current_pe': round(current_pe, 2) if current_pe else None,
                    'forward_pe': round(forward_pe, 2) if forward_pe else None,
                    'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
                    'revenue_history': revenue_history,
                    'pe_history': pe_history,
                    'revenue_estimates': [],
                    'market_cap': info.get('marketCap'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'earnings_growth': earnings_growth,
                    'from_database': True
                }
            except Exception as e:
                print(f"[DB] Error getting current metrics: {e}")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current valuation metrics
        current_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        
        # Get forward-looking long-term EPS growth estimate (industry standard for PEG)
        # This is the 3-5 year expected CAGR from analyst estimates
        forward_growth = _get_forward_growth_estimate(stock, ticker)
        
        # Store for reference (will be returned in the result)
        earnings_growth = forward_growth if forward_growth else info.get('earningsGrowth')
        
        # Calculate PEG using industry standard: Forward P/E / Long-term EPS Growth Rate
        # Only use provider's PEG if we don't have forward growth data to calculate our own
        if forward_pe and forward_growth and forward_growth > 0:
            # forward_growth is a decimal (0.25 = 25%), convert to percentage for PEG formula
            growth_pct = forward_growth * 100
            peg_ratio = forward_pe / growth_pct
            print(f"[{ticker}] PEG calculated: {forward_pe:.1f} / {growth_pct:.1f}% = {peg_ratio:.2f}")
        elif peg_ratio is None and forward_pe:
            # Fallback: use trailing earnings growth if no forward estimate available
            trailing_growth = info.get('earningsGrowth')
            if trailing_growth and trailing_growth > 0:
                growth_pct = trailing_growth * 100
                peg_ratio = forward_pe / growth_pct
                print(f"[{ticker}] PEG (fallback to trailing): {forward_pe:.1f} / {growth_pct:.1f}% = {peg_ratio:.2f}")
        
        # Optionally use FinanceToolkit for current P/E and PEG (consistent methodology)
        try:
            from financetoolkit_adapter import (
                USE_FINANCETOOLKIT,
                fetch_current_valuation_ratios_financetoolkit,
            )
            if USE_FINANCETOOLKIT:
                ft_ratios = fetch_current_valuation_ratios_financetoolkit(ticker)
                if ft_ratios:
                    if ft_ratios.get("current_pe") is not None:
                        current_pe = ft_ratios["current_pe"]
                    if ft_ratios.get("peg_ratio") is not None:
                        peg_ratio = ft_ratios["peg_ratio"]
        except ImportError:
            pass
        
        # Get revenue data - try Alpha Vantage first for better historical coverage
        revenue_history = []
        
        # Priority 1: Alpha Vantage Income Statement (best historical coverage)
        if ALPHA_VANTAGE_API_KEY:
            av_revenue = _fetch_alpha_vantage_income_statement(ticker)
            if av_revenue:
                # Sort by date ascending (oldest first)
                av_revenue.sort(key=lambda x: x['date'])
                
                # Calculate YoY growth
                for i, item in enumerate(av_revenue):
                    yoy_growth = None
                    qoq_growth = None
                    
                    # Find same quarter last year (4 quarters back)
                    if i >= 4:
                        prev_revenue = av_revenue[i - 4]['revenue']
                        if prev_revenue > 0:
                            yoy_growth = ((item['revenue'] - prev_revenue) / prev_revenue) * 100
                    
                    # QoQ growth
                    if i >= 1:
                        prev_q_revenue = av_revenue[i - 1]['revenue']
                        if prev_q_revenue > 0:
                            qoq_growth = ((item['revenue'] - prev_q_revenue) / prev_q_revenue) * 100
                    
                    growth = yoy_growth if yoy_growth is not None else qoq_growth
                    growth_type = 'yoy' if yoy_growth is not None else 'qoq'
                    
                    revenue_history.append({
                        'date': item['date'],
                        'revenue': item['revenue'],
                        'yoy_growth': round(yoy_growth, 2) if yoy_growth is not None else None,
                        'qoq_growth': round(qoq_growth, 2) if qoq_growth is not None else None,
                        'growth': round(growth, 2) if growth is not None else None,
                        'growth_type': growth_type if growth is not None else None,
                        'is_annual': False
                    })
                
                # Sort descending and limit
                revenue_history.sort(key=lambda x: x['date'], reverse=True)
                revenue_history = revenue_history[:years * 4]
                print(f"[{ticker}] Using Alpha Vantage revenue data: {len(revenue_history)} quarters")
        
        # Fallback: yfinance if Alpha Vantage didn't work
        if not revenue_history:
            print(f"[{ticker}] Falling back to yfinance for revenue data...")
            revenue_history = _fetch_revenue_from_yfinance(stock, ticker, years)
        
        # Print API configuration status (once)
        _print_api_status()
        
        # Get historical P/E - try all available sources and use the one with most quarters
        # (Time range toggle and abundance: sources like Finnhub often return only ~8 quarters;
        #  yfinance or Alpha Vantage may have more history for 5Y/10Y.)
        pe_history = []
        pe_candidates = []  # list of (list of dicts, source_name)
        
        if ALPHA_VANTAGE_API_KEY:
            av_pe = _calculate_pe_from_alpha_vantage(ticker, years)
            if av_pe:
                pe_candidates.append((av_pe, 'alpha_vantage'))
        if FINNHUB_API_KEY:
            fh_pe = _calculate_pe_from_finnhub(ticker, years)
            if fh_pe:
                pe_candidates.append((fh_pe, 'finnhub'))
        if FMP_API_KEY:
            fmp_pe = _fetch_fmp_historical_pe(ticker, years)
            if fmp_pe:
                pe_candidates.append((fmp_pe, 'fmp'))
        
        yf_pe = _calculate_pe_from_earnings(stock, years)
        if yf_pe:
            pe_candidates.append((yf_pe, 'calculated'))
        
        # Pick the series with the most quarters so the chart has the most data
        if pe_candidates:
            pe_candidates.sort(key=lambda x: len(x[0]), reverse=True)
            pe_history = pe_candidates[0][0][:years * 4]  # cap to requested range
            source_used = pe_candidates[0][1]
            print(f"[{ticker}] P/E: using {source_used} ({len(pe_history)} quarters)")
        else:
            print(f"[{ticker}] No P/E from APIs; trying simple estimation...")
            pe_history = _calculate_historical_pe_simple(stock, info, years)
        
        # Get analyst revenue estimates (forward projections)
        revenue_estimates = _get_revenue_estimates(stock)
        
        return {
            'ticker': ticker,
            'current_pe': round(current_pe, 2) if current_pe else None,
            'forward_pe': round(forward_pe, 2) if forward_pe else None,
            'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
            'revenue_history': revenue_history[:years * 4],  # Limit to requested years
            'pe_history': pe_history,
            'revenue_estimates': revenue_estimates,
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'earnings_growth': earnings_growth
        }
        
    except Exception as e:
        print(f"Error fetching valuation data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fetch_revenue_from_yfinance(stock, ticker: str, years: int) -> list:
    """
    Fetch revenue data from yfinance as a fallback.
    
    Returns list of revenue history dicts.
    """
    revenue_history = []
    
    try:
        # Get financial data - try income statement for revenue
        income_stmt = None
        
        try:
            income_stmt = stock.quarterly_income_stmt
            if income_stmt is not None and not income_stmt.empty:
                print(f"[{ticker}] quarterly_income_stmt shape: {income_stmt.shape}")
        except Exception as e:
            print(f"Error getting quarterly_income_stmt: {e}")
        
        # Fallback to quarterly_financials if income_stmt is empty
        if income_stmt is None or income_stmt.empty:
            try:
                income_stmt = stock.quarterly_financials
                if income_stmt is not None and not income_stmt.empty:
                    print(f"[{ticker}] Using quarterly_financials instead, shape: {income_stmt.shape}")
            except Exception as e:
                print(f"Error getting quarterly_financials: {e}")
        
        if income_stmt is None or income_stmt.empty:
            print(f"[{ticker}] No income statement data available from yfinance")
            return []
        
        # For longer time periods (5+ years), also try annual data
        annual_stmt = None
        if years >= 5:
            try:
                annual_stmt = stock.income_stmt
                if annual_stmt is not None and not annual_stmt.empty:
                    print(f"[{ticker}] Also loaded annual income_stmt shape: {annual_stmt.shape}")
            except Exception as e:
                print(f"Error getting annual income_stmt: {e}")
        
        # Extract revenue data (Total Revenue row)
        revenue_row = None
        for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue', 'TotalRevenue']:
            if row_name in income_stmt.index:
                revenue_row = income_stmt.loc[row_name]
                break
        
        # Also extract annual revenue for longer time periods
        annual_revenue_row = None
        if annual_stmt is not None and not annual_stmt.empty:
            for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue', 'TotalRevenue']:
                if row_name in annual_stmt.index:
                    annual_revenue_row = annual_stmt.loc[row_name]
                    break
        
        if revenue_row is None and annual_revenue_row is None:
            print(f"[{ticker}] No revenue row found. Available rows: {list(income_stmt.index)[:10]}")
            return []
        
        # Build revenue history from yfinance data
        date_value_pairs = []
        
        # Process quarterly data
        if revenue_row is not None:
            print(f"[{ticker}] Quarterly revenue dates: {len(revenue_row.index)} quarters")
            
            for idx, val in revenue_row.items():
                if pd.notna(val) and val > 0:
                    try:
                        if isinstance(idx, pd.Timestamp):
                            dt = idx.to_pydatetime()
                        elif hasattr(idx, 'to_pydatetime'):
                            dt = idx.to_pydatetime()
                        else:
                            dt = pd.to_datetime(idx).to_pydatetime()
                        
                        if dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        
                        date_value_pairs.append((dt, float(val), False))
                    except Exception as e:
                        continue
        
        # Process annual data for longer time periods
        if annual_revenue_row is not None and years >= 5:
            print(f"[{ticker}] Annual revenue dates: {len(annual_revenue_row.index)} years")
            quarterly_dates = set(dt.year for dt, _, _ in date_value_pairs)
            
            for idx, val in annual_revenue_row.items():
                if pd.notna(val) and val > 0:
                    try:
                        if isinstance(idx, pd.Timestamp):
                            dt = idx.to_pydatetime()
                        elif hasattr(idx, 'to_pydatetime'):
                            dt = idx.to_pydatetime()
                        else:
                            dt = pd.to_datetime(idx).to_pydatetime()
                        
                        if dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        
                        if dt.year not in quarterly_dates:
                            date_value_pairs.append((dt, float(val), True))
                    except Exception:
                        continue
        
        if not date_value_pairs:
            return []
        
        # Sort by date ascending for growth calculation
        date_value_pairs.sort(key=lambda x: x[0])
        
        # Build lookup dicts for growth calculation
        quarterly_revenue = {}
        annual_revenue = {}
        for dt, val, is_annual in date_value_pairs:
            if is_annual:
                annual_revenue[dt.year] = (dt, val)
            else:
                quarter = (dt.month - 1) // 3 + 1
                quarterly_revenue[(dt.year, quarter)] = (dt, val)
        
        # Calculate growth rates
        for i, (dt, value, is_annual) in enumerate(date_value_pairs):
            yoy_growth = None
            qoq_growth = None
            
            if is_annual:
                if dt.year - 1 in annual_revenue:
                    prev_value = annual_revenue[dt.year - 1][1]
                    if prev_value > 0:
                        yoy_growth = ((value - prev_value) / prev_value) * 100
            else:
                quarter = (dt.month - 1) // 3 + 1
                last_year_key = (dt.year - 1, quarter)
                if last_year_key in quarterly_revenue:
                    prev_value = quarterly_revenue[last_year_key][1]
                    if prev_value > 0:
                        yoy_growth = ((value - prev_value) / prev_value) * 100
                
                if i > 0 and not date_value_pairs[i - 1][2]:
                    prev_quarter_value = date_value_pairs[i - 1][1]
                    if prev_quarter_value > 0:
                        qoq_growth = ((value - prev_quarter_value) / prev_quarter_value) * 100
            
            growth = yoy_growth if yoy_growth is not None else qoq_growth
            growth_type = 'yoy' if yoy_growth is not None else 'qoq'
            
            revenue_history.append({
                'date': dt,
                'revenue': value,
                'yoy_growth': round(yoy_growth, 2) if yoy_growth is not None else None,
                'qoq_growth': round(qoq_growth, 2) if qoq_growth is not None else None,
                'growth': round(growth, 2) if growth is not None else None,
                'growth_type': growth_type if growth is not None else None,
                'is_annual': is_annual
            })
        
        # Sort descending and limit
        revenue_history.sort(key=lambda x: x['date'], reverse=True)
        revenue_history = revenue_history[:years * 4]
        
        print(f"[{ticker}] yfinance revenue data: {len(revenue_history)} periods")
        
    except Exception as e:
        print(f"[{ticker}] Error fetching yfinance revenue: {e}")
    
    return revenue_history


def _fetch_alpha_vantage_income_statement(ticker: str) -> list:
    """
    Fetch historical quarterly revenue from Alpha Vantage INCOME_STATEMENT endpoint.
    
    Returns quarterly revenue data going back many years.
    Results are cached for 24 hours to preserve API quota.
    """
    ticker = ticker.upper()
    
    # Check cache first
    cached = _load_av_cache(ticker, 'income_statement')
    if cached:
        return cached
    
    if not ALPHA_VANTAGE_API_KEY:
        return []
    
    revenue_data = []
    
    try:
        # Wait for rate limit if needed (Alpha Vantage free tier: 5 calls/minute)
        _wait_for_av_rate_limit()
        
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        print(f"[Alpha Vantage] Fetching income statement for {ticker}...")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for API limit message
            if 'Note' in data or 'Information' in data:
                print(f"[Alpha Vantage] API limit reached")
                return []
            
            # Get quarterly reports
            quarterly = data.get('quarterlyReports', [])
            
            if quarterly:
                for item in quarterly:
                    try:
                        date_str = item.get('fiscalDateEnding')
                        revenue = item.get('totalRevenue')
                        
                        if date_str and revenue and revenue != 'None':
                            revenue_data.append({
                                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                                'revenue': float(revenue)
                            })
                    except (ValueError, TypeError):
                        continue
                
                print(f"[Alpha Vantage] Retrieved {len(revenue_data)} quarters of revenue data")
                
                # Cache the results
                if revenue_data:
                    _save_av_cache(ticker, 'income_statement', revenue_data)
            else:
                print(f"[Alpha Vantage] No quarterly reports in response")
        else:
            print(f"[Alpha Vantage] Income statement API error: {response.status_code}")
            
    except Exception as e:
        print(f"[Alpha Vantage] Income statement error: {e}")
    
    return revenue_data


def _get_av_cache_path(ticker: str, endpoint: str) -> Path:
    """Get cache file path for Alpha Vantage data."""
    return CACHE_DIR / f"{ticker.upper()}_av_{endpoint}.json"


def _load_av_cache(ticker: str, endpoint: str, max_age_hours: int = 24) -> Optional[list]:
    """Load Alpha Vantage data from cache if fresh enough."""
    cache_path = _get_av_cache_path(ticker, endpoint)
    
    if not cache_path.exists():
        return None
    
    try:
        # Check if cache is fresh enough (default 24 hours)
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(hours=max_age_hours):
            return None
        
        with open(cache_path, 'r') as f:
            cached = json.load(f)
            # Convert date strings back to datetime
            for item in cached:
                if 'date' in item and isinstance(item['date'], str):
                    item['date'] = datetime.strptime(item['date'], '%Y-%m-%d')
            print(f"[Alpha Vantage Cache] Loaded {len(cached)} entries for {ticker} {endpoint}")
            return cached
    except Exception as e:
        print(f"[Alpha Vantage Cache] Error loading: {e}")
        return None


def _save_av_cache(ticker: str, endpoint: str, data: list):
    """Save Alpha Vantage data to cache."""
    cache_path = _get_av_cache_path(ticker, endpoint)
    
    try:
        # Convert datetime to strings for JSON
        cached = []
        for item in data:
            cache_item = item.copy()
            if 'date' in cache_item and hasattr(cache_item['date'], 'strftime'):
                cache_item['date'] = cache_item['date'].strftime('%Y-%m-%d')
            cached.append(cache_item)
        
        with open(cache_path, 'w') as f:
            json.dump(cached, f)
        print(f"[Alpha Vantage Cache] Saved {len(cached)} entries for {ticker} {endpoint}")
    except Exception as e:
        print(f"[Alpha Vantage Cache] Error saving: {e}")


# =============================================================================
# PostgreSQL Database Functions for Valuation History
# =============================================================================

def save_valuation_to_db(ticker: str, pe_history: list, revenue_history: list, source: str = 'api') -> bool:
    """
    Save valuation data (P/E and revenue) to PostgreSQL database.
    
    This stores the data permanently so it doesn't need to be fetched again.
    Only new quarters are added; existing data is updated if source is better.
    
    Args:
        ticker: Stock ticker symbol
        pe_history: List of P/E data dicts with 'date', 'pe', 'ttm_eps', 'price'
        revenue_history: List of revenue data dicts with 'date', 'revenue', 'yoy_growth', 'qoq_growth'
        source: Data source name ('alpha_vantage', 'finnhub', 'yfinance')
        
    Returns:
        True if saved successfully, False otherwise
    """
    ticker = ticker.upper()
    
    try:
        db = _get_db_session()
        ValuationHistory = _get_valuation_model()
        
        saved_count = 0
        
        # Combine P/E and revenue data by date
        data_by_date = {}
        
        # Process P/E history
        for item in pe_history:
            dt = item['date']
            if hasattr(dt, 'date'):
                dt = dt.date() if hasattr(dt, 'date') else dt
            elif isinstance(dt, str):
                dt = datetime.strptime(dt, '%Y-%m-%d').date()
            
            if dt not in data_by_date:
                data_by_date[dt] = {}
            data_by_date[dt]['pe_ratio'] = item.get('pe')
            data_by_date[dt]['ttm_eps'] = item.get('ttm_eps')
            data_by_date[dt]['price_at_quarter'] = item.get('price')
            data_by_date[dt]['source'] = item.get('source', source)
        
        # Process revenue history
        for item in revenue_history:
            dt = item['date']
            if hasattr(dt, 'date'):
                dt = dt.date() if hasattr(dt, 'date') else dt
            elif isinstance(dt, str):
                dt = datetime.strptime(dt, '%Y-%m-%d').date()
            
            if dt not in data_by_date:
                data_by_date[dt] = {}
            data_by_date[dt]['revenue'] = item.get('revenue')
            data_by_date[dt]['revenue_growth_yoy'] = item.get('yoy_growth')
            data_by_date[dt]['revenue_growth_qoq'] = item.get('qoq_growth')
            if 'source' not in data_by_date[dt]:
                data_by_date[dt]['source'] = source
        
        # Save to database
        for quarter_date, data in data_by_date.items():
            # Check if record exists
            existing = db.query(ValuationHistory).filter(
                and_(
                    ValuationHistory.ticker == ticker,
                    ValuationHistory.quarter_date == quarter_date
                )
            ).first()
            
            if existing:
                # Update existing record if we have new data
                if data.get('pe_ratio') and not existing.pe_ratio:
                    existing.pe_ratio = data['pe_ratio']
                    existing.ttm_eps = data.get('ttm_eps')
                    existing.price_at_quarter = data.get('price_at_quarter')
                if data.get('revenue') and not existing.revenue:
                    existing.revenue = data['revenue']
                    existing.revenue_growth_yoy = data.get('revenue_growth_yoy')
                    existing.revenue_growth_qoq = data.get('revenue_growth_qoq')
                existing.fetched_at = datetime.utcnow()
            else:
                # Create new record
                new_record = ValuationHistory(
                    ticker=ticker,
                    quarter_date=quarter_date,
                    pe_ratio=data.get('pe_ratio'),
                    ttm_eps=data.get('ttm_eps'),
                    price_at_quarter=data.get('price_at_quarter'),
                    revenue=data.get('revenue'),
                    revenue_growth_yoy=data.get('revenue_growth_yoy'),
                    revenue_growth_qoq=data.get('revenue_growth_qoq'),
                    data_source=data.get('source', source),
                    fetched_at=datetime.utcnow()
                )
                db.add(new_record)
                saved_count += 1
        
        db.commit()
        print(f"[DB] Saved {saved_count} new quarters for {ticker} (total: {len(data_by_date)} quarters)")
        db.close()
        return True
        
    except Exception as e:
        print(f"[DB] Error saving valuation data: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_valuation_from_db(ticker: str, min_quarters: int = 8) -> Optional[Dict]:
    """
    Load valuation data from PostgreSQL database.
    
    Returns data if:
    1. Most recent data point is within 1 month
    2. We have at least min_quarters of data
    
    Args:
        ticker: Stock ticker symbol
        min_quarters: Minimum number of quarters required (default 8 = 2 years)
        
    Returns:
        Dict with 'pe_history' and 'revenue_history', or None if insufficient data
    """
    ticker = ticker.upper()
    
    try:
        db = _get_db_session()
        ValuationHistory = _get_valuation_model()
        
        # Query all data for this ticker, ordered by date
        records = db.query(ValuationHistory).filter(
            ValuationHistory.ticker == ticker
        ).order_by(ValuationHistory.quarter_date.desc()).all()
        
        db.close()
        
        if not records:
            print(f"[DB] No data found for {ticker}")
            return None
        
        # Check if data is fresh enough
        # Earnings are reported quarterly, typically 4-6 weeks after quarter end
        # e.g., Q3 2025 (ending Sep 30) is reported in Oct/Nov 2025
        #       Q4 2025 (ending Dec 31) is reported in late Jan/Feb 2026
        # So in late January 2026, Q3 2025 is still the most recent available quarter
        # 
        # Use 140 days (~4.5 months) as the staleness threshold:
        # This covers one full quarter (90 days) + earnings reporting delay (~45 days)
        most_recent = records[0].quarter_date
        staleness_threshold = (datetime.now() - timedelta(days=140)).date()
        
        if most_recent < staleness_threshold:
            print(f"[DB] Data for {ticker} is stale (most recent: {most_recent}, threshold: {staleness_threshold})")
            return None
        
        print(f"[DB] Data freshness OK for {ticker} (most recent: {most_recent})")
        
        # Check if we have enough quarters
        if len(records) < min_quarters:
            print(f"[DB] Insufficient data for {ticker} ({len(records)} quarters, need {min_quarters}) - will fetch from API")
            return None
        
        print(f"[DB] Found sufficient data for {ticker}: {len(records)} quarters (needed {min_quarters})")
        
        # Convert to the expected format
        pe_history = []
        revenue_history = []
        
        for record in records:
            # P/E data
            if record.pe_ratio:
                pe_history.append({
                    'date': datetime.combine(record.quarter_date, datetime.min.time()),
                    'pe': record.pe_ratio,
                    'ttm_eps': record.ttm_eps,
                    'price': record.price_at_quarter,
                    'source': record.data_source or 'database'
                })
            
            # Revenue data
            if record.revenue:
                growth = record.revenue_growth_yoy if record.revenue_growth_yoy else record.revenue_growth_qoq
                growth_type = 'yoy' if record.revenue_growth_yoy else 'qoq'
                revenue_history.append({
                    'date': datetime.combine(record.quarter_date, datetime.min.time()),
                    'revenue': record.revenue,
                    'yoy_growth': record.revenue_growth_yoy,
                    'qoq_growth': record.revenue_growth_qoq,
                    'growth': growth,
                    'growth_type': growth_type,
                    'is_annual': False
                })
        
        print(f"[DB] Loaded {len(pe_history)} P/E quarters and {len(revenue_history)} revenue quarters for {ticker}")
        
        return {
            'pe_history': pe_history,
            'revenue_history': revenue_history,
            'from_database': True
        }
        
    except Exception as e:
        print(f"[DB] Error loading valuation data: {e}")
        return None


def check_ticker_in_db(ticker: str) -> Dict:
    """
    Check if a ticker has data in the database and return stats.
    
    Returns:
        Dict with 'has_data', 'quarters', 'most_recent', 'is_fresh'
    """
    ticker = ticker.upper()
    
    try:
        db = _get_db_session()
        ValuationHistory = _get_valuation_model()
        
        records = db.query(ValuationHistory).filter(
            ValuationHistory.ticker == ticker
        ).order_by(ValuationHistory.quarter_date.desc()).all()
        
        db.close()
        
        if not records:
            return {'has_data': False, 'quarters': 0, 'most_recent': None, 'is_fresh': False}
        
        most_recent = records[0].quarter_date
        one_month_ago = (datetime.now() - timedelta(days=35)).date()
        
        return {
            'has_data': True,
            'quarters': len(records),
            'most_recent': most_recent,
            'is_fresh': most_recent >= one_month_ago
        }
        
    except Exception as e:
        print(f"[DB] Error checking ticker: {e}")
        return {'has_data': False, 'quarters': 0, 'most_recent': None, 'is_fresh': False}


def _fetch_alpha_vantage_earnings(ticker: str, retry_on_limit: bool = True) -> list:
    """
    Fetch historical quarterly EPS from Alpha Vantage EARNINGS endpoint.
    
    Alpha Vantage provides split-adjusted historical EPS data.
    Free tier: 25 requests/day, 5 requests/minute. Results are cached for 24 hours.
    
    Args:
        ticker: Stock ticker symbol
        retry_on_limit: If True, retry once after waiting if rate limit is hit
        
    Returns:
        List of dicts with fiscalDateEnding and reportedEPS
    """
    ticker = ticker.upper()
    
    # Check cache first
    cached = _load_av_cache(ticker, 'earnings')
    if cached:
        return cached
    
    if not ALPHA_VANTAGE_API_KEY:
        print("[Alpha Vantage] No API key configured. Set ALPHA_VANTAGE_API_KEY environment variable.")
        return []
    
    earnings_data = []
    
    try:
        # Wait for rate limit if needed (Alpha Vantage free tier: 5 calls/minute)
        _wait_for_av_rate_limit()
        
        params = {
            'function': 'EARNINGS',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        print(f"[Alpha Vantage] Fetching earnings data for {ticker}...")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for API limit message
            if 'Note' in data or 'Information' in data:
                if retry_on_limit:
                    print(f"[Alpha Vantage] Rate limit hit, waiting 12s and retrying...")
                    time.sleep(12)  # Wait for the minute window to pass
                    return _fetch_alpha_vantage_earnings(ticker, retry_on_limit=False)
                else:
                    print(f"[Alpha Vantage] API limit reached or invalid key")
                    return []
            
            quarterly = data.get('quarterlyEarnings', [])
            
            if quarterly:
                for item in quarterly:
                    try:
                        date_str = item.get('fiscalDateEnding')
                        eps = item.get('reportedEPS')
                        
                        if date_str and eps and eps != 'None':
                            earnings_data.append({
                                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                                'eps': float(eps)
                            })
                    except (ValueError, TypeError):
                        continue
                
                print(f"[Alpha Vantage] Retrieved {len(earnings_data)} quarters of EPS data")
                
                # Cache the results
                if earnings_data:
                    _save_av_cache(ticker, 'earnings', earnings_data)
            else:
                print(f"[Alpha Vantage] No earnings data in response")
        else:
            print(f"[Alpha Vantage] API error: {response.status_code}")
            
    except Exception as e:
        print(f"[Alpha Vantage] Error: {e}")
    
    return earnings_data


def _calculate_pe_from_alpha_vantage(ticker: str, years: int = 2) -> list:
    """
    Calculate historical P/E using Alpha Vantage EPS data and yfinance prices.
    
    Both sources provide split-adjusted data, so the P/E calculation is accurate.
    """
    pe_history = []
    
    # Get historical EPS from Alpha Vantage
    earnings_data = _fetch_alpha_vantage_earnings(ticker)
    
    if not earnings_data:
        return pe_history
    
    try:
        # Get historical prices from yfinance (split-adjusted)
        print(f"[Alpha Vantage P/E] Fetching {years} years of price data...")
        stock = yf.Ticker(ticker)
        period_map = {2: "2y", 5: "5y", 10: "10y"}
        period = period_map.get(years, f"{years}y")
        print(f"[Alpha Vantage P/E] Using yfinance period: {period}")
        hist = stock.history(period=period, interval="1d")
        
        if hist.empty:
            print(f"[Alpha Vantage P/E] No price history from yfinance")
            return pe_history
        
        # Remove timezone from price index for comparison
        # Convert to timezone-naive for comparison with Python datetime
        if hist.index.tz is not None:
            # For timezone-aware index, convert to None (removes tz info)
            hist.index = hist.index.tz_convert(None)
        
        print(f"[Alpha Vantage P/E] Price history: {hist.index.min().date()} to {hist.index.max().date()} ({len(hist)} days)")
        print(f"[Alpha Vantage P/E] Earnings data: {len(earnings_data)} quarters")
        if earnings_data:
            earnings_data.sort(key=lambda x: x['date'])
            print(f"[Alpha Vantage P/E] Earnings range: {earnings_data[0]['date'].date()} to {earnings_data[-1]['date'].date()}")
        
        # Sort earnings by date (oldest first)
        earnings_data.sort(key=lambda x: x['date'])
        
        # Calculate TTM EPS and P/E for each quarter.
        # P/E is only valid when TTM EPS > 0; quarters with negative/zero TTM are skipped
        # (e.g. HOOD only has ~7 quarters of positive TTM earnings post-IPO).
        skipped_negative_ttm = 0
        for i in range(len(earnings_data)):
            qtr_date = earnings_data[i]['date']
            
            # Convert to pandas Timestamp for proper comparison with price index
            qtr_timestamp = pd.Timestamp(qtr_date)
            
            # Calculate TTM EPS (sum of this quarter + 3 prior quarters)
            ttm_eps = 0
            quarters_summed = 0
            for j in range(4):
                if i - j >= 0:
                    ttm_eps += earnings_data[i - j]['eps']
                    quarters_summed += 1
            
            if quarters_summed < 4:
                continue  # Not enough data for TTM
            
            if ttm_eps <= 0:
                skipped_negative_ttm += 1
                continue  # Skip negative earnings (P/E undefined)
            
            # Get price at quarter end (or closest available)
            try:
                price_mask = hist.index <= qtr_timestamp
                if price_mask.any():
                    price = hist.loc[price_mask, 'Close'].iloc[-1]
                    
                    pe = price / ttm_eps
                    
                    if 0 < pe < 500:  # Sanity check
                        pe_history.append({
                            'date': qtr_date,
                            'pe': round(pe, 2),
                            'ttm_eps': round(ttm_eps, 2),
                            'price': round(price, 2),
                            'source': 'alpha_vantage'
                        })
            except Exception as e:
                print(f"[Alpha Vantage P/E] Error at {qtr_date}: {e}")
                continue
        
        if skipped_negative_ttm:
            print(f"[Alpha Vantage P/E] Skipped {skipped_negative_ttm} quarters (negative/zero TTM EPS — P/E undefined)")
        print(f"[Alpha Vantage P/E] Calculated {len(pe_history)} quarters of historical P/E")
        if pe_history:
            pe_history_sorted = sorted(pe_history, key=lambda x: x['date'])
            print(f"[Alpha Vantage P/E] P/E date range: {pe_history_sorted[0]['date'].date()} to {pe_history_sorted[-1]['date'].date()}")
        
    except Exception as e:
        print(f"[Alpha Vantage P/E] Error: {e}")
    
    # Sort descending (most recent first) and limit
    pe_history.sort(key=lambda x: x['date'], reverse=True)
    return pe_history[:years * 4]


def _fetch_finnhub_earnings(ticker: str) -> list:
    """
    Fetch historical quarterly EPS from Finnhub EARNINGS endpoint.
    
    Finnhub's /stock/earnings endpoint returns recent quarterly earnings.
    Free tier: 60 API calls/minute (very generous).
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of dicts with date and eps values
    """
    ticker = ticker.upper()
    
    # Check cache first - but only use if we have at least 8 quarters
    cached = _load_av_cache(ticker, 'finnhub_earnings')
    if cached and len(cached) >= 8:
        return cached
    
    if not FINNHUB_API_KEY:
        print("[Finnhub] No API key configured. Set FINNHUB_API_KEY environment variable.")
        return []
    
    earnings_data = []
    
    try:
        # Try the earnings endpoint first
        url = f"{FINNHUB_BASE_URL}/stock/earnings"
        params = {
            'symbol': ticker,
            'token': FINNHUB_API_KEY
        }
        
        print(f"[Finnhub] Fetching earnings data for {ticker}...")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    try:
                        # Finnhub returns 'period' as the quarter end date
                        date_str = item.get('period')
                        eps = item.get('actual')  # Actual reported EPS
                        
                        if date_str and eps is not None:
                            earnings_data.append({
                                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                                'eps': float(eps)
                            })
                    except (ValueError, TypeError):
                        continue
                
                print(f"[Finnhub] Retrieved {len(earnings_data)} quarters of EPS data")
                
                # Only cache if we have meaningful data (8+ quarters)
                if len(earnings_data) >= 8:
                    _save_av_cache(ticker, 'finnhub_earnings', earnings_data)
            else:
                print(f"[Finnhub] No earnings data in response")
        elif response.status_code == 401:
            print(f"[Finnhub] Invalid API key")
        elif response.status_code == 429:
            print(f"[Finnhub] Rate limit reached")
        else:
            print(f"[Finnhub] API error: {response.status_code}")
            
    except Exception as e:
        print(f"[Finnhub] Error fetching earnings: {e}")
    
    return earnings_data


def _calculate_pe_from_finnhub(ticker: str, years: int = 2) -> list:
    """
    Calculate historical P/E using Finnhub EPS data and yfinance prices.
    
    Finnhub provides 30+ years of historical earnings data with 60 calls/min.
    """
    pe_history = []
    
    # Get historical EPS from Finnhub
    earnings_data = _fetch_finnhub_earnings(ticker)
    
    if not earnings_data:
        return pe_history
    
    try:
        # Get historical prices from yfinance (split-adjusted)
        print(f"[Finnhub P/E] Fetching {years} years of price data...")
        stock = yf.Ticker(ticker)
        period_map = {2: "2y", 5: "5y", 10: "10y"}
        period = period_map.get(years, f"{years}y")
        hist = stock.history(period=period, interval="1d")
        
        if hist.empty:
            print(f"[Finnhub P/E] No price history from yfinance")
            return pe_history
        
        # Remove timezone from price index
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        
        print(f"[Finnhub P/E] Price history: {hist.index.min().date()} to {hist.index.max().date()}")
        print(f"[Finnhub P/E] Earnings data: {len(earnings_data)} quarters")
        
        # Sort earnings by date (oldest first)
        earnings_data.sort(key=lambda x: x['date'])
        
        # Calculate TTM EPS and P/E for each quarter
        for i in range(len(earnings_data)):
            qtr_date = earnings_data[i]['date']
            qtr_timestamp = pd.Timestamp(qtr_date)
            
            # Calculate TTM EPS (sum of this quarter + 3 prior quarters)
            ttm_eps = 0
            quarters_summed = 0
            for j in range(4):
                if i - j >= 0:
                    ttm_eps += earnings_data[i - j]['eps']
                    quarters_summed += 1
            
            if quarters_summed < 4:
                continue  # Not enough data for TTM
            
            if ttm_eps <= 0:
                continue  # Skip negative earnings
            
            # Get price at quarter end (or closest available)
            try:
                price_mask = hist.index <= qtr_timestamp
                if price_mask.any():
                    price = hist.loc[price_mask, 'Close'].iloc[-1]
                    
                    pe = price / ttm_eps
                    
                    if 0 < pe < 500:  # Sanity check
                        pe_history.append({
                            'date': qtr_date,
                            'pe': round(pe, 2),
                            'ttm_eps': round(ttm_eps, 2),
                            'price': round(price, 2),
                            'source': 'finnhub'
                        })
            except Exception:
                continue
        
        print(f"[Finnhub P/E] Calculated {len(pe_history)} quarters of historical P/E")
        if pe_history:
            pe_history_sorted = sorted(pe_history, key=lambda x: x['date'])
            print(f"[Finnhub P/E] P/E date range: {pe_history_sorted[0]['date'].date()} to {pe_history_sorted[-1]['date'].date()}")
        
    except Exception as e:
        print(f"[Finnhub P/E] Error: {e}")
    
    # Sort descending (most recent first) and limit
    pe_history.sort(key=lambda x: x['date'], reverse=True)
    return pe_history[:years * 4]


def _fetch_fmp_historical_pe(ticker: str, years: int = 2) -> list:
    """
    Fetch historical P/E ratios from Financial Modeling Prep API.
    
    FMP provides pre-calculated key metrics including P/E ratio for each quarter.
    Free tier: 250 requests/day.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of data to fetch
        
    Returns:
        List of dicts with date and pe values
    """
    if not FMP_API_KEY:
        print("[FMP] No API key configured. Set FMP_API_KEY environment variable.")
        return []
    
    pe_history = []
    
    try:
        # FMP Key Metrics endpoint provides historical P/E
        url = f"{FMP_BASE_URL}/key-metrics/{ticker.upper()}"
        params = {
            'period': 'quarter',
            'limit': years * 4 + 4,  # Extra quarters for safety
            'apikey': FMP_API_KEY
        }
        
        print(f"[FMP] Fetching historical P/E for {ticker}...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    try:
                        date_str = item.get('date')
                        pe_ratio = item.get('peRatio')
                        
                        if date_str and pe_ratio is not None and pe_ratio > 0 and pe_ratio < 1000:
                            dt = datetime.strptime(date_str, '%Y-%m-%d')
                            pe_history.append({
                                'date': dt,
                                'pe': round(float(pe_ratio), 2),
                                'source': 'fmp'
                            })
                    except Exception as e:
                        continue
                
                print(f"[FMP] Retrieved {len(pe_history)} quarters of P/E data for {ticker}")
            else:
                print(f"[FMP] No data returned for {ticker}")
        else:
            print(f"[FMP] API error: {response.status_code} - {response.text[:100]}")
            
    except requests.exceptions.RequestException as e:
        print(f"[FMP] Request failed: {e}")
    except Exception as e:
        print(f"[FMP] Error: {e}")
    
    # Sort by date descending
    pe_history.sort(key=lambda x: x['date'], reverse=True)
    return pe_history[:years * 4]


def _calculate_pe_from_earnings(stock, years: int = 2) -> list:
    """
    Calculate historical P/E ratios from quarterly EPS data (Option B fallback).
    
    This calculates actual historical P/E by dividing each quarter's price
    by the trailing 12-month EPS at that time.
    
    Args:
        stock: yfinance Ticker object
        years: Number of years of data
        
    Returns:
        List of dicts with date and pe values
    """
    pe_history = []
    
    try:
        # Get quarterly income statement for EPS data
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is None or income_stmt.empty:
            return pe_history
        
        # Get historical prices
        period_map = {2: "2y", 5: "5y", 10: "10y"}
        period = period_map.get(years, f"{years}y")
        hist = stock.history(period=period, interval="1d")
        
        if hist.empty:
            return pe_history
        
        # Get Basic EPS or calculate from Net Income / Shares Outstanding
        eps_row = None
        for row_name in ['Basic EPS', 'Diluted EPS', 'BasicEPS', 'DilutedEPS']:
            if row_name in income_stmt.index:
                eps_row = income_stmt.loc[row_name]
                break
        
        # Fallback: calculate EPS from Net Income
        if eps_row is None:
            net_income_row = None
            for row_name in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
                if row_name in income_stmt.index:
                    net_income_row = income_stmt.loc[row_name]
                    break
            
            if net_income_row is None:
                print("[Calc P/E] No EPS or Net Income data available")
                return pe_history
            
            # Get shares outstanding
            shares = stock.info.get('sharesOutstanding', 0)
            if shares <= 0:
                return pe_history
            
            # Calculate EPS
            eps_data = {}
            for col in net_income_row.index:
                ni = net_income_row[col]
                if pd.notna(ni):
                    eps_data[col] = ni / shares
        else:
            # Use reported EPS
            eps_data = {}
            for col in eps_row.index:
                eps = eps_row[col]
                if pd.notna(eps):
                    eps_data[col] = eps
        
        # Calculate TTM EPS for each quarter
        sorted_dates = sorted(eps_data.keys(), reverse=True)
        
        for i, qtr_date in enumerate(sorted_dates):
            # Calculate TTM EPS (sum of 4 quarters)
            ttm_eps = 0
            quarters_counted = 0
            for j in range(4):
                if i + j < len(sorted_dates):
                    ttm_eps += eps_data[sorted_dates[i + j]]
                    quarters_counted += 1
            
            if quarters_counted < 4 or ttm_eps <= 0:
                continue
            
            # Get price at quarter end
            qtr_dt = qtr_date.to_pydatetime() if hasattr(qtr_date, 'to_pydatetime') else qtr_date
            if qtr_dt.tzinfo:
                qtr_dt = qtr_dt.replace(tzinfo=None)
            
            # Find closest price
            try:
                hist_idx = hist.index.tz_convert(None) if hist.index.tz else hist.index
                price_mask = hist_idx <= qtr_dt
                if price_mask.any():
                    closest_price = hist.loc[price_mask, 'Close'].iloc[-1]
                    
                    pe = closest_price / ttm_eps
                    if 0 < pe < 500:  # Sanity check
                        pe_history.append({
                            'date': qtr_dt,
                            'pe': round(pe, 2),
                            'source': 'calculated'
                        })
            except Exception as e:
                continue
        
        print(f"[Calc P/E] Calculated {len(pe_history)} quarters of P/E data")
        
    except Exception as e:
        print(f"[Calc P/E] Error: {e}")
        import traceback
        traceback.print_exc()
    
    pe_history.sort(key=lambda x: x['date'], reverse=True)
    return pe_history[:years * 4]


def _calculate_historical_pe_simple(stock, info, years: int = 2) -> list:
    """
    Calculate historical P/E ratios using price history and current EPS.
    
    This is a simplified approach that estimates historical P/E 
    by using the current trailing EPS as a proxy.
    """
    pe_history = []
    
    try:
        # Get trailing EPS
        trailing_eps = info.get('trailingEps')
        
        if not trailing_eps or trailing_eps <= 0:
            print("No positive trailing EPS available")
            return pe_history
        
        # Get price history based on years parameter
        period_map = {2: "2y", 5: "5y", 10: "10y"}
        period = period_map.get(years, f"{years}y")
        
        hist = stock.history(period=period, interval="1mo")
        if hist.empty:
            return pe_history
        
        # Sample monthly prices and calculate implied P/E
        # Take end of each quarter
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)
        
        # Resample to quarterly (take last value of each quarter)
        quarterly = hist['Close'].resample('QE').last().dropna()
        
        for date, price in quarterly.items():
            if pd.notna(price) and price > 0:
                # Estimate P/E (this assumes EPS was similar historically - rough approximation)
                implied_pe = price / trailing_eps
                
                # Sanity check
                if 0 < implied_pe < 1000:
                    pe_history.append({
                        'date': date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                        'pe': round(implied_pe, 2),
                        'price': round(float(price), 2)
                    })
        
        # Sort by date descending and take appropriate number of quarters
        pe_history.sort(key=lambda x: x['date'], reverse=True)
        max_quarters = years * 4
        return pe_history[:max_quarters]
        
    except Exception as e:
        print(f"Error calculating historical P/E: {e}")
        import traceback
        traceback.print_exc()
        return pe_history


def _get_revenue_estimates(stock) -> list:
    """
    Get analyst revenue estimates for future quarters.
    """
    estimates = []
    
    try:
        # Try to get revenue estimates from yfinance
        revenue_est = stock.revenue_forecasts
        
        if revenue_est is not None and not revenue_est.empty:
            for col in revenue_est.columns:
                try:
                    avg_estimate = revenue_est.loc['avg', col] if 'avg' in revenue_est.index else None
                    if avg_estimate and pd.notna(avg_estimate):
                        estimates.append({
                            'period': str(col),
                            'estimate': float(avg_estimate),
                            'is_projected': True
                        })
                except Exception:
                    continue
    except Exception:
        pass
    
    return estimates[:4]  # Next 4 periods


def get_valuation_chart_data(ticker: str, years: int = 2, skip_db: bool = False) -> Optional[Dict]:
    """
    Get formatted data ready for the valuation chart.
    
    Combines P/E history and revenue growth into chart-ready format.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data (2, 5, or 10)
        skip_db: If True, fetch from API only (use to refresh and get best-available P/E)
        
    Returns:
        Dict with 'pe_data' and 'revenue_data' DataFrames
    """
    print(f"\n[get_valuation_chart_data] Called with ticker={ticker}, years={years}, skip_db={skip_db}")
    valuation = fetch_valuation_data(ticker, years, skip_db=skip_db)
    
    if valuation is None:
        return None
    
    # Format P/E data for charting
    pe_data = []
    pe_source = 'unknown'
    if valuation['pe_history']:
        for item in reversed(valuation['pe_history']):
            # Ensure date is proper datetime
            dt = item['date']
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            pe_data.append({
                'date': dt,
                'pe': item['pe'],
                'is_projected': False,
                'source': item.get('source', 'unknown')
            })
            pe_source = item.get('source', 'unknown')
    
    # Format revenue data for charting
    revenue_data = []
    if valuation['revenue_history']:
        for item in reversed(valuation['revenue_history']):
            # Include entries with any growth calculated (YoY or QoQ)
            growth_val = item.get('growth') or item.get('yoy_growth') or item.get('qoq_growth')
            if growth_val is not None:
                # Ensure date is proper datetime
                dt = item['date']
                if isinstance(dt, str):
                    dt = pd.to_datetime(dt)
                elif hasattr(dt, 'to_pydatetime'):
                    dt = dt.to_pydatetime()
                revenue_data.append({
                    'date': dt,
                    'growth': growth_val,
                    'revenue': item['revenue'],
                    'is_projected': False,
                    'growth_type': item.get('growth_type', 'yoy')
                })
        
        print(f"Revenue data for chart: {len(revenue_data)} quarters with growth data")
    
    # Add projected revenue if available
    if valuation.get('revenue_estimates') and valuation.get('revenue_history'):
        last_revenue = valuation['revenue_history'][0]['revenue'] if valuation['revenue_history'] else 0
        for est in valuation['revenue_estimates']:
            try:
                # Estimate YoY growth from projection
                if last_revenue > 0:
                    projected_growth = ((est['estimate'] - last_revenue) / last_revenue) * 100
                    # Parse the period string to datetime
                    try:
                        est_date = pd.to_datetime(est['period'])
                    except:
                        continue
                    revenue_data.append({
                        'date': est_date,
                        'growth': round(projected_growth, 2),
                        'revenue': est['estimate'],
                        'is_projected': True
                    })
            except Exception:
                continue
    
    # Create DataFrames and ensure dates are timezone-naive
    pe_df = pd.DataFrame()
    if pe_data:
        pe_df = pd.DataFrame(pe_data)
        # Convert to datetime and remove timezone if present
        pe_df['date'] = pd.to_datetime(pe_df['date'])
        if pe_df['date'].dt.tz is not None:
            pe_df['date'] = pe_df['date'].dt.tz_convert(None)
        pe_df = pe_df.sort_values('date').reset_index(drop=True)
        print(f"[{ticker}] P/E DataFrame: {len(pe_df)} entries, {pe_df['date'].min()} to {pe_df['date'].max()}")
    
    revenue_df = pd.DataFrame()
    if revenue_data:
        revenue_df = pd.DataFrame(revenue_data)
        # Convert to datetime and remove timezone if present
        revenue_df['date'] = pd.to_datetime(revenue_df['date'])
        if revenue_df['date'].dt.tz is not None:
            revenue_df['date'] = revenue_df['date'].dt.tz_convert(None)
        revenue_df = revenue_df.sort_values('date').reset_index(drop=True)
        print(f"[{ticker}] Revenue DataFrame: {len(revenue_df)} entries, {revenue_df['date'].min()} to {revenue_df['date'].max()}")
    
    # Pass through from_database so UI can show "From DB" when data came from PostgreSQL
    from_db = valuation.get('from_database', False)

    # Auto-save to DB when data came from API (so next time it's a DB hit and no API cost)
    if not from_db and valuation.get('pe_history') and valuation.get('revenue_history'):
        pe_list = []
        for item in valuation['pe_history']:
            pe_list.append({
                'date': item['date'],
                'pe': item['pe'],
                'ttm_eps': item.get('ttm_eps'),
                'price': item.get('price'),
                'source': item.get('source', 'api')
            })
        revenue_list = []
        for item in valuation['revenue_history']:
            revenue_list.append({
                'date': item['date'],
                'revenue': item.get('revenue'),
                'yoy_growth': item.get('yoy_growth'),
                'qoq_growth': item.get('qoq_growth')
            })
        try:
            if save_valuation_to_db(ticker, pe_list, revenue_list):
                print(f"[get_valuation_chart_data] Auto-saved {ticker} valuation to DB")
        except Exception as e:
            print(f"[get_valuation_chart_data] Auto-save to DB skipped: {e}")

    # Return even if we only have metrics (no historical data)
    return {
        'ticker': ticker,
        'pe_data': pe_df,
        'revenue_data': revenue_df,
        'current_pe': valuation['current_pe'],
        'forward_pe': valuation['forward_pe'],
        'peg_ratio': valuation.get('peg_ratio'),
        'sector': valuation.get('sector'),
        'industry': valuation.get('industry'),
        'earnings_growth': valuation.get('earnings_growth'),
        'pe_source': pe_source,  # 'fmp', 'calculated', or 'estimated'
        'from_database': from_db,
        'requested_years': years,
    }


# =============================================================================
# Company Profile (FMP stable) - sector, industry, description, web, employees, CEO
# =============================================================================

def fetch_company_profile_fmp(ticker: str) -> Optional[Dict]:
    """
    Fetch company profile from FMP stable profile API.
    Returns dict with sector, industry, description, website, employees, ceo; or None.
    """
    if not FMP_API_KEY:
        return None
    ticker = ticker.upper().strip()
    try:
        url = f"{FMP_STABLE_URL}/profile"
        params = {"symbol": ticker, "apikey": FMP_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        item = data[0]
        return {
            "ticker": ticker,
            "companyName": item.get("companyName") or item.get("company_name"),
            "sector": item.get("sector") or item.get("industry") or None,
            "industry": item.get("industry") or item.get("sector") or None,
            "description": item.get("description"),
            "website": item.get("website") or item.get("url"),
            "employees": item.get("fullTimeEmployees"),
            "ceo": item.get("ceo"),
            "data_source": "fmp",
        }
    except Exception as e:
        print(f"[FMP Profile] Error for {ticker}: {e}")
        return None


def get_company_profile(ticker: str) -> Optional[Dict]:
    """
    Get company profile with read-through cache (DB, then FMP).
    Returns dict for UI; None if unavailable.
    """
    ticker = ticker.upper().strip()
    try:
        db = _get_db_session()
        from models import CompanyProfile
        row = db.query(CompanyProfile).filter(CompanyProfile.ticker == ticker).first()
        cutoff = datetime.utcnow() - timedelta(days=7)
        if row and row.fetched_at and row.fetched_at.replace(tzinfo=None) >= cutoff:
            out = {
                "ticker": row.ticker,
                "sector": row.sector,
                "industry": row.industry,
                "description": row.description,
                "website": row.website,
                "employees": row.full_time_employees,
                "ceo": row.ceo,
                "data_source": row.data_source,
            }
            db.close()
            return out
        db.close()
    except Exception as e:
        print(f"[CompanyProfile] DB read error: {e}")
        try:
            db.close()
        except Exception:
            pass
    try:
        from openbb_adapter import fetch_profile_openbb
        profile = fetch_profile_openbb(ticker)
    except ImportError:
        profile = None
    if not profile:
        profile = fetch_company_profile_fmp(ticker)
    if not profile:
        return None
    try:
        db = _get_db_session()
        from models import CompanyProfile
        existing = db.query(CompanyProfile).filter(CompanyProfile.ticker == ticker).first()
        if existing:
            existing.sector = profile.get("sector")
            existing.industry = profile.get("industry")
            existing.description = profile.get("description")
            existing.website = profile.get("website")
            existing.full_time_employees = profile.get("employees")
            existing.ceo = profile.get("ceo")
            existing.data_source = profile.get("data_source", "fmp")
            existing.fetched_at = datetime.utcnow()
        else:
            db.add(CompanyProfile(
                ticker=ticker,
                sector=profile.get("sector"),
                industry=profile.get("industry"),
                description=profile.get("description"),
                website=profile.get("website"),
                full_time_employees=profile.get("employees"),
                ceo=profile.get("ceo"),
                data_source=profile.get("data_source", "fmp"),
                fetched_at=datetime.utcnow(),
            ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[CompanyProfile] DB write error: {e}")
        try:
            db.rollback()
            db.close()
        except Exception:
            pass
    return profile


# =============================================================================
# Company Fundamentals (FMP ratios / income) - revenue, margins, ratios
# =============================================================================

def _parse_ratio_item(item: dict) -> dict:
    """Extract margin/ratio values from an FMP ratio object (camelCase or snake_case)."""
    out = {}
    # Margins: FMP often returns as decimal (0.25) or as percentage (25)
    for fmp_key, our_key in [
        ("grossProfitMargin", "gross_margin"),
        ("operatingProfitMargin", "operating_margin"),
        ("netProfitMargin", "net_margin"),
        ("returnOnEquity", "roe"),
        ("returnOnAssets", "roa"),
    ]:
        val = item.get(fmp_key) or item.get(fmp_key[0].lower() + fmp_key[1:])
        if val is not None:
            try:
                v = float(val)
                if our_key.endswith("_margin") and abs(v) > 1 and abs(v) <= 100:
                    v = v / 100.0
                out[our_key] = v
            except (TypeError, ValueError):
                pass
    return out


def fetch_fundamentals_fmp(ticker: str) -> Optional[Dict]:
    """
    Fetch latest ratios and optionally revenue from FMP.
    Uses stable ratios-ttm first (documented), then v3 ratios/income-statement.
    Returns dict with revenue_ttm, gross_margin, operating_margin, net_margin, roe, roa; or None.
    """
    if not FMP_API_KEY:
        return None
    ticker = ticker.upper().strip()
    result = {}
    try:
        # Prefer stable ratios-ttm (documented: ?symbol=)
        url_ttm = f"{FMP_STABLE_URL}/ratios-ttm"
        params_ttm = {"symbol": ticker, "apikey": FMP_API_KEY}
        resp_ttm = requests.get(url_ttm, params=params_ttm, timeout=10)
        if resp_ttm.status_code == 200:
            data_ttm = resp_ttm.json()
            if isinstance(data_ttm, list) and len(data_ttm) > 0:
                result.update(_parse_ratio_item(data_ttm[0]))
            elif isinstance(data_ttm, dict) and data_ttm:
                result.update(_parse_ratio_item(data_ttm))
        if not result:
            url = f"{FMP_BASE_URL}/ratios/{ticker}"
            params = {"period": "quarter", "limit": 4, "apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    result.update(_parse_ratio_item(data[0]))
        # Revenue TTM from income-statement (sum last 4 quarters)
        url_inc = f"{FMP_BASE_URL}/income-statement/{ticker}"
        params_inc = {"period": "quarter", "limit": 4, "apikey": FMP_API_KEY}
        resp_inc = requests.get(url_inc, params=params_inc, timeout=10)
        if resp_inc.status_code == 200:
            data_inc = resp_inc.json()
            if isinstance(data_inc, list) and len(data_inc) > 0:
                total_revenue = sum((row.get("revenue") or 0) for row in data_inc[:4])
                if total_revenue:
                    result["revenue_ttm"] = total_revenue
        if not result:
            return None
        result["ticker"] = ticker
        result["data_source"] = "fmp"
        return result
    except Exception as e:
        print(f"[FMP Fundamentals] Error for {ticker}: {e}")
        return None


def get_fundamentals_ratios(ticker: str) -> Optional[Dict]:
    """
    Get fundamentals/ratios with read-through cache (DB, then FMP).
    Returns dict for UI; None if unavailable.
    """
    ticker = ticker.upper().strip()
    try:
        db = _get_db_session()
        from models import CompanyFundamentals
        row = db.query(CompanyFundamentals).filter(CompanyFundamentals.ticker == ticker).first()
        cutoff = datetime.utcnow() - timedelta(days=1)
        if row and row.fetched_at and row.fetched_at.replace(tzinfo=None) >= cutoff:
            out = {
                "ticker": row.ticker,
                "revenue_ttm": row.revenue_ttm,
                "gross_margin": row.gross_margin,
                "operating_margin": row.operating_margin,
                "net_margin": row.net_margin,
                "roe": row.roe,
                "roa": row.roa,
            }
            db.close()
            return out
        db.close()
    except Exception as e:
        print(f"[CompanyFundamentals] DB read error: {e}")
        try:
            db.close()
        except Exception:
            pass
    # FinanceToolkit first (when enabled), then OpenBB, then FMP
    try:
        from financetoolkit_adapter import (
            USE_FINANCETOOLKIT,
            fetch_fundamentals_financetoolkit,
        )
        if USE_FINANCETOOLKIT:
            data = fetch_fundamentals_financetoolkit(ticker)
        else:
            data = None
    except ImportError:
        data = None
    if not data:
        try:
            from openbb_adapter import fetch_fundamentals_openbb
            data = fetch_fundamentals_openbb(ticker)
        except ImportError:
            data = None
    if not data:
        data = fetch_fundamentals_fmp(ticker)
    if not data:
        return None
    try:
        db = _get_db_session()
        from models import CompanyFundamentals
        existing = db.query(CompanyFundamentals).filter(CompanyFundamentals.ticker == ticker).first()
        if existing:
            existing.revenue_ttm = data.get("revenue_ttm")
            existing.gross_margin = data.get("gross_margin")
            existing.operating_margin = data.get("operating_margin")
            existing.net_margin = data.get("net_margin")
            existing.roe = data.get("roe")
            existing.roa = data.get("roa")
            existing.data_source = data.get("data_source", "fmp")
            existing.fetched_at = datetime.utcnow()
        else:
            db.add(CompanyFundamentals(
                ticker=ticker,
                revenue_ttm=data.get("revenue_ttm"),
                gross_margin=data.get("gross_margin"),
                operating_margin=data.get("operating_margin"),
                net_margin=data.get("net_margin"),
                roe=data.get("roe"),
                roa=data.get("roa"),
                data_source=data.get("data_source", "fmp"),
                fetched_at=datetime.utcnow(),
            ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[CompanyFundamentals] DB write error: {e}")
        try:
            db.rollback()
            db.close()
        except Exception:
            pass
    return {
        "ticker": data.get("ticker"),
        "revenue_ttm": data.get("revenue_ttm"),
        "gross_margin": data.get("gross_margin"),
        "operating_margin": data.get("operating_margin"),
        "net_margin": data.get("net_margin"),
        "roe": data.get("roe"),
        "roa": data.get("roa"),
    }


# =============================================================================
# Company News (Finnhub)
# =============================================================================

def fetch_company_news(ticker: str, limit: int = 10) -> List[Dict]:
    """
    Fetch recent company news from OpenBB (when available), then Finnhub. Returns list of dicts with headline, url, source, datetime.
    """
    ticker = ticker.upper().strip()
    try:
        from openbb_adapter import fetch_news_openbb
        news = fetch_news_openbb(ticker, limit=limit)
        if news:
            return news
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenBB News] fallback for {ticker}: {e}")
    if not FINNHUB_API_KEY:
        return []
    try:
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=14)
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            "symbol": ticker,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        if not isinstance(data, list):
            return []
        out = []
        for item in data[:limit]:
            out.append({
                "headline": item.get("headline") or item.get("summary") or "No title",
                "url": item.get("url") or "",
                "source": item.get("source") or "",
                "datetime": item.get("datetime") or item.get("publishedDate") or "",
            })
        return out
    except Exception as e:
        print(f"[Finnhub News] Error for {ticker}: {e}")
        return []


def _parse_insider_date(value) -> Optional[date]:
    """Parse Finnhub insider date: ISO, 'Mar 08 \'24', Unix ms, or MM/DD/YYYY. Return date or None."""
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date() if hasattr(value, "date") else value
    # Unix timestamp (seconds or milliseconds)
    if isinstance(value, (int, float)):
        try:
            ts = int(value)
            if ts > 1e12:
                ts //= 1000
            return datetime.utcfromtimestamp(ts).date()
        except (ValueError, OSError):
            return None
    s = str(value).strip()
    if not s:
        return None
    # ISO: 2024-03-11 or 2024-03-11T16:34:00.000Z
    if "T" in s:
        s = s.split("T")[0]
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    # "Mar 08 '24" style
    try:
        return datetime.strptime(s, "%b %d '%y").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%b %d, %Y").date()
    except ValueError:
        pass
    # MM/DD/YYYY
    try:
        return datetime.strptime(s[:10], "%m/%d/%Y").date()
    except ValueError:
        pass
    return None


def fetch_insider_transactions(
    ticker: str,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> List[Dict]:
    """
    Fetch insider transactions for a ticker from Finnhub.
    Returns list of dicts: date (date), transaction (Buy/Sale), shares, value, name, relationship, sec_link.
    Cached ~24h by ticker.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []

    cache_path = CACHE_DIR / f"{ticker}_insider.json"
    if cache_path.exists():
        try:
            mtime = cache_path.stat().st_mtime
            if (datetime.now().timestamp() - mtime) / 3600 < 24:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                out = []
                for item in raw:
                    d = item.get("date")
                    if isinstance(d, str):
                        try:
                            d = datetime.strptime(d[:10], "%Y-%m-%d").date()
                        except ValueError:
                            continue
                    out.append({**item, "date": d})
                return out
        except (json.JSONDecodeError, OSError):
            pass

    if not FINNHUB_API_KEY:
        return []

    try:
        url = f"{FINNHUB_BASE_URL}/stock/insider-transactions"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        # Finnhub may use "insiderTransactions" or "data"
        raw_list = (
            data.get("insiderTransactions")
            if isinstance(data, dict) else []
        )
        if not isinstance(raw_list, list) or not raw_list:
            raw_list = data.get("data") if isinstance(data, dict) else []
        if not isinstance(raw_list, list):
            raw_list = []

        out = []
        for t in raw_list:
            if not isinstance(t, dict):
                continue
            dt = _parse_insider_date(
                t.get("SECForm4Date")
                or t.get("date")
                or t.get("reportDate")
                or t.get("filingDate")
            )
            if not dt:
                continue
            trans = (
                (t.get("transaction") or t.get("transactionType") or t.get("type") or "")
                .strip()
            )
            shares_val = int(t.get("shares") or t.get("share") or t.get("numberOfShares") or 0)
            # If no transaction type, infer from shares (negative = sale) or default Sale
            if not trans:
                trans = "Sale" if shares_val <= 0 else "Purchase"
            # Normalize to Buy / Sale for chart
            u = trans.upper()
            if "SALE" in u or u.startswith("S") or u == "D" or "DISPOS" in u:
                trans_norm = "Sale"
            elif "PURCHASE" in u or "BUY" in u or u.startswith("P") or u in ("B", "A", "G") or "ACQUIS" in u:
                trans_norm = "Buy"
            else:
                trans_norm = "Sale"

            out.append({
                "date": dt,
                "transaction": trans_norm,
                "transaction_raw": trans,
                "shares": shares_val,
                "value": int(t.get("USDValue") or t.get("value") or t.get("totalValue") or 0),
                "name": (
                    (t.get("insiderTradings") or t.get("name") or t.get("reportingName") or t.get("insiderName") or "")
                    .strip()
                ),
                "relationship": (t.get("relationship") or t.get("reportingOwnerTitle") or "").strip(),
                "sec_link": (t.get("SECForm4Link") or t.get("secLink") or t.get("filingUrl") or "").strip(),
            })
        # Cache only when we have data (avoid caching empty for 24h)
        if out:
            to_cache = []
            for item in out:
                c = {**item, "date": item["date"].strftime("%Y-%m-%d") if hasattr(item["date"], "strftime") else str(item["date"])}
                to_cache.append(c)
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(to_cache, f)
            except OSError:
                pass
        elif cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass
        return out
    except Exception as e:
        print(f"[Finnhub Insider] Error for {ticker}: {e}")
        return []


# =============================================================================
# Competitors / Peers (FMP screener + description similarity + overrides)
# =============================================================================

PEERS_CACHE_HOURS = 6
PEERS_RESULT_CACHE_MINUTES = 30
MAX_PEERS_DISPLAY = 8

# Buzzwords to exclude from description similarity (reduce noise)
PEERS_DESCRIPTION_STOP_WORDS = {
    "platform", "solutions", "leading", "innovative", "global", "world-class",
    "ai", "cloud", "digital", "technology", "services", "company", "inc",
    "ltd", "corp", "the", "and", "for", "with", "its", "our", "we", "are",
}

# Optional geography filter (FMP screener); set via env PEERS_COUNTRY or PEERS_EXCHANGE
PEERS_COUNTRY = os.environ.get("PEERS_COUNTRY", "")
PEERS_EXCHANGE = os.environ.get("PEERS_EXCHANGE", "")


def _peers_candidates_cache_path(ticker: str) -> Path:
    """Cache path for screener candidates per ticker."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"peers_candidates_{ticker.upper()}.json"


def _load_peers_candidates_cache(ticker: str) -> Optional[Tuple[List[Dict], str, bool]]:
    """Load cached screener result: (candidates, fallback_used, used_sector_fallback)."""
    path = _peers_candidates_cache_path(ticker)
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=PEERS_CACHE_HOURS):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("candidates", []),
            data.get("fallback_used", "industry"),
            data.get("used_sector_fallback", False),
        )
    except Exception:
        return None


def _save_peers_candidates_cache(ticker: str, candidates: List[Dict], fallback_used: str, used_sector_fallback: bool):
    """Save screener result to cache."""
    path = _peers_candidates_cache_path(ticker)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "candidates": candidates,
                "fallback_used": fallback_used,
                "used_sector_fallback": used_sector_fallback,
            }, f)
    except OSError:
        pass


def _normalize_industry_for_fmp(industry: Optional[str]) -> Optional[str]:
    """Normalize profile industry to FMP-style (strip, lower for comparison). FMP often uses Title Case."""
    if not industry or not isinstance(industry, str):
        return None
    s = industry.strip()
    return s if s else None


def _fetch_stock_peers_fmp(ticker: str) -> List[Dict]:
    """
    Fetch peer symbols from FMP stock-peers endpoint (same exchange, sector, similar market cap).
    Often available on free/lower tier when company-screener is restricted.
    Returns list of dicts with at least 'symbol'.
    """
    if not FMP_API_KEY:
        return []
    url = f"{FMP_STABLE_URL}/stock-peers"
    params = {"symbol": ticker.upper().strip(), "apikey": FMP_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code in (402, 403):
            return []
        if r.status_code != 200:
            return []
        data = r.json()
        symbols = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sym = (item.get("symbol") or item.get("Symbol") or "").strip().upper()
                    if sym and sym != ticker.upper().strip():
                        symbols.append({"symbol": sym, "sector": item.get("sector"), "industry": item.get("industry"), "marketCap": item.get("marketCap")})
                elif isinstance(item, str) and item.upper() != ticker.upper().strip():
                    symbols.append({"symbol": item.upper().strip()})
        elif isinstance(data, dict):
            peers_list = data.get("peersList") or data.get("peers") or data.get("symbols") or []
            for s in peers_list:
                sym = (s if isinstance(s, str) else (s.get("symbol") or s.get("Symbol") or "")).strip().upper()
                if sym and sym != ticker.upper().strip():
                    symbols.append({"symbol": sym})
        return symbols
    except Exception as e:
        print(f"[Peers FMP] stock-peers error: {e}")
        return []


def fetch_peer_candidates_fmp(
    ticker: str,
    industry: Optional[str],
    sector: Optional[str],
    market_cap: Optional[float],
    limit: int = 50,
) -> Tuple[List[Dict], str, bool]:
    """
    Fetch peer candidates from FMP company screener with fallback chain.
    Returns (list of candidate dicts with at least 'symbol', 'sector', 'industry', 'marketCap' when present),
    fallback_used in ('industry', 'sector', 'sector_wide_cap'),
    used_sector_fallback (True if industry was not used).
    """
    ticker = ticker.upper().strip()
    if not FMP_API_KEY:
        return [], "industry", False

    # Try cache first (only use if we have candidates; avoid caching empty after 402)
    cached = _load_peers_candidates_cache(ticker)
    if cached is not None and len(cached[0]) > 0:
        return cached[0][:limit], cached[1], cached[2]

    cap_min = None
    cap_max = None
    if market_cap and market_cap > 0:
        cap_min = int(market_cap * 0.33)
        cap_max = int(market_cap * 3.0)
    cap_min_wide = int(market_cap * 0.1) if market_cap and market_cap > 0 else None
    cap_max_wide = int(market_cap * 10) if market_cap and market_cap > 0 else None

    base_url = f"{FMP_STABLE_URL}/company-screener"
    params_base = {"apikey": FMP_API_KEY, "limit": limit}
    if cap_min is not None:
        params_base["marketCapMoreThan"] = cap_min
    if cap_max is not None:
        params_base["marketCapLowerThan"] = cap_max
    if PEERS_COUNTRY:
        params_base["country"] = PEERS_COUNTRY
    if PEERS_EXCHANGE:
        params_base["exchange"] = PEERS_EXCHANGE

    used_sector_fallback = False
    fallback_used = "industry"

    def _is_restricted_response(r: "requests.Response") -> bool:
        """True if FMP returned 402/403 or an error body indicating restricted endpoint."""
        if r.status_code in (402, 403):
            return True
        if r.status_code != 200:
            return False
        try:
            body = r.json()
            if isinstance(body, dict) and ("Error Message" in body or "Restricted" in str(body).lower() or "subscription" in str(body).lower()):
                return True
        except Exception:
            pass
        return False

    # (1) Try industry + market cap
    norm_industry = _normalize_industry_for_fmp(industry)
    candidates = []
    screener_restricted = False
    if norm_industry:
        params = {**params_base, "industry": norm_industry}
        try:
            r = requests.get(base_url, params=params, timeout=15)
            if _is_restricted_response(r):
                screener_restricted = True
            elif r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) >= 3:
                    candidates = [x for x in data if isinstance(x, dict) and (x.get("symbol") or x.get("Symbol")) != ticker]
                    fallback_used = "industry"
        except Exception as e:
            print(f"[Peers FMP] industry request error: {e}")
        if not candidates:
            used_sector_fallback = True

    # (2) If few or none (and screener not restricted), try sector only
    if not screener_restricted and len(candidates) < 3 and sector:
        params = {**params_base, "sector": sector.strip()}
        try:
            r = requests.get(base_url, params=params, timeout=15)
            if _is_restricted_response(r):
                screener_restricted = True
            elif r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    by_symbol = {str(x.get("symbol") or x.get("Symbol", "")).upper(): x for x in data if isinstance(x, dict)}
                    by_symbol.pop(ticker, None)
                    candidates = list(by_symbol.values())
                    if candidates:
                        fallback_used = "sector"
        except Exception as e:
            print(f"[Peers FMP] sector request error: {e}")

    # (3) If still 0 and not restricted, sector + wider cap
    if not screener_restricted and len(candidates) == 0 and sector:
        params = {"apikey": FMP_API_KEY, "limit": limit, "sector": sector.strip()}
        if cap_min_wide is not None:
            params["marketCapMoreThan"] = cap_min_wide
        if cap_max_wide is not None:
            params["marketCapLowerThan"] = cap_max_wide
        if PEERS_COUNTRY:
            params["country"] = PEERS_COUNTRY
        if PEERS_EXCHANGE:
            params["exchange"] = PEERS_EXCHANGE
        try:
            r = requests.get(base_url, params=params, timeout=15)
            if _is_restricted_response(r):
                screener_restricted = True
            elif r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    by_symbol = {str(x.get("symbol") or x.get("Symbol", "")).upper(): x for x in data if isinstance(x, dict)}
                    by_symbol.pop(ticker, None)
                    candidates = list(by_symbol.values())
                    fallback_used = "sector_wide_cap"
        except Exception as e:
            print(f"[Peers FMP] sector_wide_cap request error: {e}")

    # (4) If still no candidates (e.g. screener 402/restricted), try stock-peers endpoint (often on free tier)
    if len(candidates) == 0:
        stock_peers = _fetch_stock_peers_fmp(ticker)
        if stock_peers:
            candidates = stock_peers
            fallback_used = "stock_peers"
            if screener_restricted:
                print(f"[Peers FMP] Company-screener restricted (402); using stock-peers for {ticker}.")

    # Normalize candidate dicts to have 'symbol'
    out = []
    for c in candidates[:limit]:
        sym = (c.get("symbol") or c.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "sector": c.get("sector") or c.get("Sector"),
            "industry": c.get("industry") or c.get("Industry"),
            "marketCap": c.get("marketCap") or c.get("market_cap"),
        })
    if out:
        _save_peers_candidates_cache(ticker, out, fallback_used, used_sector_fallback)
    return out, fallback_used, used_sector_fallback


def _get_full_description_for_matching(ticker: str) -> str:
    """
    Get the fullest available description for a ticker for similarity matching.
    Uses profile description (full, never truncated) and, if longer, yfinance longBusinessSummary.
    """
    desc = ""
    profile = get_company_profile(ticker)
    if profile and isinstance(profile.get("description"), str):
        desc = profile["description"].strip()
    try:
        stock = yf.Ticker(ticker)
        info = getattr(stock, "info", None) or {}
        yf_summary = info.get("longBusinessSummary") or info.get("description") or ""
        if isinstance(yf_summary, str) and len(yf_summary.strip()) > len(desc):
            desc = yf_summary.strip()
    except Exception:
        pass
    return desc


def rank_peers_by_description_similarity(
    focus_description: Optional[str],
    candidate_symbols: List[str],
    top_n: int = 5,
    focus_ticker: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Rank candidate symbols by TF-IDF cosine similarity to focus description.
    Uses full description text only (no truncation). If focus_ticker is provided and
    focus_description is short, will try to get a longer description for the focus company.
    Returns list of (symbol, score) tuples, highest first, length up to top_n.
    """
    if not focus_description or not isinstance(focus_description, str):
        focus_description = ""
    focus_description = focus_description.strip()
    if focus_ticker and len(focus_description) < 500:
        full_focus = _get_full_description_for_matching(focus_ticker)
        if len(full_focus) > len(focus_description):
            focus_description = full_focus
    if not focus_description:
        return [(s, 0.0) for s in candidate_symbols[:top_n]]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return [(s, 0.0) for s in candidate_symbols[:top_n]]

    def clean(text: str) -> str:
        if not text:
            return ""
        t = text.lower().strip()
        for w in PEERS_DESCRIPTION_STOP_WORDS:
            t = t.replace(f" {w} ", " ")
        return t

    focus_clean = clean(focus_description)
    if not focus_clean:
        return [(s, 0.0) for s in candidate_symbols[:top_n]]

    texts = [focus_clean]
    symbol_to_idx = {}
    for sym in candidate_symbols:
        profile = get_company_profile(sym)
        desc = (profile or {}).get("description") or ""
        if isinstance(desc, str) and desc.strip():
            desc = desc.strip()
        else:
            try:
                stock = yf.Ticker(sym)
                info = getattr(stock, "info", None) or {}
                desc = (info.get("longBusinessSummary") or info.get("description") or "") or ""
                if isinstance(desc, str):
                    desc = desc.strip()
            except Exception:
                desc = ""
        texts.append(clean(desc))
        symbol_to_idx[len(texts) - 1] = sym

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(texts)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()
        scored = [(symbol_to_idx[i + 1], float(sims[i])) for i in range(len(candidate_symbols)) if (i + 1) in symbol_to_idx]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]
    except Exception as e:
        print(f"[Peers] description similarity error: {e}")
        return [(s, 0.0) for s in candidate_symbols[:top_n]]


def _get_peer_overrides(focus_ticker: str) -> Tuple[List[str], List[str]]:
    """Return (excluded_peers, always_include_peers) from DB."""
    focus_ticker = focus_ticker.upper().strip()
    try:
        db = _get_db_session()
        from models import PeerOverride
        rows = db.query(PeerOverride).filter(PeerOverride.focus_ticker == focus_ticker).all()
        excluded = [r.peer_ticker.upper().strip() for r in rows if r.is_excluded == 1]
        always = [r.peer_ticker.upper().strip() for r in rows if r.is_excluded == 0]
        db.close()
        return excluded, always
    except Exception as e:
        print(f"[Peers] overrides load error: {e}")
        try:
            db.close()
        except Exception:
            pass
        return [], []


def get_competitors(
    ticker: str,
    sort_by: Literal["industry_size", "description"],
    max_peers: int = 5,
) -> Dict:
    """
    Get competitor peers for a ticker. Applies user overrides (exclude / always include).
    Returns dict with keys: peers (list of peer dicts), fallback_used (str), used_sector_fallback (bool).
    Each peer dict: ticker, name, sector, industry, market_cap, pe_ratio, revenue_ttm, description_match_score (optional).
    """
    ticker = ticker.upper().strip()
    profile = get_company_profile(ticker)
    summary = None
    try:
        summary = get_ticker_summary(ticker)
    except Exception:
        pass
    industry = (profile or {}).get("industry")
    sector = (profile or {}).get("sector")
    market_cap = (summary or {}).get("market_cap") if summary else (profile or {}).get("market_cap")

    candidates, fallback_used, used_sector_fallback = fetch_peer_candidates_fmp(
        ticker, industry, sector, market_cap, limit=30
    )
    candidate_symbols = [c.get("symbol") for c in candidates if c.get("symbol")]

    description_scores = {}  # symbol -> score 0-100
    if sort_by == "description":
        # Use full description for matching (never truncated). Merge profile + yfinance for longest available.
        focus_desc = _get_full_description_for_matching(ticker)
        if focus_desc:
            ranked = rank_peers_by_description_similarity(focus_desc, candidate_symbols, top_n=max_peers, focus_ticker=ticker)
            ordered_symbols = [s for s, _ in ranked]
            description_scores = {s: round(score * 100) for s, score in ranked}
            for s in candidate_symbols:
                if s not in ordered_symbols and len(ordered_symbols) < max_peers:
                    ordered_symbols.append(s)
        else:
            ordered_symbols = candidate_symbols[:max_peers]
    else:
        ordered_symbols = candidate_symbols[:max_peers]

    excluded, always_include = _get_peer_overrides(ticker)
    peer_set = [s for s in ordered_symbols if s not in excluded]
    for a in always_include:
        if a not in peer_set and len(peer_set) < MAX_PEERS_DISPLAY:
            peer_set.append(a)
    peer_set = peer_set[:MAX_PEERS_DISPLAY]

    peers_out = []
    for sym in peer_set:
        p_profile = get_company_profile(sym)
        p_fund = get_fundamentals_ratios(sym)
        screener_row = next((c for c in candidates if (c.get("symbol") or "").upper() == sym), {})
        market_cap_val = screener_row.get("marketCap")
        if market_cap_val is None and p_profile:
            pass  # could get from summary if needed
        try:
            s_sum = get_ticker_summary(sym)
            if s_sum and market_cap_val is None:
                market_cap_val = s_sum.get("market_cap")
        except Exception:
            pass
        pe_val = None
        if p_fund:
            pe_val = p_fund.get("pe_ratio") or p_fund.get("price_earnings_ratio")
        if pe_val is None and p_profile:
            pass
        ti = None
        try:
            ti = _fetch_and_cache_ticker_info(sym)
            if pe_val is None and ti:
                pe_val = ti.get("pe_ratio")
            if market_cap_val is None and ti:
                market_cap_val = ti.get("market_cap")
        except Exception:
            pass
        rev = (p_fund or {}).get("revenue_ttm")
        name = (p_profile or {}).get("companyName") or (p_profile or {}).get("company_name") or None
        if not name and p_profile and isinstance(p_profile.get("description"), str):
            name = (p_profile["description"][:50] + "…") if len(p_profile["description"]) > 50 else p_profile["description"]
        if not name and ti:
            name = ti.get("longName") or ti.get("shortName")
        if not name:
            name = sym
        desc_score = description_scores.get(sym)
        peers_out.append({
            "ticker": sym,
            "name": name[:60] if name else sym,
            "sector": (p_profile or {}).get("sector") or (screener_row or {}).get("sector") or "—",
            "industry": (p_profile or {}).get("industry") or (screener_row or {}).get("industry") or "—",
            "market_cap": market_cap_val,
            "pe_ratio": pe_val,
            "revenue_ttm": rev,
            "description_match_score": desc_score,
        })
    return {"peers": peers_out, "fallback_used": fallback_used, "used_sector_fallback": used_sector_fallback}


def clear_peers_cache(ticker: Optional[str] = None):
    """Clear peers candidate cache for a ticker (or all if ticker is None)."""
    if ticker:
        path = _peers_candidates_cache_path(ticker.upper().strip())
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    else:
        for p in CACHE_DIR.glob("peers_candidates_*.json"):
            try:
                p.unlink()
            except OSError:
                pass

