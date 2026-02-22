"""
Tax Engine - HIFO Logic & Tax Lot Tracking.

This module provides:
- Tax lot tracking and management
- HIFO (Highest-In-First-Out) cost basis calculation
- Capital gains classification (Short-Term vs Long-Term)
- Portfolio valuation with real-time prices (Yahoo Finance + Alpha Vantage fallback)
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time
import os
import requests
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv
from models import Trades, TradeType

# Load environment variables
load_dotenv()


def _safe_decimal(value, default: Decimal = Decimal('0')) -> Decimal:
    """
    Safely convert a value to Decimal, handling None/NaN/Inf/invalid values.
    
    Args:
        value: Value to convert (can be None, str, int, float, Decimal)
        default: Default value to return if conversion fails
        
    Returns:
        Valid Decimal value, or default if conversion fails
    """
    if value is None:
        return default
    try:
        # Handle case where value is already a Decimal
        if isinstance(value, Decimal):
            if value.is_nan() or value.is_infinite():
                return default
            return value
        
        # Convert to Decimal via string (safest conversion)
        d = Decimal(str(value))
        
        # Check for NaN or Infinity
        if d.is_nan() or d.is_infinite():
            return default
        
        return d
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        return default

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_API_KEY', '')

# Global price cache with expiration
_price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
CACHE_EXPIRY_MINUTES = 15  # Cache prices for 15 minutes

# Ticker symbol corrections (brokerage format → Yahoo format)
# Note: Some tickers may temporarily fail on Yahoo Finance even when valid.
# The Alpha Vantage fallback will attempt to fetch prices for failed tickers.
TICKER_CORRECTIONS = {
    'BRKB': 'BRK-B',
    'BRKA': 'BRK-A',
    'BRK.B': 'BRK-B',
    'BRK.A': 'BRK-A',
    'GOOG.L': 'GOOGL',
    # Common mutual fund / ETF corrections
    # Add any tickers that consistently fail here
}


def normalize_ticker(ticker: str) -> str:
    """Convert brokerage ticker format to Yahoo Finance format."""
    ticker = ticker.upper().strip()
    return TICKER_CORRECTIONS.get(ticker, ticker)


def fetch_price_alpha_vantage(ticker: str) -> Optional[Decimal]:
    """
    Fetch price from Alpha Vantage as a fallback.
    
    Uses the GLOBAL_QUOTE endpoint for real-time price.
    Free tier: 25 requests/day, so use sparingly as fallback only.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Price as Decimal, or None if unavailable
    """
    if not ALPHA_VANTAGE_API_KEY:
        print(f"Alpha Vantage: No API key configured")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Debug: print what we got back
        print(f"Alpha Vantage response for {ticker}: {list(data.keys())}")
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price_str = quote.get("05. price")
            if price_str:
                price = Decimal(str(price_str))
                if price > 0:
                    print(f"✓ Alpha Vantage: {ticker} = ${price}")
                    return price
            else:
                print(f"Alpha Vantage: No price in quote for {ticker}")
        elif "Global Quote" in data:
            print(f"Alpha Vantage: Empty quote for {ticker} (ticker may be invalid)")
        
        # Check for API limit message
        if "Note" in data:
            print(f"Alpha Vantage rate limit: {data['Note'][:80]}...")
        elif "Error Message" in data:
            print(f"Alpha Vantage error for {ticker}: {data['Error Message'][:80]}...")
        elif "Information" in data:
            print(f"Alpha Vantage info: {data['Information'][:80]}...")
            
    except Exception as e:
        print(f"Alpha Vantage exception for {ticker}: {e}")
    
    return None


def get_cached_price(ticker: str) -> Optional[Decimal]:
    """Get price from cache if not expired."""
    if ticker in _price_cache:
        price, cached_time = _price_cache[ticker]
        if datetime.now() - cached_time < timedelta(minutes=CACHE_EXPIRY_MINUTES):
            return price
    return None


def set_cached_price(ticker: str, price: Decimal):
    """Store price in cache with timestamp."""
    _price_cache[ticker] = (price, datetime.now())


def fetch_single_price(ticker: str) -> Optional[Decimal]:
    """
    Fetch price for a single ticker using multiple methods.
    
    Tries several approaches in order of reliability.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Price as Decimal, or None if all methods fail
    """
    # OpenBB first (when available)
    try:
        from openbb_adapter import fetch_quote_openbb
        price = fetch_quote_openbb(ticker)
        if price is not None and price > 0:
            set_cached_price(ticker, price)
            return price
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenBB] fetch_single_price fallback for {ticker}: {e}")

    # Normalize ticker for Yahoo Finance
    yahoo_ticker = normalize_ticker(ticker)
    
    # Method 1: Try fast_info (fastest, most reliable)
    try:
        stock = yf.Ticker(yahoo_ticker)
        fast_info = stock.fast_info
        
        # Try different price attributes
        price = None
        if hasattr(fast_info, 'last_price') and fast_info.last_price:
            price = fast_info.last_price
        elif hasattr(fast_info, 'previous_close') and fast_info.previous_close:
            price = fast_info.previous_close
        
        if price and price > 0:
            return Decimal(str(round(price, 2)))
    except Exception as e:
        print(f"Method 1 (fast_info) failed for {ticker}: {e}")
    
    # Method 2: Try history with 5d period (more reliable than 1d)
    try:
        time.sleep(0.3)  # Small delay between attempts
        stock = yf.Ticker(yahoo_ticker)
        hist = stock.history(period="5d")
        if not hist.empty and 'Close' in hist.columns:
            price = hist['Close'].dropna().iloc[-1]
            if price > 0:
                return Decimal(str(round(price, 2)))
    except Exception as e:
        print(f"Method 2 (history) failed for {ticker}: {e}")
    
    # Method 3: Try download function
    try:
        time.sleep(0.3)
        data = yf.download(yahoo_ticker, period="5d", progress=False, threads=False)
        if not data.empty and 'Close' in data.columns:
            price = data['Close'].dropna().iloc[-1]
            if price > 0:
                return Decimal(str(round(price, 2)))
    except Exception as e:
        print(f"Method 3 (download) failed for {ticker}: {e}")
    
    # Method 4: Polygon (Massive) fallback
    try:
        from api_clients import fetch_price_polygon, fetch_price_twelvedata, fetch_price_eodhd
        price = fetch_price_polygon(ticker)
        if price:
            return price
        # Method 5: Twelve Data fallback
        price = fetch_price_twelvedata(ticker)
        if price:
            return price
        # Method 6: EODHD fallback
        price = fetch_price_eodhd(ticker)
        if price:
            return price
    except ImportError:
        pass

    # Method 7: Alpha Vantage fallback (use original ticker, not Yahoo-normalized)
    if ALPHA_VANTAGE_API_KEY:
        print(f"Trying Alpha Vantage fallback for {ticker}...")
        price = fetch_price_alpha_vantage(ticker)
        if price:
            return price
    
    return None


def fetch_prices_batch(tickers: List[str]) -> Dict[str, Decimal]:
    """
    Fetch prices for multiple tickers with robust error handling.
    
    Uses caching and multiple fallback methods to ensure reliability.
    
    Args:
        tickers: List of ticker symbols (in original format)
        
    Returns:
        Dict mapping original ticker to price
    """
    prices = {}
    
    # Filter out tickers we already have cached
    tickers_to_fetch = []
    for ticker in tickers:
        cached = get_cached_price(ticker)
        if cached is not None:
            prices[ticker] = cached
        else:
            tickers_to_fetch.append(ticker)
    
    if not tickers_to_fetch:
        return prices
    
    # Create mapping of original ticker -> Yahoo ticker
    ticker_map = {t: normalize_ticker(t) for t in tickers_to_fetch}
    yahoo_tickers = list(set(ticker_map.values()))  # Unique Yahoo tickers
    
    # Try batch download first (most efficient)
    try:
        tickers_str = " ".join(yahoo_tickers)
        data = yf.download(tickers_str, period="5d", progress=False, threads=False)
        
        if not data.empty:
            # Handle single ticker vs multiple tickers (different DataFrame structure)
            if len(yahoo_tickers) == 1:
                yahoo_ticker = yahoo_tickers[0]
                if 'Close' in data.columns and len(data) > 0:
                    price_val = data['Close'].dropna().iloc[-1]
                    if price_val > 0:
                        price = Decimal(str(round(price_val, 2)))
                        # Map back to all original tickers that use this Yahoo ticker
                        for orig_ticker, yt in ticker_map.items():
                            if yt == yahoo_ticker:
                                prices[orig_ticker] = price
                                set_cached_price(orig_ticker, price)
            else:
                # Multiple tickers - data has multi-level columns
                if 'Close' in data.columns.get_level_values(0):
                    close_prices = data['Close']
                    for orig_ticker, yahoo_ticker in ticker_map.items():
                        if yahoo_ticker in close_prices.columns:
                            last_price = close_prices[yahoo_ticker].dropna()
                            if len(last_price) > 0:
                                price_val = last_price.iloc[-1]
                                if price_val > 0:
                                    price = Decimal(str(round(price_val, 2)))
                                    prices[orig_ticker] = price
                                    set_cached_price(orig_ticker, price)
                                
    except Exception as e:
        print(f"Batch download failed: {e}")
    
    # For any tickers that failed, try individual fetch with fallbacks
    for ticker in tickers_to_fetch:
        if ticker not in prices:
            price = fetch_single_price(ticker)
            if price is not None:
                prices[ticker] = price
                set_cached_price(ticker, price)
            else:
                # Last resort: set to 0 so it doesn't block the app
                print(f"Could not fetch price for {ticker}, using 0")
                prices[ticker] = Decimal('0')
    
    return prices


@dataclass
class TaxLot:
    """Represents an individual tax lot (a specific purchase of shares)."""
    trade_id: int
    ticker: str
    purchase_date: date
    shares: Decimal
    cost_basis: Decimal  # Price per share at purchase
    
    @property
    def total_cost(self) -> Decimal:
        """Total cost of this tax lot."""
        return self.shares * self.cost_basis
    
    @property
    def holding_days(self) -> int:
        """Number of days this lot has been held."""
        return (date.today() - self.purchase_date).days
    
    @property
    def is_long_term(self) -> bool:
        """True if held for more than 365 days."""
        return self.holding_days > 365
    
    @property
    def tax_status(self) -> str:
        """Return 'Long-Term' or 'Short-Term' based on holding period."""
        return "Long-Term" if self.is_long_term else "Short-Term"
    
    @property
    def days_until_long_term(self) -> int:
        """Days remaining until this lot becomes long-term. Returns 0 if already long-term."""
        if self.is_long_term:
            return 0
        return 366 - self.holding_days
    
    @property
    def is_near_long_term(self) -> bool:
        """True if within 30 days of becoming long-term."""
        return not self.is_long_term and self.days_until_long_term <= 30


@dataclass
class PositionSummary:
    """Summary of a position in a single ticker."""
    ticker: str
    total_shares: Decimal
    total_cost_basis: Decimal
    current_price: Decimal
    current_value: Decimal
    unrealized_gain: Decimal
    unrealized_gain_pct: Decimal
    tax_lots: List[TaxLot]
    
    # Tax breakdown
    short_term_shares: Decimal
    short_term_gain: Decimal
    long_term_shares: Decimal
    long_term_gain: Decimal
    
    # Alerts
    lots_near_long_term: List[TaxLot]


@dataclass
class PortfolioSummary:
    """Overall portfolio summary."""
    total_value: Decimal
    total_cost_basis: Decimal
    total_unrealized_gain: Decimal
    total_unrealized_gain_pct: Decimal
    
    # Tax breakdown
    short_term_gain: Decimal
    long_term_gain: Decimal
    
    # Position details
    positions: List[PositionSummary]
    
    # Alerts
    urgent_lots: List[TaxLot]  # Lots within 30 days of long-term


class TaxEngine:
    """
    Tax calculation engine using HIFO (Highest-In-First-Out) methodology.
    
    HIFO sells the highest cost basis shares first to minimize taxable gains.
    """
    
    def __init__(self, db_session: Session):
        """Initialize with database session."""
        self.db = db_session
        self._prices_loaded = False
    
    def _preload_prices(self):
        """Preload all prices in a single batch request."""
        if self._prices_loaded:
            return
        
        tickers = self.get_unique_tickers()
        if tickers:
            fetch_prices_batch(tickers)
        self._prices_loaded = True
    
    def get_current_price(self, ticker: str) -> Decimal:
        """
        Get current market price for a ticker.
        
        Uses cached prices when available, fetches in batch otherwise.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price as Decimal, or Decimal(0) if unavailable
        """
        # Check cache first
        cached = get_cached_price(ticker)
        if cached is not None:
            return cached
        
        # Fetch this ticker (and cache it)
        prices = fetch_prices_batch([ticker])
        return prices.get(ticker, Decimal('0'))
    
    def get_all_trades(self, ticker: Optional[str] = None) -> List[Trades]:
        """
        Fetch all trades from database, optionally filtered by ticker.
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            List of Trade records ordered by date
        """
        query = self.db.query(Trades).order_by(Trades.trade_date.asc())
        
        if ticker:
            query = query.filter(Trades.ticker == ticker)
        
        return query.all()
    
    def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers in the portfolio."""
        result = self.db.query(Trades.ticker).distinct().all()
        return [r[0] for r in result]
    
    def calculate_tax_lots(self, ticker: str) -> List[TaxLot]:
        """
        Calculate open tax lots for a ticker using FIFO for lot matching.
        
        This processes all buys and sells chronologically, matching sells
        to the earliest buys first (for lot tracking purposes).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of open TaxLot objects
        """
        trades = self.get_all_trades(ticker)
        
        # Track open lots (from buys)
        open_lots: List[TaxLot] = []
        
        for trade in trades:
            if trade.trade_type == TradeType.BUY:
                # Safely convert shares and price, skipping invalid trades
                shares = _safe_decimal(trade.shares)
                cost_basis = _safe_decimal(trade.price_per_share)
                
                # Skip trades with invalid or zero values
                if shares <= 0 or cost_basis <= 0:
                    print(f"Skipping invalid BUY trade for {ticker}: shares={trade.shares}, price={trade.price_per_share}")
                    continue
                
                # Add new tax lot
                lot = TaxLot(
                    trade_id=trade.id,
                    ticker=trade.ticker,
                    purchase_date=trade.trade_date,
                    shares=shares,
                    cost_basis=cost_basis
                )
                open_lots.append(lot)
            
            elif trade.trade_type == TradeType.SELL:
                # Safely convert shares to sell
                shares_to_sell = _safe_decimal(trade.shares)
                
                # Skip invalid sell trades
                if shares_to_sell <= 0:
                    print(f"Skipping invalid SELL trade for {ticker}: shares={trade.shares}")
                    continue
                
                # Sort by date (FIFO) for matching sells to buys
                open_lots.sort(key=lambda x: x.purchase_date)
                
                while shares_to_sell > 0 and open_lots:
                    lot = open_lots[0]
                    
                    # Safe comparison - both values are now guaranteed valid Decimals
                    if lot.shares <= shares_to_sell:
                        # Entire lot is sold
                        shares_to_sell -= lot.shares
                        open_lots.pop(0)
                    else:
                        # Partial lot sale
                        lot.shares -= shares_to_sell
                        shares_to_sell = Decimal('0')
        
        return open_lots
    
    def calculate_unrealized_gains_hifo(
        self, 
        ticker: str, 
        shares_to_sell: Optional[Decimal] = None
    ) -> Tuple[Decimal, List[TaxLot]]:
        """
        Calculate unrealized gains using HIFO methodology.
        
        HIFO: Sell highest cost basis lots first to minimize taxable gains.
        
        Args:
            ticker: Stock ticker symbol
            shares_to_sell: If provided, simulate selling this many shares.
                           If None, calculate for all shares.
                           
        Returns:
            Tuple of (unrealized_gain, lots_that_would_be_sold)
        """
        open_lots = self.calculate_tax_lots(ticker)
        current_price = self.get_current_price(ticker)
        
        if not open_lots or current_price == 0:
            return Decimal('0'), []
        
        # Sort by cost basis descending (HIFO - highest first)
        sorted_lots = sorted(open_lots, key=lambda x: x.cost_basis, reverse=True)
        
        if shares_to_sell is None:
            # Calculate unrealized gain for all shares
            total_gain = Decimal('0')
            for lot in sorted_lots:
                gain = (current_price - lot.cost_basis) * lot.shares
                total_gain += gain
            return total_gain, sorted_lots
        
        # Simulate selling specific number of shares
        remaining_to_sell = shares_to_sell
        total_gain = Decimal('0')
        lots_sold = []
        
        for lot in sorted_lots:
            if remaining_to_sell <= 0:
                break
            
            shares_from_lot = min(lot.shares, remaining_to_sell)
            gain = (current_price - lot.cost_basis) * shares_from_lot
            total_gain += gain
            remaining_to_sell -= shares_from_lot
            
            # Create a copy of the lot with the shares being sold
            sold_lot = TaxLot(
                trade_id=lot.trade_id,
                ticker=lot.ticker,
                purchase_date=lot.purchase_date,
                shares=shares_from_lot,
                cost_basis=lot.cost_basis
            )
            lots_sold.append(sold_lot)
        
        return total_gain, lots_sold
    
    def get_position_summary(self, ticker: str) -> Optional[PositionSummary]:
        """
        Get complete position summary for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PositionSummary object or None if no position
        """
        try:
            tax_lots = self.calculate_tax_lots(ticker)
            
            if not tax_lots:
                return None
            
            current_price = self.get_current_price(ticker)
            
            # Ensure current_price is a valid Decimal
            if current_price is None:
                current_price = Decimal('0')
            
            # Calculate totals with safe Decimal operations
            total_shares = sum((lot.shares for lot in tax_lots), Decimal('0'))
            total_cost_basis = sum((lot.total_cost for lot in tax_lots), Decimal('0'))
            current_value = total_shares * current_price
            unrealized_gain = current_value - total_cost_basis
            
            # Calculate percentage gain (handle zero cost basis)
            try:
                if total_cost_basis > 0:
                    unrealized_gain_pct = (unrealized_gain / total_cost_basis) * Decimal('100')
                else:
                    unrealized_gain_pct = Decimal('0')
            except Exception:
                unrealized_gain_pct = Decimal('0')
            
            # Tax breakdown by holding period
            short_term_lots = [lot for lot in tax_lots if not lot.is_long_term]
            long_term_lots = [lot for lot in tax_lots if lot.is_long_term]
            
            short_term_shares = sum((lot.shares for lot in short_term_lots), Decimal('0'))
            long_term_shares = sum((lot.shares for lot in long_term_lots), Decimal('0'))
            
            # Calculate gains by tax status (with safe operations)
            short_term_gain = sum(
                ((current_price - lot.cost_basis) * lot.shares 
                 for lot in short_term_lots),
                Decimal('0')
            )
            long_term_gain = sum(
                ((current_price - lot.cost_basis) * lot.shares 
                 for lot in long_term_lots),
                Decimal('0')
            )
            
            # Find lots near long-term threshold
            lots_near_long_term = [lot for lot in tax_lots if lot.is_near_long_term]
            
            return PositionSummary(
                ticker=ticker,
                total_shares=total_shares,
                total_cost_basis=total_cost_basis,
                current_price=current_price,
                current_value=current_value,
                unrealized_gain=unrealized_gain,
                unrealized_gain_pct=unrealized_gain_pct,
                tax_lots=tax_lots,
                short_term_shares=short_term_shares,
                short_term_gain=short_term_gain,
                long_term_shares=long_term_shares,
                long_term_gain=long_term_gain,
                lots_near_long_term=lots_near_long_term
            )
        except Exception as e:
            import traceback
            print(f"Error calculating position summary for {ticker}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return a minimal valid position with zeros
            try:
                tax_lots = self.calculate_tax_lots(ticker) or []
                total_shares = sum((lot.shares for lot in tax_lots), Decimal('0'))
                total_cost_basis = sum((lot.total_cost for lot in tax_lots), Decimal('0'))
            except Exception:
                tax_lots = []
                total_shares = Decimal('0')
                total_cost_basis = Decimal('0')
            
            return PositionSummary(
                ticker=ticker,
                total_shares=total_shares,
                total_cost_basis=total_cost_basis,
                current_price=Decimal('0'),
                current_value=Decimal('0'),
                unrealized_gain=Decimal('0'),
                unrealized_gain_pct=Decimal('0'),
                tax_lots=tax_lots,
                short_term_shares=Decimal('0'),
                short_term_gain=Decimal('0'),
                long_term_shares=Decimal('0'),
                long_term_gain=Decimal('0'),
                lots_near_long_term=[]
            )
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Get complete portfolio summary across all positions.
        
        Returns:
            PortfolioSummary object
        """
        tickers = self.get_unique_tickers()
        
        # Preload all prices in a single batch request
        if tickers:
            fetch_prices_batch(tickers)
        
        positions: List[PositionSummary] = []
        urgent_lots: List[TaxLot] = []
        
        for ticker in tickers:
            try:
                position = self.get_position_summary(ticker)
                if position and position.total_shares > 0:
                    positions.append(position)
                    if position.lots_near_long_term:
                        urgent_lots.extend(position.lots_near_long_term)
            except Exception as e:
                print(f"Error getting position for {ticker}: {e}")
                continue
        
        # Calculate portfolio totals with safe Decimal operations
        total_value = sum((p.current_value for p in positions), Decimal('0'))
        total_cost_basis = sum((p.total_cost_basis for p in positions), Decimal('0'))
        total_unrealized_gain = sum((p.unrealized_gain for p in positions), Decimal('0'))
        
        try:
            if total_cost_basis > 0:
                total_unrealized_gain_pct = (total_unrealized_gain / total_cost_basis) * Decimal('100')
            else:
                total_unrealized_gain_pct = Decimal('0')
        except Exception:
            total_unrealized_gain_pct = Decimal('0')
        
        # Tax breakdown with safe sums
        short_term_gain = sum((p.short_term_gain for p in positions), Decimal('0'))
        long_term_gain = sum((p.long_term_gain for p in positions), Decimal('0'))
        
        return PortfolioSummary(
            total_value=total_value,
            total_cost_basis=total_cost_basis,
            total_unrealized_gain=total_unrealized_gain,
            total_unrealized_gain_pct=total_unrealized_gain_pct,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            positions=positions,
            urgent_lots=urgent_lots
        )
    
    def get_allocation_data(self) -> Dict[str, float]:
        """
        Get portfolio allocation by ticker for visualization.
        
        Returns:
            Dict mapping ticker to current value percentage
        """
        summary = self.get_portfolio_summary()
        
        if summary.total_value == 0:
            return {}
        
        allocation = {}
        for position in summary.positions:
            pct = float((position.current_value / summary.total_value) * 100)
            allocation[position.ticker] = pct
        
        return allocation
    
    def get_gains_by_tax_status(self) -> Dict[str, float]:
        """
        Get gains breakdown by tax status for visualization.
        
        Returns:
            Dict with 'Short-Term' and 'Long-Term' gains
        """
        summary = self.get_portfolio_summary()
        
        return {
            'Short-Term': float(summary.short_term_gain),
            'Long-Term': float(summary.long_term_gain)
        }


def parse_csv_trades(csv_data, column_mapping: Dict[str, str]) -> List[Dict]:
    """
    Parse CSV data into trade records.
    
    Args:
        csv_data: Pandas DataFrame from uploaded CSV
        column_mapping: Dict mapping our fields to CSV column names
            Expected keys: 'ticker', 'date', 'quantity', 'price', 'action'
            
    Returns:
        List of dicts with trade data ready for database insertion
    """
    import pandas as pd
    from datetime import datetime
    
    # Define action type mappings
    BUY_ACTIONS = [
        'buy', 'b', 'bought', 'purchase',
        'buy to open',
        'reinvest shares',
        'reinvest dividend',
        'qual div reinvest',
        'pr yr div reinvest',
        'long term cap gain reinvest',
    ]
    
    SELL_ACTIONS = [
        'sell', 's', 'sold', 'sale',
        'sell to close',
    ]
    
    # Actions to skip (cash dividends, adjustments, etc.)
    SKIP_ACTIONS = [
        'cash dividend',
        'qualified dividend',
        'non-qualified div',
        'pr yr cash div',
        'special qual div',
        'foreign tax reclaim adj',
        'mandatory reorg exc',
    ]
    
    trades = []
    skipped_actions = set()
    
    for _, row in csv_data.iterrows():
        try:
            # Parse action first to determine if we should process this row
            action = str(row[column_mapping['action']]).strip().lower()
            
            # Check if this is a buy action
            if action in BUY_ACTIONS:
                trade_type = TradeType.BUY
            elif action in SELL_ACTIONS:
                trade_type = TradeType.SELL
            elif action in SKIP_ACTIONS:
                skipped_actions.add(action)
                continue  # Skip cash dividends and adjustments
            else:
                skipped_actions.add(action)
                continue  # Skip unknown action types
            
            # Get values using column mapping
            ticker = str(row[column_mapping['ticker']]).upper().strip()
            
            # Skip if ticker is empty or invalid
            if not ticker or ticker in ['NAN', 'NONE', '']:
                continue
            
            # Parse date
            date_val = row[column_mapping['date']]
            if isinstance(date_val, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%y']:
                    try:
                        trade_date = datetime.strptime(date_val, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    trade_date = pd.to_datetime(date_val).date()
            else:
                trade_date = pd.to_datetime(date_val).date()
            
            # Parse quantity and price (handle potential negative values)
            quantity_str = str(row[column_mapping['quantity']]).replace(',', '').replace('$', '')
            price_str = str(row[column_mapping['price']]).replace(',', '').replace('$', '')
            
            # Use safe conversion and take absolute value
            quantity = abs(_safe_decimal(quantity_str))
            price = abs(_safe_decimal(price_str))
            
            # Skip if quantity or price is zero or invalid
            if quantity <= 0 or price <= 0:
                continue
            
            trades.append({
                'ticker': ticker,
                'trade_date': trade_date,
                'shares': quantity,
                'price_per_share': price,
                'trade_type': trade_type
            })
            
        except Exception as e:
            print(f"Error parsing row: {row}, Error: {e}")
            continue
    
    # Log skipped actions for debugging
    if skipped_actions:
        print(f"Skipped actions: {skipped_actions}")
    
    return trades


def check_duplicate_trade(
    db_session: Session,
    ticker: str,
    trade_date: date,
    shares: Decimal,
    price_per_share: Decimal,
    trade_type: TradeType
) -> bool:
    """
    Check if a trade already exists in the database.
    
    Args:
        db_session: SQLAlchemy session
        ticker, trade_date, shares, price_per_share, trade_type: Trade details
        
    Returns:
        True if duplicate exists, False otherwise
    """
    existing = db_session.query(Trades).filter(
        Trades.ticker == ticker,
        Trades.trade_date == trade_date,
        Trades.shares == shares,
        Trades.price_per_share == price_per_share,
        Trades.trade_type == trade_type
    ).first()
    
    return existing is not None


def import_trades_from_csv(
    db_session: Session, 
    csv_data, 
    column_mapping: Dict[str, str],
    skip_duplicates: bool = True
) -> Tuple[int, int, int]:
    """
    Import trades from CSV into database with duplicate detection.
    
    Args:
        db_session: SQLAlchemy session
        csv_data: Pandas DataFrame
        column_mapping: Column mapping dict
        skip_duplicates: If True, skip trades that already exist
        
    Returns:
        Tuple of (successful_imports, failed_imports, skipped_duplicates)
    """
    trades = parse_csv_trades(csv_data, column_mapping)
    
    successful = 0
    failed = 0
    duplicates = 0
    
    for trade_data in trades:
        try:
            # Check for duplicates
            if skip_duplicates and check_duplicate_trade(
                db_session,
                trade_data['ticker'],
                trade_data['trade_date'],
                trade_data['shares'],
                trade_data['price_per_share'],
                trade_data['trade_type']
            ):
                duplicates += 1
                continue
            
            trade = Trades(
                ticker=trade_data['ticker'],
                trade_date=trade_data['trade_date'],
                shares=trade_data['shares'],
                price_per_share=trade_data['price_per_share'],
                trade_type=trade_data['trade_type']
            )
            db_session.add(trade)
            successful += 1
        except Exception as e:
            print(f"Error importing trade: {e}")
            failed += 1
    
    db_session.commit()
    
    return successful, failed, duplicates

