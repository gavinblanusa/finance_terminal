"""
Database models for Gavin Financial Terminal.

This module defines SQLAlchemy models for:
- Trades: Individual buy/sell transactions
- Watchlist: Stocks to monitor with price alerts
- IPO_Registry: IPO tracking and vintage analysis
- ValuationHistory: Cached earnings and revenue data
- CompanyProfile: Cached company profile (sector, industry, description, etc.)
- CompanyFundamentals: Cached fundamentals snapshot (revenue, margins, ratios)
"""

from sqlalchemy import Column, Integer, String, Date, Numeric, Enum, DateTime, BigInteger, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class TradeType(enum.Enum):
    """Enumeration for trade types."""
    BUY = "Buy"
    SELL = "Sell"


class Trades(Base):
    """Model for tracking individual trades."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)  # Increased to 20 for longer symbols
    trade_date = Column(Date, nullable=False, index=True)
    shares = Column(Numeric(15, 4), nullable=False)
    price_per_share = Column(Numeric(15, 4), nullable=False)
    trade_type = Column(Enum(TradeType), nullable=False)

    def __repr__(self):
        return (f"<Trades(id={self.id}, ticker='{self.ticker}', "
                f"date={self.trade_date}, shares={self.shares}, "
                f"price={self.price_per_share}, type={self.trade_type.value})>")


class Watchlist(Base):
    """Model for tracking stocks on watchlist with price alerts."""
    __tablename__ = 'watchlist'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, unique=True, index=True)
    alert_price = Column(Numeric(15, 4), nullable=True)

    def __repr__(self):
        return (f"<Watchlist(id={self.id}, ticker='{self.ticker}', "
                f"alert_price={self.alert_price})>")


class IPO_Registry(Base):
    """Model for tracking IPOs and their vintage performance."""
    __tablename__ = 'ipo_registry'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, unique=True, index=True)
    company_name = Column(String(200), nullable=True)
    ipo_date = Column(Date, nullable=False, index=True)
    ipo_price = Column(Numeric(15, 4), nullable=True)
    exchange = Column(String(20), nullable=True)
    shares_offered = Column(Integer, nullable=True)
    market_cap_at_ipo = Column(Numeric(20, 2), nullable=True)
    is_following = Column(Integer, default=1)  # 1 = following, 0 = not following
    created_at = Column(Date, nullable=True)

    def __repr__(self):
        return (f"<IPO_Registry(id={self.id}, ticker='{self.ticker}', "
                f"company='{self.company_name}', ipo_date={self.ipo_date}, "
                f"ipo_price={self.ipo_price})>")


class ValuationHistory(Base):
    """
    Model for storing historical earnings and revenue data.
    
    Stores quarterly financial data to avoid repeated API calls.
    Data is fetched once and stored permanently, with only new quarters
    being added over time.
    """
    __tablename__ = 'valuation_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    quarter_date = Column(Date, nullable=False, index=True)
    
    # Earnings data
    eps = Column(Float, nullable=True)  # Earnings per share
    
    # Revenue data
    revenue = Column(BigInteger, nullable=True)  # Total revenue
    revenue_growth_yoy = Column(Float, nullable=True)  # YoY growth %
    revenue_growth_qoq = Column(Float, nullable=True)  # QoQ growth %
    
    # P/E data (pre-calculated)
    pe_ratio = Column(Float, nullable=True)
    ttm_eps = Column(Float, nullable=True)
    price_at_quarter = Column(Float, nullable=True)
    
    # Metadata
    data_source = Column(String(30), nullable=True)  # 'alpha_vantage', 'finnhub', 'yfinance'
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite unique constraint
    __table_args__ = (
        # Ensure we don't have duplicate ticker + quarter combinations
        {'sqlite_autoincrement': True},
    )

    def __repr__(self):
        return (f"<ValuationHistory(ticker='{self.ticker}', "
                f"quarter={self.quarter_date}, eps={self.eps}, "
                f"pe={self.pe_ratio})>")


class CompanyProfile(Base):
    """
    Cached company profile (sector, industry, description, web, employees, CEO).
    One row per ticker; read-through cache with 7-day TTL.
    """
    __tablename__ = 'company_profile'

    ticker = Column(String(20), primary_key=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    website = Column(String(500), nullable=True)
    full_time_employees = Column(Integer, nullable=True)
    ceo = Column(String(200), nullable=True)
    data_source = Column(String(30), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CompanyProfile(ticker='{self.ticker}', sector='{self.sector}')>"


class CompanyFundamentals(Base):
    """
    Cached fundamentals snapshot (revenue, margins, ratios).
    One row per ticker; read-through cache with 1-day TTL.
    """
    __tablename__ = 'company_fundamentals'

    ticker = Column(String(20), primary_key=True)
    revenue_ttm = Column(BigInteger, nullable=True)
    gross_margin = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    net_margin = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    roa = Column(Float, nullable=True)
    data_source = Column(String(30), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CompanyFundamentals(ticker='{self.ticker}')>"

