"""
Gavin Financial Terminal - Main Streamlit Application.

A personal financial intelligence platform with portfolio tracking,
tax optimization, market analysis, and IPO vintage tracking.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import decimal
from decimal import Decimal
from sqlalchemy.orm import Session
from db import get_db_session, init_db
from models import Trades, TradeType, Watchlist, IPO_Registry
from tax_engine import TaxEngine, import_trades_from_csv, fetch_prices_batch
from ipo_service import (
    fetch_ipo_calendar,
    get_vintage_performance,
    get_ipo_price_history,
    check_vintage_anniversaries,
    clear_ipo_cache,
    IPOEntry,
    VintagePerformance
)
from market_data import (
    fetch_ohlcv,
    calculate_signals,
    calculate_tradingview_signals,
    get_ticker_summary,
    get_current_signal,
    clear_cache,
    get_valuation_chart_data,
    get_company_profile,
    get_fundamentals_ratios,
    fetch_company_news,
    fetch_insider_transactions,
    save_tv_signals_to_cache,
    load_tv_signals_from_cache,
    check_tv_signals_in_cache,
    get_competitors,
    clear_peers_cache,
)
from edgar_service import get_partnership_events, refresh_edgar_data
from plotly_chart_rescale import render_plotly_chart_with_y_rescale
from streamlit_lightweight_charts import renderLightweightCharts
from chart_utils import df_to_technical_chart_data, build_technical_chart_config
from thirteenf_config import THIRTEENF_INSTITUTIONS
from thirteenf_service import (
    get_13f_filings_for_institution,
    get_13f_holdings,
    get_13f_holdings_by_quarter,
    get_13f_compare,
    get_holders_by_cusip,
    get_overlap_holdings,
)
from pathlib import Path
from typing import Optional

# Project root (caches at Invest/ root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default on error."""
    try:
        if value is None:
            return default
        # Handle special Decimal cases
        if isinstance(value, Decimal):
            if value.is_nan() or value.is_infinite():
                return default
        return float(value)
    except Exception:
        return default


@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_portfolio_data():
    """
    Fetch portfolio summary with caching to reduce API calls.
    
    Returns cached data for 15 minutes to avoid Yahoo Finance rate limits.
    """
    try:
        db = get_db_session()
        engine = TaxEngine(db)
        summary = engine.get_portfolio_summary()
        db.close()
        
        # Convert to serializable format for caching (with safe conversion)
        return {
            'total_value': safe_float(summary.total_value),
            'total_cost_basis': safe_float(summary.total_cost_basis),
            'total_unrealized_gain': safe_float(summary.total_unrealized_gain),
            'total_unrealized_gain_pct': safe_float(summary.total_unrealized_gain_pct),
            'short_term_gain': safe_float(summary.short_term_gain),
            'long_term_gain': safe_float(summary.long_term_gain),
            'positions': [
                {
                    'ticker': p.ticker,
                    'total_shares': safe_float(p.total_shares),
                    'total_cost_basis': safe_float(p.total_cost_basis),
                    'current_price': safe_float(p.current_price),
                    'current_value': safe_float(p.current_value),
                    'unrealized_gain': safe_float(p.unrealized_gain),
                    'unrealized_gain_pct': safe_float(p.unrealized_gain_pct),
                    'short_term_shares': safe_float(p.short_term_shares),
                    'short_term_gain': safe_float(p.short_term_gain),
                    'long_term_shares': safe_float(p.long_term_shares),
                    'long_term_gain': safe_float(p.long_term_gain),
                    'tax_lots': [
                        {
                            'trade_id': lot.trade_id,
                            'ticker': lot.ticker,
                            'purchase_date': lot.purchase_date.isoformat(),
                            'shares': safe_float(lot.shares),
                            'cost_basis': safe_float(lot.cost_basis),
                            'holding_days': lot.holding_days,
                            'is_long_term': lot.is_long_term,
                            'tax_status': lot.tax_status,
                            'days_until_long_term': lot.days_until_long_term,
                            'is_near_long_term': lot.is_near_long_term,
                        }
                        for lot in p.tax_lots
                    ],
                    'lots_near_long_term': [
                        {
                            'ticker': lot.ticker,
                            'shares': safe_float(lot.shares),
                            'purchase_date': lot.purchase_date.isoformat(),
                            'cost_basis': safe_float(lot.cost_basis),
                            'days_until_long_term': lot.days_until_long_term,
                        }
                        for lot in p.lots_near_long_term
                    ]
                }
                for p in summary.positions
            ],
            'urgent_lots': [
                {
                    'ticker': lot.ticker,
                    'shares': safe_float(lot.shares),
                    'purchase_date': lot.purchase_date.isoformat(),
                    'cost_basis': safe_float(lot.cost_basis),
                    'days_until_long_term': lot.days_until_long_term,
                }
                for lot in summary.urgent_lots
            ]
        }
    except Exception as e:
        import traceback
        print(f"Error fetching portfolio data: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


# -----------------------------------------------------------------------------
# Cached data loaders (st.cache_data) to avoid redundant work on reruns.
# Only idempotent, serializable-return functions; TTLs keep data reasonably fresh.
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _cached_get_ticker_summary(ticker: str, alert_price: Optional[float] = None):
    return get_ticker_summary(ticker, alert_price=alert_price)


@st.cache_data(ttl=600)
def _cached_get_company_profile(ticker: str):
    return get_company_profile(ticker)


@st.cache_data(ttl=600)
def _cached_get_valuation_chart_data(ticker: str, years: int, skip_db: bool):
    return get_valuation_chart_data(ticker, years, skip_db=skip_db)


@st.cache_data(ttl=600)
def _cached_get_fundamentals_ratios(ticker: str):
    return get_fundamentals_ratios(ticker)


@st.cache_data(ttl=600)
def _cached_fetch_company_news(ticker: str, limit: int):
    return fetch_company_news(ticker, limit)


@st.cache_data(ttl=600)
def _cached_fetch_insider_transactions(ticker: str):
    return fetch_insider_transactions(ticker)


@st.cache_data(ttl=1800)
def _cached_fetch_ipo_calendar(days_ahead: int):
    return fetch_ipo_calendar(days_ahead=days_ahead)


@st.cache_data(ttl=900)
def _cached_get_partnership_events(limit: int):
    return get_partnership_events(limit=limit)


@st.cache_data(ttl=900)
def _cached_get_13f_filings_for_institution(cik: str):
    return get_13f_filings_for_institution(cik)


@st.cache_data(ttl=900)
def _cached_get_13f_holdings_by_quarter(cik: str, year: int, quarter: int):
    return get_13f_holdings_by_quarter(cik, year, quarter)


@st.cache_data(ttl=900)
def _cached_get_13f_compare(cik: str, accession_a: str, accession_b: str):
    return get_13f_compare(cik, accession_a, accession_b)


@st.cache_data(ttl=900)
def _cached_get_holders_by_cusip(cusip: str, institution_ciks: tuple, year: int, quarter: int):
    return get_holders_by_cusip(cusip, list(institution_ciks), year, quarter)


@st.cache_data(ttl=900)
def _cached_get_overlap_holdings(cik_list: tuple, year: int, quarter: int):
    return get_overlap_holdings(list(cik_list), year, quarter)


# Page configuration
st.set_page_config(
    page_title="Gavin Financial Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stSidebar {
        background-color: #1E1E1E;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    .stDateInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stNumberInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .urgent-alert {
        background-color: #FF4B4B;
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_database():
    """Initialize database tables if they don't exist."""
    try:
        init_db()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False


def format_currency(value) -> str:
    """Format a value as currency."""
    if value is None:
        return "$0.00"
    val = float(value)
    if val >= 0:
        return f"${val:,.2f}"
    return f"-${abs(val):,.2f}"


def format_percentage(value) -> str:
    """Format a value as percentage."""
    if value is None:
        return "0.00%"
    return f"{float(value):+.2f}%"


def dashboard_page():
    """Display the main dashboard page with portfolio overview."""
    st.title("📊 Dashboard")
    
    # Add refresh button
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Prices"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    try:
        # Use cached portfolio data to avoid rate limits
        summary = get_portfolio_data()
        
        if summary is None:
            st.error("Error loading portfolio data. Please try again.")
            return
        
        # Key metrics row
        st.subheader("Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Positions",
                len(summary['positions']),
                help="Number of unique stocks held"
            )
        
        with col2:
            st.metric(
                "Portfolio Value",
                format_currency(summary['total_value']),
                help="Current market value of all holdings"
            )
        
        with col3:
            delta_color = "normal" if summary['total_unrealized_gain'] >= 0 else "inverse"
            st.metric(
                "Unrealized Gain/Loss",
                format_currency(summary['total_unrealized_gain']),
                format_percentage(summary['total_unrealized_gain_pct']),
                delta_color=delta_color
            )
        
        with col4:
            st.metric(
                "Cost Basis",
                format_currency(summary['total_cost_basis']),
                help="Total amount invested"
            )
        
        st.markdown("---")
        
        # Tax summary row
        st.subheader("Tax Liability Summary")
        tax_col1, tax_col2, tax_col3 = st.columns(3)
        
        with tax_col1:
            st.metric(
                "Short-Term Gains",
                format_currency(summary['short_term_gain']),
                help="Gains on positions held ≤ 365 days (taxed as ordinary income)"
            )
        
        with tax_col2:
            st.metric(
                "Long-Term Gains",
                format_currency(summary['long_term_gain']),
                help="Gains on positions held > 365 days (preferential tax rates)"
            )
        
        with tax_col3:
            total_gain = summary['short_term_gain'] + summary['long_term_gain']
            st.metric(
                "Total Unrealized Gains",
                format_currency(total_gain),
                help="Combined short-term and long-term gains"
            )
        
        # Urgent alerts
        if summary['urgent_lots']:
            st.markdown("---")
            st.subheader("⚠️ Urgent Tax Alerts")
            st.warning(f"**{len(summary['urgent_lots'])} tax lot(s) approaching long-term status!**")
            
            for lot in summary['urgent_lots']:
                st.markdown(f"""
                🔔 **{lot['ticker']}**: {lot['shares']:.4f} shares purchased on {lot['purchase_date']} 
                will become long-term in **{lot['days_until_long_term']} days** 
                (Cost basis: {format_currency(lot['cost_basis'])})
                """)
        
        st.markdown("---")
        
        # Visualizations
        if summary['positions']:
            st.subheader("Portfolio Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Pie chart - Portfolio allocation
                allocation_data = {p['ticker']: p['current_value'] for p in summary['positions']}
                
                fig_pie = px.pie(
                    values=list(allocation_data.values()),
                    names=list(allocation_data.keys()),
                    title="Portfolio Allocation by Ticker",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=16
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_col2:
                # Bar chart - Gains by tax status
                gains_data = {
                    'Tax Status': ['Short-Term', 'Long-Term'],
                    'Gain/Loss': [summary['short_term_gain'], summary['long_term_gain']]
                }
                
                colors = ['#FF6B6B' if g < 0 else '#4ECDC4' for g in gains_data['Gain/Loss']]
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=gains_data['Tax Status'],
                        y=gains_data['Gain/Loss'],
                        marker_color=colors,
                        text=[format_currency(g) for g in gains_data['Gain/Loss']],
                        textposition='outside'
                    )
                ])
                fig_bar.update_layout(
                    title="Gains by Tax Status",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=16,
                    yaxis_title="Gain/Loss ($)",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Positions table
            st.markdown("---")
            st.subheader("Position Details")
            
            positions_data = []
            for p in summary['positions']:
                avg_cost = p['total_cost_basis'] / p['total_shares'] if p['total_shares'] else 0
                positions_data.append({
                    "Ticker": p['ticker'],
                    "Shares": p['total_shares'],
                    "Avg Cost": format_currency(avg_cost),
                    "Current Price": format_currency(p['current_price']),
                    "Market Value": format_currency(p['current_value']),
                    "Gain/Loss": format_currency(p['unrealized_gain']),
                    "Gain %": format_percentage(p['unrealized_gain_pct']),
                    "ST Shares": p['short_term_shares'],
                    "LT Shares": p['long_term_shares']
                })
            
            st.dataframe(positions_data, use_container_width=True, hide_index=True)
            
            # Cache notice
            st.caption("💡 Prices are cached for 15 minutes to avoid rate limits. Click 'Refresh Prices' to update.")
        else:
            st.info("No positions found. Add trades in the 'Portfolio & Taxes' page to see your portfolio summary.")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.info("Make sure your database is properly configured.")


def portfolio_taxes_page():
    """Display portfolio and tax management page with trade entry and CSV import."""
    st.title("💼 Portfolio & Taxes")
    st.markdown("---")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📁 CSV Import", "📊 Tax Lots"])
    
    # Tab 1: Manual Trade Entry
    with tab1:
        st.subheader("Add New Trade")
        
        with st.form("trade_entry_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                ticker = st.text_input(
                    "Ticker Symbol",
                    placeholder="e.g., AAPL",
                    help="Enter the stock ticker symbol"
                ).upper()
                
                trade_date = st.date_input(
                    "Trade Date",
                    value=date.today(),
                    help="Select the date of the trade"
                )
                
                shares = st.number_input(
                    "Number of Shares",
                    min_value=0.0001,
                    step=0.0001,
                    format="%.4f",
                    help="Enter the number of shares"
                )
            
            with col2:
                price_per_share = st.number_input(
                    "Price per Share ($)",
                    min_value=0.01,
                    step=0.01,
                    format="%.2f",
                    help="Enter the price per share"
                )
                
                trade_type = st.selectbox(
                    "Trade Type",
                    options=["Buy", "Sell"],
                    help="Select whether this is a buy or sell transaction"
                )
            
            submitted = st.form_submit_button("Add Trade", use_container_width=True)
            
            if submitted:
                if not ticker:
                    st.error("Please enter a ticker symbol.")
                elif shares <= 0:
                    st.error("Number of shares must be greater than 0.")
                elif price_per_share <= 0:
                    st.error("Price per share must be greater than 0.")
                else:
                    try:
                        db: Session = get_db_session()
                        
                        new_trade = Trades(
                            ticker=ticker,
                            trade_date=trade_date,
                            shares=Decimal(str(shares)),
                            price_per_share=Decimal(str(price_per_share)),
                            trade_type=TradeType.BUY if trade_type == "Buy" else TradeType.SELL
                        )
                        
                        db.add(new_trade)
                        db.commit()
                        db.close()
                        
                        # Clear cache so new trade shows up
                        st.cache_data.clear()
                        
                        st.success(f"✅ Successfully added {trade_type} trade: {shares} shares of {ticker} at ${price_per_share:.2f}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error adding trade: {str(e)}")
                        if db:
                            db.rollback()
                            db.close()
    
    # Tab 2: CSV Import
    with tab2:
        st.subheader("Import Trades from CSV")
        
        st.markdown("""
        Upload a CSV file with your trade history. The CSV should contain columns for:
        - **Ticker**: Stock symbol (e.g., AAPL)
        - **Date**: Trade date (e.g., 2024-01-15 or 01/15/2024)
        - **Quantity**: Number of shares
        - **Price**: Price per share
        - **Action**: Buy or Sell
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your trade history CSV"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("---")
                st.subheader("Map Your Columns")
                st.write("Select which column in your CSV corresponds to each field:")
                
                csv_columns = list(df.columns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    ticker_col = st.selectbox(
                        "Ticker Column",
                        options=csv_columns,
                        help="Column containing stock symbols"
                    )
                    
                    date_col = st.selectbox(
                        "Date Column",
                        options=csv_columns,
                        help="Column containing trade dates"
                    )
                    
                    quantity_col = st.selectbox(
                        "Quantity Column",
                        options=csv_columns,
                        help="Column containing number of shares"
                    )
                
                with col2:
                    price_col = st.selectbox(
                        "Price Column",
                        options=csv_columns,
                        help="Column containing price per share"
                    )
                    
                    action_col = st.selectbox(
                        "Action Column",
                        options=csv_columns,
                        help="Column containing Buy/Sell indicator"
                    )
                
                # Option to handle duplicates
                skip_dupes = st.checkbox(
                    "Skip duplicate trades",
                    value=True,
                    help="If checked, trades that already exist in the database will be skipped"
                )
                
                if st.button("Import Trades", use_container_width=True, type="primary"):
                    column_mapping = {
                        'ticker': ticker_col,
                        'date': date_col,
                        'quantity': quantity_col,
                        'price': price_col,
                        'action': action_col
                    }
                    
                    try:
                        db = get_db_session()
                        successful, failed, duplicates = import_trades_from_csv(
                            db, df, column_mapping, skip_duplicates=skip_dupes
                        )
                        db.close()
                        
                        if successful > 0:
                            st.success(f"✅ Successfully imported {successful} trades!")
                            # Clear cache so new trades show up
                            st.cache_data.clear()
                        if duplicates > 0:
                            st.info(f"ℹ️ {duplicates} duplicate trades were skipped")
                        if failed > 0:
                            st.warning(f"⚠️ {failed} trades could not be imported (check data format)")
                        if successful == 0 and duplicates == 0 and failed == 0:
                            st.info("No valid trades found in CSV. Check that your Action column contains Buy/Sell transactions.")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error importing trades: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Tab 3: Tax Lots View
    with tab3:
        st.subheader("Tax Lot Details")
        
        try:
            # Use cached portfolio data
            summary = get_portfolio_data()
            
            if summary is None:
                st.error("Error loading portfolio data.")
                return
            
            if not summary['positions']:
                st.info("No tax lots found. Add some trades first!")
            else:
                # Filter by ticker
                tickers = ["All"] + [p['ticker'] for p in summary['positions']]
                selected_ticker = st.selectbox("Filter by Ticker", tickers)
                
                for position in summary['positions']:
                    if selected_ticker != "All" and position['ticker'] != selected_ticker:
                        continue
                    
                    with st.expander(f"📈 {position['ticker']} - {position['total_shares']:.4f} shares", expanded=True):
                        # Position summary metrics
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        
                        with m_col1:
                            st.metric("Current Price", format_currency(position['current_price']))
                        with m_col2:
                            st.metric("Market Value", format_currency(position['current_value']))
                        with m_col3:
                            st.metric("Unrealized Gain", format_currency(position['unrealized_gain']))
                        with m_col4:
                            st.metric("Gain %", format_percentage(position['unrealized_gain_pct']))
                        
                        # Tax lots table
                        st.markdown("**Individual Tax Lots:**")
                        
                        lots_data = []
                        for lot in position['tax_lots']:
                            status_emoji = "🟢" if lot['is_long_term'] else "🟡"
                            if lot['is_near_long_term']:
                                status_emoji = "🔴"
                            
                            current_price = position['current_price']
                            lot_gain = (current_price - lot['cost_basis']) * lot['shares']
                            total_cost = lot['cost_basis'] * lot['shares']
                            
                            lots_data.append({
                                "Status": status_emoji,
                                "Purchase Date": lot['purchase_date'],
                                "Shares": lot['shares'],
                                "Cost Basis": format_currency(lot['cost_basis']),
                                "Total Cost": format_currency(total_cost),
                                "Current Value": format_currency(lot['shares'] * current_price),
                                "Gain/Loss": format_currency(lot_gain),
                                "Holding Period": f"{lot['holding_days']} days",
                                "Tax Status": lot['tax_status'],
                                "Days to LT": lot['days_until_long_term'] if not lot['is_long_term'] else "—"
                            })
                        
                        st.dataframe(lots_data, use_container_width=True, hide_index=True)
                        
                        # Alert for lots near long-term
                        if position['lots_near_long_term']:
                            st.warning(
                                f"⚠️ {len(position['lots_near_long_term'])} lot(s) approaching long-term status! "
                                "Consider holding to qualify for lower tax rates."
                            )
                        
        except Exception as e:
            st.error(f"Error loading tax lots: {str(e)}")
    
    st.markdown("---")
    
    # Recent trades section
    st.subheader("📋 Recent Trades")
    try:
        db: Session = get_db_session()
        recent_trades = db.query(Trades).order_by(Trades.trade_date.desc()).limit(20).all()
        db.close()
        
        if recent_trades:
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    "ID": trade.id,
                    "Ticker": trade.ticker,
                    "Date": trade.trade_date.strftime("%Y-%m-%d"),
                    "Shares": float(trade.shares),
                    "Price": format_currency(trade.price_per_share),
                    "Type": trade.trade_type.value,
                    "Total": format_currency(trade.shares * trade.price_per_share)
                })
            
            st.dataframe(trades_data, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet. Add your first trade using the form above.")
            
    except Exception as e:
        st.error(f"Error loading trades: {str(e)}")


def create_bloomberg_chart(df: pd.DataFrame, ticker: str, insider_transactions=None):
    """
    Create a Bloomberg-style dual chart with candlestick and RSI.
    Optionally overlay insider buy/sell markers (small) on the price panel.
    insider_transactions: list of dicts with 'date', 'transaction' (Buy/Sale), 'shares', 'value', 'name', etc.
    """
    # Bloomberg Dark color palette
    COLORS = {
        'background': '#0d1117',
        'grid': '#21262d',
        'text': '#c9d1d9',
        'gain': '#00ff41',      # Neon green
        'loss': '#ff073a',       # Neon red
        'sma_50': '#58a6ff',     # Blue
        'sma_200': '#f78166',    # Orange
        'bb_fill': 'rgba(88, 166, 255, 0.1)',
        'bb_line': '#8b949e',
        'rsi_line': '#d2a8ff',   # Purple
        'overbought': '#ff6b6b',
        'oversold': '#69db7c',
        'volume_up': 'rgba(0, 255, 65, 0.5)',
        'volume_down': 'rgba(255, 7, 58, 0.5)',
    }
    
    # Create subplot figure with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Action', 'RSI (14)')
    )
    
    # === TOP CHART: Candlestick + Indicators ===
    
    # Bollinger Bands fill
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            mode='lines',
            line=dict(width=1, color=COLORS['bb_line'], dash='dot'),
            name='BB Upper',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            mode='lines',
            line=dict(width=1, color=COLORS['bb_line'], dash='dot'),
            fill='tonexty',
            fillcolor=COLORS['bb_fill'],
            name='BB Lower',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=COLORS['gain'],
            decreasing_line_color=COLORS['loss'],
            increasing_fillcolor=COLORS['gain'],
            decreasing_fillcolor=COLORS['loss'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # 50-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            line=dict(width=2, color=COLORS['sma_50']),
            name='SMA 50'
        ),
        row=1, col=1
    )
    
    # 200-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            mode='lines',
            line=dict(width=2, color=COLORS['sma_200']),
            name='SMA 200'
        ),
        row=1, col=1
    )
    
    # Add signal markers
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    golden_cross = df[df['Signal'] == 'GOLDEN CROSS']
    death_cross = df[df['Signal'] == 'DEATH CROSS']
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color=COLORS['gain'],
                    line=dict(width=1, color='white')
                ),
                name='BUY Signal',
                hovertext='BUY'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color=COLORS['loss'],
                    line=dict(width=1, color='white')
                ),
                name='SELL Signal',
                hovertext='SELL'
            ),
            row=1, col=1
        )
    
    if not golden_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=golden_cross.index,
                y=golden_cross['SMA_50'],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=16,
                    color='#ffd700',
                    line=dict(width=1, color='white')
                ),
                name='Golden Cross',
                hovertext='GOLDEN CROSS'
            ),
            row=1, col=1
        )
    
    if not death_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=death_cross.index,
                y=death_cross['SMA_50'],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=14,
                    color=COLORS['loss'],
                    line=dict(width=2, color=COLORS['loss'])
                ),
                name='Death Cross',
                hovertext='DEATH CROSS'
            ),
            row=1, col=1
        )

    # Insider buy/sell markers (small, cap 20)
    if insider_transactions and len(insider_transactions) > 0:
        try:
            date_to_idx = {}
            for i in range(len(df)):
                ts = df.index[i]
                d = ts.date() if hasattr(ts, "date") and callable(getattr(ts, "date")) else (ts if isinstance(ts, date) else None)
                if d is not None:
                    date_to_idx.setdefault(d, []).append(i)
            limited = insider_transactions[:20]
            buy_x, buy_y, buy_hover = [], [], []
            sell_x, sell_y, sell_hover = [], [], []
            for t in limited:
                d = t.get("date")
                if d is None:
                    continue
                d = d.date() if hasattr(d, "date") and callable(getattr(d, "date")) else d
                if d not in date_to_idx:
                    continue
                idx = date_to_idx[d][0]
                ts = df.index[idx]
                low_val = df["Low"].iloc[idx]
                high_val = df["High"].iloc[idx]
                name = t.get("name") or ""
                sh = t.get("shares") or 0
                h = f"{name} {t.get('transaction', '')} {sh} shares"
                if t.get("transaction") == "Buy":
                    buy_x.append(ts)
                    buy_y.append(float(low_val) * 0.97)
                    buy_hover.append(h)
                else:
                    sell_x.append(ts)
                    sell_y.append(float(high_val) * 1.02)
                    sell_hover.append(h)
            if buy_x:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x, y=buy_y, mode="markers",
                        marker=dict(symbol="circle", size=8, color=COLORS["gain"], line=dict(width=1, color="white")),
                        name="Insider buy", hovertext=buy_hover,
                    ),
                    row=1, col=1,
                )
            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x, y=sell_y, mode="markers",
                        marker=dict(symbol="circle", size=8, color=COLORS["loss"], line=dict(width=1, color="white")),
                        name="Insider sell", hovertext=sell_hover,
                    ),
                    row=1, col=1,
                )
        except Exception:
            pass

    # === BOTTOM CHART: RSI ===
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI_14'],
            mode='lines',
            line=dict(width=2, color=COLORS['rsi_line']),
            name='RSI',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Overbought line (70)
    fig.add_hline(
        y=70, 
        line_dash='dash', 
        line_color=COLORS['overbought'],
        line_width=1,
        row=2, col=1,
        annotation_text='Overbought (70)',
        annotation_position='right'
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30, 
        line_dash='dash', 
        line_color=COLORS['oversold'],
        line_width=1,
        row=2, col=1,
        annotation_text='Oversold (30)',
        annotation_position='right'
    )
    
    # Middle line (50)
    fig.add_hline(
        y=50, 
        line_dash='dot', 
        line_color=COLORS['grid'],
        line_width=1,
        row=2, col=1
    )
    
    # Add RSI fill zones
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor='rgba(255, 107, 107, 0.1)',
        line_width=0,
        row=2, col=1
    )
    
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor='rgba(105, 219, 124, 0.1)',
        line_width=0,
        row=2, col=1
    )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> Technical Analysis',
            font=dict(size=24, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Monaco, monospace'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        height=800,
        margin=dict(l=60, r=60, t=100, b=60),
        xaxis_rangeslider_visible=False
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    # Price panel y-axis: scale to visible data so dropdown/custom range views are readable
    y_min = float(df['Low'].min()) * 0.99
    y_max = float(df['High'].max()) * 1.01
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
    # RSI y-axis range
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig


def create_valuation_chart(valuation_data: dict) -> go.Figure:
    """
    Create a stacked panel valuation chart with P/E Multiple and Revenue Growth.
    
    Top panel: P/E Multiple over time
    Bottom panel: Revenue Growth YoY % (historical + projected)
    """
    # Color palette matching Bloomberg dark theme
    COLORS = {
        'background': '#0d1117',
        'grid': '#21262d',
        'text': '#c9d1d9',
        'pe_line': '#00D4AA',         # Teal for P/E
        'revenue_line': '#FF6B6B',     # Coral for revenue
        'projected_line': '#FF6B6B',   # Same color, dashed
        'zero_line': '#8b949e',
        'positive_zone': 'rgba(0, 212, 170, 0.1)',
        'negative_zone': 'rgba(255, 107, 107, 0.1)',
    }
    
    ticker = valuation_data['ticker']
    pe_df = valuation_data['pe_data']
    revenue_df = valuation_data['revenue_data']
    
    # Check if revenue data has QoQ growth
    has_qoq_data = False
    if not revenue_df.empty and 'growth_type' in revenue_df.columns:
        has_qoq_data = (revenue_df['growth_type'] == 'qoq').any()
    
    revenue_title = '<b>Revenue Growth %</b> <span style="font-size:12px;color:#8b949e">(YoY where available, QoQ otherwise)</span>' if has_qoq_data else '<b>Revenue Growth YoY %</b> <span style="font-size:12px;color:#8b949e">(Quarterly)</span>'
    
    # Determine P/E data source for title
    pe_source = valuation_data.get('pe_source', 'unknown')
    pe_source_labels = {
        'alpha_vantage': 'via Alpha Vantage',
        'finnhub': 'via Finnhub',
        'fmp': 'via Financial Modeling Prep',
        'calculated': 'Calculated from EPS',
        'estimated': 'Estimated',
        'unknown': ''
    }
    pe_source_label = pe_source_labels.get(pe_source, '')
    pe_title = f'<b>P/E Multiple</b> <span style="font-size:12px;color:#8b949e">(TTM{" - " + pe_source_label if pe_source_label else ""})</span>'
    
    # Create subplot figure with shared x-axis concept (but separate time scales)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            pe_title,
            revenue_title
        )
    )
    
    # === TOP PANEL: P/E Multiple ===
    if not pe_df.empty and len(pe_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=pe_df['date'],
                y=pe_df['pe'],
                mode='lines+markers',
                line=dict(width=3, color=COLORS['pe_line']),
                marker=dict(size=8, color=COLORS['pe_line'], 
                           line=dict(width=1, color='white')),
                name='P/E Ratio (quarter-end)',
                hovertemplate='<b>P/E:</b> %{y:.1f}x<br><b>Date:</b> %{x|%b %Y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add "Today" point so current P/E is visible and distinct from last quarterly point
        if valuation_data['current_pe']:
            from datetime import datetime as _dt_now
            now = _dt_now.now()
            current_pe = valuation_data['current_pe']
            last_qtr_date = pe_df['date'].iloc[-1]
            last_qtr_pe = pe_df['pe'].iloc[-1]
            # Connector line from last quarter to today (dashed) if they differ
            if abs(last_qtr_pe - current_pe) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=[last_qtr_date, now],
                        y=[last_qtr_pe, current_pe],
                        mode='lines',
                        line=dict(width=2, color=COLORS['pe_line'], dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            fig.add_trace(
                go.Scatter(
                    x=[now],
                    y=[current_pe],
                    mode='markers',
                    marker=dict(size=12, color=COLORS['pe_line'], symbol='diamond',
                               line=dict(width=2, color='white')),
                    name='Today (live)',
                    hovertemplate='<b>Today (live) P/E:</b> %{y:.1f}x<extra></extra>'
                ),
                row=1, col=1
            )
            fig.add_annotation(
                x=now,
                y=current_pe,
                text=f"Today: {current_pe:.1f}x",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLORS['pe_line'],
                font=dict(size=11, color=COLORS['text']),
                bgcolor='rgba(0,0,0,0.7)',
                borderpad=4,
                ax=40,
                ay=0,
                row=1, col=1
            )
        
        # Add forward P/E reference line (show even when Current P/E is N/A, e.g. SNOW)
        if valuation_data.get('forward_pe'):
            current_pe_val = valuation_data.get('current_pe')
            if current_pe_val is None or abs(valuation_data['forward_pe'] - current_pe_val) > 1:
                fig.add_hline(
                    y=valuation_data['forward_pe'],
                    line_dash='dot',
                    line_color='#58a6ff',
                    line_width=1,
                    annotation_text=f"Forward P/E: {valuation_data['forward_pe']:.1f}x",
                    annotation_position='right',
                    annotation_font_color='#58a6ff',
                    row=1, col=1
                )
    elif valuation_data['current_pe']:
        # Show current P/E as a single point if we have it but no history
        import datetime
        now = datetime.datetime.now()
        fig.add_trace(
            go.Scatter(
                x=[now],
                y=[valuation_data['current_pe']],
                mode='markers+text',
                marker=dict(size=16, color=COLORS['pe_line'],
                           line=dict(width=2, color='white')),
                text=[f"{valuation_data['current_pe']:.1f}x"],
                textposition='top center',
                textfont=dict(size=14, color=COLORS['text']),
                name='Current P/E',
                hovertemplate='<b>Current P/E:</b> %{y:.1f}x<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add forward P/E as reference line
        if valuation_data['forward_pe']:
            fig.add_hline(
                y=valuation_data['forward_pe'],
                line_dash='dot',
                line_color='#58a6ff',
                line_width=2,
                annotation_text=f"Forward P/E: {valuation_data['forward_pe']:.1f}x",
                annotation_position='right',
                annotation_font_color='#58a6ff',
                row=1, col=1
            )
    else:
        # Show placeholder if no P/E data at all
        fig.add_annotation(
            text="P/E data unavailable (negative earnings)",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['text']),
            row=1, col=1
        )
    
    # === BOTTOM PANEL: Revenue Growth ===
    if not revenue_df.empty and len(revenue_df) > 0:
        # Separate historical and projected data
        historical = revenue_df[revenue_df['is_projected'] == False].copy()
        projected = revenue_df[revenue_df['is_projected'] == True].copy()
        
        # Check if we have mixed growth types (YoY and QoQ)
        has_qoq = 'growth_type' in historical.columns and (historical['growth_type'] == 'qoq').any()
        growth_label = 'Revenue Growth (YoY/QoQ)' if has_qoq else 'Revenue Growth (YoY)'
        
        # Historical revenue growth (solid line)
        if not historical.empty:
            # Create hover text that shows growth type
            hover_texts = []
            for _, row in historical.iterrows():
                growth_type = row.get('growth_type', 'yoy').upper()
                hover_texts.append(f"<b>{growth_type}:</b> {row['growth']:+.1f}%<br><b>Quarter:</b> {row['date'].strftime('%b %Y')}")
            
            fig.add_trace(
                go.Scatter(
                    x=historical['date'],
                    y=historical['growth'],
                    mode='lines+markers',
                    line=dict(width=3, color=COLORS['revenue_line']),
                    marker=dict(size=8, color=COLORS['revenue_line'],
                               line=dict(width=1, color='white')),
                    name=growth_label,
                    hovertext=hover_texts,
                    hoverinfo='text'
                ),
                row=2, col=1
            )
        
        # Projected revenue growth (dashed line)
        if not projected.empty:
            # Connect projected to last historical point
            if not historical.empty:
                connection_x = [historical['date'].iloc[-1], projected['date'].iloc[0]]
                connection_y = [historical['growth'].iloc[-1], projected['growth'].iloc[0]]
                
                fig.add_trace(
                    go.Scatter(
                        x=connection_x,
                        y=connection_y,
                        mode='lines',
                        line=dict(width=2, color=COLORS['projected_line'], dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
            
            fig.add_trace(
                go.Scatter(
                    x=projected['date'],
                    y=projected['growth'],
                    mode='lines+markers',
                    line=dict(width=3, color=COLORS['projected_line'], dash='dash'),
                    marker=dict(size=8, color=COLORS['projected_line'],
                               symbol='diamond',
                               line=dict(width=1, color='white')),
                    name='Revenue Growth (Projected)',
                    hovertemplate='<b>Projected Growth:</b> %{y:+.1f}%<br><b>Period:</b> %{x|%b %Y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add zero line for reference
        fig.add_hline(
            y=0,
            line_dash='solid',
            line_color=COLORS['zero_line'],
            line_width=1,
            row=2, col=1
        )
        
        # Add positive/negative zones
        all_growth = revenue_df['growth'].dropna()
        if not all_growth.empty:
            max_growth = max(all_growth.max(), 10)
            min_growth = min(all_growth.min(), -10)
            
            # Positive zone (above 0)
            fig.add_hrect(
                y0=0, y1=max_growth + 5,
                fillcolor=COLORS['positive_zone'],
                line_width=0,
                row=2, col=1
            )
            
            # Negative zone (below 0)
            fig.add_hrect(
                y0=min_growth - 5, y1=0,
                fillcolor=COLORS['negative_zone'],
                line_width=0,
                row=2, col=1
            )
    else:
        # Show placeholder if no revenue data
        fig.add_annotation(
            text="Revenue data unavailable",
            xref="x2 domain", yref="y2 domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['text']),
            row=2, col=1
        )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> Valuation Analysis',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Monaco, monospace'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=620,
        margin=dict(l=64, r=64, t=90, b=56),
        hovermode='x unified',
        dragmode='zoom',
    )
    
    # Apply time range from toggle so both panels show the same x-axis window
    requested_years = valuation_data.get('requested_years', 2)
    if requested_years:
        from datetime import datetime as _dt_now, timedelta as _td
        range_end = _dt_now.now()
        range_start = range_end - _td(days=int(requested_years * 365.25))
        fig.update_xaxes(
            range=[range_start, range_end],
            row=1, col=1
        )
        fig.update_xaxes(
            range=[range_start, range_end],
            row=2, col=1
        )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10),
        tickformat='%b %Y'
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=10)
    )
    
    # Y-axis formatting
    fig.update_yaxes(title_text="P/E Multiple", ticksuffix="x", row=1, col=1)
    fig.update_yaxes(title_text="YoY Growth", ticksuffix="%", row=2, col=1)
    
    # Ensure P/E panel y-axis includes Forward P/E line when it sits below/above the data
    if not pe_df.empty and len(pe_df) > 0:
        y_vals = list(pe_df['pe'])
        if valuation_data.get('current_pe'):
            y_vals.append(valuation_data['current_pe'])
        if valuation_data.get('forward_pe'):
            y_vals.append(valuation_data['forward_pe'])
        y_min = max(0, min(y_vals) - 5)
        y_max = max(y_vals) + 10
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
    # Update subplot title colors
    fig.update_annotations(font=dict(color=COLORS['text']))
    
    return fig


def create_tradingview_chart(df: pd.DataFrame, ticker: str, timeframe: str = '1W') -> go.Figure:
    """
    Create a TradingView-style chart with three panels.
    
    Top panel: Momentum oscillator (-50 to +50)
    Middle panel: Candlestick with supply/demand zones and signals
    Bottom panel: Secondary oscillator (-40 to +40)
    
    Args:
        df: DataFrame with OHLCV data and TradingView signals calculated
        ticker: Stock ticker symbol
        timeframe: Display timeframe label (e.g., '1W', '1D')
        
    Returns:
        Plotly figure object
    """
    # TradingView-inspired color palette
    COLORS = {
        'background': '#131722',      # TradingView dark background
        'grid': '#1e222d',
        'text': '#d1d4dc',
        'candle_up': '#26a69a',       # Teal/cyan for bullish
        'candle_down': '#ef5350',     # Coral/pink for bearish (lighter than red)
        'candle_up_body': '#26a69a',
        'candle_down_body': '#ef5350',
        'zone_upper': 'rgba(233, 30, 99, 0.15)',    # Pink zones
        'zone_lower': 'rgba(233, 30, 99, 0.15)',
        'zone_line': 'rgba(233, 30, 99, 0.5)',
        'support_line': '#ffd700',    # Gold support line
        'momentum_pos': '#26a69a',    # Teal when positive
        'momentum_neg': '#ef5350',    # Pink when negative
        'oscillator_pos': '#26a69a',
        'oscillator_neg': '#ef5350',
        'signal_buy': '#00e5ff',      # Cyan diamond for buy
        'signal_sell': '#ff4081',     # Pink diamond for sell
    }
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.15, 0.65, 0.20],
        subplot_titles=('', f'{ticker} · {timeframe}', '')
    )
    
    # === TOP PANEL: Momentum Oscillator ===
    if 'TV_Momentum' in df.columns:
        momentum = df['TV_Momentum'].fillna(0)
        
        # Create color array based on positive/negative values
        colors = [COLORS['momentum_pos'] if v >= 0 else COLORS['momentum_neg'] for v in momentum]
        
        # Area fill for momentum
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=momentum,
                mode='lines',
                fill='tozeroy',
                line=dict(width=1, color=COLORS['momentum_pos']),
                fillcolor='rgba(38, 166, 154, 0.3)',
                name='Momentum',
                showlegend=False,
                hovertemplate='Momentum: %{y:.1f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add negative fill separately
        momentum_neg = momentum.where(momentum < 0, 0)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=momentum_neg,
                mode='lines',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(239, 83, 80, 0.3)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash='solid', line_color=COLORS['grid'], line_width=1, row=1, col=1)
        
        # Add current value annotation
        current_momentum = momentum.iloc[-1]
        fig.add_annotation(
            x=df.index[-1],
            y=current_momentum,
            text=f"{current_momentum:.1f}",
            showarrow=False,
            font=dict(size=10, color=COLORS['momentum_pos'] if current_momentum >= 0 else COLORS['momentum_neg']),
            bgcolor=COLORS['background'],
            xanchor='left',
            xshift=5,
            row=1, col=1
        )
        
        # Add sell signals on momentum panel (pink diamonds at peaks)
        if 'TV_Signal' in df.columns:
            sell_signals = df[(df['TV_Signal'] == 'SELL') & (df['TV_Momentum'] > 15)]
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['TV_Momentum'],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=COLORS['signal_sell'],
                            line=dict(width=1, color='white')
                        ),
                        name='Sell Signal (Momentum)',
                        showlegend=False,
                        hovertemplate='SELL Signal<extra></extra>'
                    ),
                    row=1, col=1
                )
    
    # === MIDDLE PANEL: Candlestick with Zones ===
    
    # Supply zone (upper pink band)
    if 'TV_Zone_Outer_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Outer_Upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Inner_Upper'],
                mode='lines',
                line=dict(width=1, color=COLORS['zone_line']),
                fill='tonexty',
                fillcolor=COLORS['zone_upper'],
                name='Supply Zone',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Demand zone (lower pink band)
    if 'TV_Zone_Inner_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Inner_Lower'],
                mode='lines',
                line=dict(width=1, color=COLORS['zone_line']),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Zone_Outer_Lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=COLORS['zone_lower'],
                name='Demand Zone',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=COLORS['candle_up'],
            decreasing_line_color=COLORS['candle_down'],
            increasing_fillcolor=COLORS['candle_up_body'],
            decreasing_fillcolor=COLORS['candle_down_body'],
            name='Price',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Gold support line (50 SMA)
    if 'TV_Support' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TV_Support'],
                mode='lines',
                line=dict(width=2, color=COLORS['support_line']),
                name='Support (SMA 50)',
                hovertemplate='Support: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add price annotation
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_color = COLORS['candle_up'] if current_price >= prev_price else COLORS['candle_down']
    
    fig.add_annotation(
        x=df.index[-1],
        y=current_price,
        text=f"${current_price:.2f}",
        showarrow=False,
        font=dict(size=11, color=price_color, family='Monaco, monospace'),
        bgcolor=price_color,
        bordercolor=price_color,
        borderwidth=1,
        borderpad=3,
        xanchor='left',
        xshift=10,
        row=2, col=1
    )
    
    # Buy signals on price chart (cyan diamonds)
    if 'TV_Signal' in df.columns:
        buy_signals = df[df['TV_Signal'] == 'BUY']
        sell_signals = df[df['TV_Signal'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Low'] * 0.98,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=COLORS['signal_buy'],
                        line=dict(width=1, color='white')
                    ),
                    name='Buy Signal',
                    hovertemplate='BUY Signal<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['High'] * 1.02,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=COLORS['signal_sell'],
                        line=dict(width=1, color='white')
                    ),
                    name='Sell Signal',
                    hovertemplate='SELL Signal<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # === BOTTOM PANEL: Secondary Oscillator ===
    if 'TV_Oscillator' in df.columns:
        oscillator = df['TV_Oscillator'].fillna(0)
        
        # Area fill for oscillator
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=oscillator,
                mode='lines',
                fill='tozeroy',
                line=dict(width=1, color=COLORS['oscillator_pos']),
                fillcolor='rgba(38, 166, 154, 0.3)',
                name='Oscillator',
                showlegend=False,
                hovertemplate='Oscillator: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Negative fill
        oscillator_neg = oscillator.where(oscillator < 0, 0)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=oscillator_neg,
                mode='lines',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(239, 83, 80, 0.3)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash='solid', line_color=COLORS['grid'], line_width=1, row=3, col=1)
        
        # Current value annotation
        current_osc = oscillator.iloc[-1]
        fig.add_annotation(
            x=df.index[-1],
            y=current_osc,
            text=f"{current_osc:.1f}",
            showarrow=False,
            font=dict(size=10, color=COLORS['oscillator_pos'] if current_osc >= 0 else COLORS['oscillator_neg']),
            bgcolor=COLORS['background'],
            xanchor='left',
            xshift=5,
            row=3, col=1
        )
        
        # Add buy signals on oscillator panel (cyan diamonds at troughs)
        if 'TV_Signal' in df.columns:
            buy_signals = df[(df['TV_Signal'] == 'BUY') & (df['TV_Oscillator'] < -10)]
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['TV_Oscillator'],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=COLORS['signal_buy'],
                            line=dict(width=1, color='white')
                        ),
                        name='Buy Signal (Oscillator)',
                        showlegend=False,
                        hovertemplate='BUY Signal<extra></extra>'
                    ),
                    row=3, col=1
                )
    
    # === LAYOUT STYLING ===
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> TradingView Analysis · {timeframe}',
            font=dict(size=20, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'], family='Trebuchet MS, sans-serif'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=10),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        height=900,
        margin=dict(l=60, r=80, t=80, b=40),
        xaxis_rangeslider_visible=False
    )
    
    # Update all x-axes
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=False,  # Hide labels except bottom
        row=1, col=1
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=False,
        row=2, col=1
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        showticklabels=True,
        row=3, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        range=[-50, 50],
        title_text='',
        row=1, col=1
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        tickprefix='$',
        title_text='',
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=COLORS['grid'],
        tickfont=dict(size=9),
        range=[-40, 40],
        title_text='',
        row=3, col=1
    )
    
    # Update subplot title styling
    fig.update_annotations(font=dict(color=COLORS['text'], size=14))
    
    return fig


def get_signal_badge(signal: str) -> str:
    """Return HTML badge for signal display."""
    if signal == 'BUY':
        return '<span style="background-color: #00ff41; color: black; padding: 4px 12px; border-radius: 4px; font-weight: bold;">BUY</span>'
    elif signal == 'SELL':
        return '<span style="background-color: #ff073a; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">SELL</span>'
    elif signal == 'GOLDEN CROSS':
        return '<span style="background-color: #ffd700; color: black; padding: 4px 12px; border-radius: 4px; font-weight: bold;">⭐ GOLDEN</span>'
    elif signal == 'DEATH CROSS':
        return '<span style="background-color: #8b0000; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">☠️ DEATH</span>'
    else:
        return '<span style="background-color: #6c757d; color: white; padding: 4px 12px; border-radius: 4px;">NEUTRAL</span>'


def market_analysis_page():
    """Display market analysis page with technical indicators and signals."""
    st.title("📈 Market Analysis")
    
    # Custom CSS for Bloomberg dark theme
    st.markdown("""
        <style>
        .signal-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }
        .metric-highlight {
            font-size: 28px;
            font-weight: bold;
            font-family: 'Monaco', monospace;
        }
        .gain { color: #00ff41; }
        .loss { color: #ff073a; }
        .neutral { color: #8b949e; }
        .indicator-label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === TICKER SEARCH ===
    col_search, col_refresh = st.columns([4, 1])
    
    with col_search:
        ticker_input = st.text_input(
            "🔍 Search Ticker",
            placeholder="Enter ticker symbol (e.g., AAPL, GOOGL, MSFT)",
            help="Enter a stock ticker to analyze",
            key="market_analysis_ticker",
        ).upper().strip()
    
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear Cache", help="Clear cached data and refresh"):
            clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    # === MAIN ANALYSIS VIEW ===
    if ticker_input:
        with st.spinner(f"Loading {ticker_input} data..."):
            df = fetch_ohlcv(ticker_input, period_years=50)
        
        if df is None or df.empty:
            st.error(f"❌ Could not fetch data for '{ticker_input}'. Please check the ticker symbol.")
        else:
            # Calculate signals
            df = calculate_signals(df)
            summary = _cached_get_ticker_summary(ticker_input)
            
            if summary:
                # === KEY METRICS ROW ===
                st.markdown("### 📊 Current Status")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    st.metric(
                        "Price",
                        f"${summary['current_price']:,.2f}",
                        f"{summary['daily_change_pct']:+.2f}%",
                        delta_color="normal" if summary['daily_change'] >= 0 else "inverse"
                    )
                
                with m2:
                    rsi_val = summary['rsi']
                    rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "⚪"
                    st.metric("RSI (14)", f"{rsi_color} {rsi_val:.1f}")
                
                with m3:
                    st.metric("SMA 50", f"${summary['sma_50']:,.2f}" if summary['sma_50'] else "N/A")
                
                with m4:
                    st.metric("SMA 200", f"${summary['sma_200']:,.2f}" if summary['sma_200'] else "N/A")
                
                with m5:
                    trend_emoji = "📈" if summary['trend'] == 'BULLISH' else "📉" if summary['trend'] == 'BEARISH' else "➡️"
                    st.metric("Trend", f"{trend_emoji} {summary['trend']}")
                
                # === COMPANY PROFILE (expander) ===
                try:
                    profile = _cached_get_company_profile(ticker_input)
                    with st.expander("Company profile", expanded=False):
                        if profile is None:
                            st.caption("Profile unavailable.")
                        else:
                            full_desc = profile.get("description") or ""
                            if full_desc:
                                st.markdown(full_desc)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption(f"**Sector:** {profile.get('sector') or 'N/A'}")
                                st.caption(f"**Industry:** {profile.get('industry') or 'N/A'}")
                                if profile.get("website"):
                                    st.markdown(f"**Website:** [Link]({profile['website']})")
                                else:
                                    st.caption("**Website:** N/A")
                            with c2:
                                st.caption(f"**Employees:** {profile.get('employees') or 'N/A'}")
                                st.caption(f"**CEO:** {profile.get('ceo') or 'N/A'}")
                except Exception:
                    with st.expander("Company profile", expanded=False):
                        st.caption("Profile unavailable.")
                
                # === SIGNAL DISPLAY ===
                st.markdown("---")
                
                signal_col, info_col = st.columns([1, 2])
                
                with signal_col:
                    st.markdown("### 🎯 Current Signal")
                    st.markdown(get_signal_badge(summary['signal']), unsafe_allow_html=True)
                
                with info_col:
                    st.markdown("### 📈 Signal Logic")
                    st.markdown("""
                    | Signal | Condition |
                    |--------|-----------|
                    | **BUY** | Price < Lower BB AND RSI < 35 |
                    | **SELL** | Price > Upper BB AND RSI > 65 |
                    | **GOLDEN CROSS** | SMA 50 crosses above SMA 200 |
                    | **DEATH CROSS** | SMA 50 crosses below SMA 200 |
                    """)
                
                st.markdown("---")
                
                # === DUAL CHART ===
                st.markdown("### 📉 Technical Chart")
                
                # Trading days per year for tail slicing
                TRADING_DAYS_PER_YEAR = 252
                time_range_options = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '25Y', '50Y', 'Max', 'Custom']
                date_range = st.selectbox(
                    "Time Range",
                    options=time_range_options,
                    index=3,
                    help="Select the time range to display"
                )
                chart_type = "line" if st.checkbox("Line chart (close only)", value=False, help="Show closing price as a line instead of candlesticks.") else "candlestick"
                show_signals = st.checkbox(
                    "Show buy/sell signals",
                    value=True,
                    help="Show BUY/SELL and cross markers on the price chart.",
                )
                strong_signals_only = st.checkbox(
                    "Strong signals only",
                    value=False,
                    help="When on, only show BUY when RSI < 30 and SELL when RSI > 70 (fewer markers).",
                )

                if date_range == 'Custom':
                    from datetime import timedelta as _td
                    _today = date.today()
                    _default_start = _today - _td(days=365)
                    custom_col1, custom_col2 = st.columns(2)
                    with custom_col1:
                        custom_start = st.date_input("Start date", value=_default_start, key="tech_chart_start")
                    with custom_col2:
                        custom_end = st.date_input("End date", value=_today, key="tech_chart_end")
                    if custom_start and custom_end and custom_start <= custom_end:
                        df_display = df[(df.index.date >= custom_start) & (df.index.date <= custom_end)]
                        if df_display.empty:
                            st.warning("No data in the selected date range.")
                            df_display = df.tail(TRADING_DAYS_PER_YEAR)
                    else:
                        df_display = df.tail(TRADING_DAYS_PER_YEAR)
                elif date_range == '3M':
                    df_display = df.tail(63)
                elif date_range == '6M':
                    df_display = df.tail(126)
                elif date_range == '1Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR)
                elif date_range == '2Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 2)
                elif date_range == '5Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 5)
                elif date_range == '10Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 10)
                elif date_range == '15Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 15)
                elif date_range == '25Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 25)
                elif date_range == '50Y':
                    df_display = df.tail(TRADING_DAYS_PER_YEAR * 50)
                else:
                    df_display = df
                
                # Edge-case captions: data shorter than requested; SMA 200 partial
                if not df_display.empty and date_range not in ('Custom', 'Max'):
                    expected_days = {
                        '5Y': TRADING_DAYS_PER_YEAR * 5, '10Y': TRADING_DAYS_PER_YEAR * 10,
                        '15Y': TRADING_DAYS_PER_YEAR * 15, '25Y': TRADING_DAYS_PER_YEAR * 25,
                        '50Y': TRADING_DAYS_PER_YEAR * 50,
                    }
                    expected = expected_days.get(date_range)
                    if expected and len(df_display) < expected * 0.9:
                        first_ts = df_display.index.min()
                        first_str = first_ts.strftime("%Y-%m-%d") if hasattr(first_ts, "strftime") else str(first_ts)[:10]
                        st.caption(f"Data from {first_str} (all available).")
                if len(df) < 200:
                    st.caption("SMA 200 is shown with partial data (fewer than 200 trading days).")
                
                # Insider transactions for chart overlay (optional)
                insider_list = []
                try:
                    insider_list = _cached_fetch_insider_transactions(ticker_input)
                except Exception:
                    pass
                # TradingView Lightweight Charts: candlestick or line + volume + support lines + markers + RSI (zoom + double-click reset)
                tech_data = df_to_technical_chart_data(df_display, strong_signals_only=strong_signals_only)
                markers_to_show = (tech_data["markers"] or None) if show_signals else None
                chart_config = build_technical_chart_config(
                    ticker_input,
                    tech_data["candles"],
                    tech_data["volume"],
                    dark_theme=True,
                    rsi=tech_data["rsi"] or None,
                    sma_50=tech_data["sma_50"] or None,
                    sma_200=tech_data["sma_200"] or None,
                    bb_upper=tech_data["bb_upper"] or None,
                    bb_lower=tech_data["bb_lower"] or None,
                    markers=markers_to_show,
                    price_series_type=chart_type,
                )
                renderLightweightCharts(chart_config, key=f"technical_chart_{ticker_input}")

                with st.expander("Chart guide — lines and signals", expanded=False):
                    st.markdown("""
                    Use **Line chart (close only)** to show closing price as a line instead of candlesticks.
                    **Show buy/sell signals** toggles markers; **Strong signals only** shows fewer markers (BUY when RSI &lt; 30, SELL when RSI &gt; 70).

                    **Price panel**
                    | Line / marker | Meaning |
                    |----------------|---------|
                    | **Blue line** | SMA 50 (50-day simple moving average) |
                    | **Orange line** | SMA 200 (200-day simple moving average) |
                    | **Gray dotted lines** | Bollinger Bands (upper and lower) |
                    | **Green ↑** | Buy signal (price &lt; lower band and RSI &lt; 35) |
                    | **Red ↓** | Sell signal (price &gt; upper band and RSI &gt; 65) |
                    | **Gold dot** | Golden cross (SMA 50 crosses above SMA 200) |
                    | **Dark red dot** | Death cross (SMA 50 crosses below SMA 200) |
                    | **Green/teal bars** | Volume (lower section) |

                    **RSI panel:** Purple line = RSI (14). Values above 70 = overbought; below 30 = oversold.
                    """)
                with st.expander("Insider transactions", expanded=False):
                    if not insider_list:
                        st.caption("No recent insider data or API unavailable. Set FINNHUB_API_KEY for insider transactions.")
                    else:
                        try:
                            profile = _cached_get_company_profile(ticker_input)
                            ceo_name = (profile.get("ceo") or "").strip() if profile else ""
                        except Exception:
                            ceo_name = ""
                        def _name_matches_ceo(name: str, ceo: str) -> bool:
                            if not name or not ceo:
                                return False
                            a = set((name or "").lower().split())
                            b = set((ceo or "").lower().split())
                            return bool(a and b and a == b)
                        rows = []
                        for t in insider_list[:30]:
                            d = t.get("date")
                            d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                            name = (t.get("name") or "").strip()[:40]
                            role = (t.get("relationship") or "").strip()[:30]
                            if not role and ceo_name and _name_matches_ceo(name, ceo_name):
                                role = "CEO"
                            trans_type = t.get("transaction", "")
                            shares = t.get("shares", 0) or 0
                            value = t.get("value", 0) or 0
                            if value <= 0 and shares > 0 and not df_display.empty:
                                try:
                                    d_obj = d.date() if hasattr(d, "date") else d
                                    ts = pd.Timestamp(d_obj)
                                    idx = df_display.index.get_indexer([ts], method="nearest")[0]
                                    if 0 <= idx < len(df_display):
                                        close = float(df_display.iloc[idx]["Close"])
                                        value = int(shares * close)
                                except Exception:
                                    pass
                            sec_link = (t.get("sec_link") or "").strip()
                            rows.append({
                                "date": d_str,
                                "name": name,
                                "role": role or "—",
                                "type": trans_type,
                                "shares": shares,
                                "value": value,
                                "sec_link": sec_link,
                            })
                        # Build HTML table with row colors (green tint = Buy, red tint = Sale)
                        def _esc(s):
                            return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
                        header = "<thead><tr><th>Date</th><th>Name</th><th>Role</th><th>Type</th><th>Shares</th><th>Value ($)</th><th>SEC</th></tr></thead><tbody>"
                        body_parts = []
                        for r in rows:
                            bg = "rgba(0,255,65,0.12)" if r["type"] == "Buy" else "rgba(255,7,58,0.12)"
                            sec_cell = f'<a href="{_esc(r["sec_link"])}" target="_blank" rel="noopener">Form 4</a>' if r["sec_link"] else "—"
                            value_fmt = f"{r['value']:,}" if r["value"] else "—"
                            body_parts.append(
                                f'<tr style="background:{bg}">'
                                f'<td>{_esc(r["date"])}</td><td>{_esc(r["name"])}</td><td>{_esc(r["role"])}</td>'
                                f'<td>{_esc(r["type"])}</td><td>{r["shares"]:,}</td><td>{value_fmt}</td>'
                                f'<td>{sec_cell}</td></tr>'
                            )
                        table_html = f'<table style="width:100%; border-collapse:collapse;"><caption style="text-align:left; margin-bottom:6px;">Buy rows in green, sell in red.</caption>{header}{"".join(body_parts)}</tbody></table>'
                        st.markdown(table_html, unsafe_allow_html=True)

                # === RECENT SIGNALS TABLE ===
                st.markdown("---")
                st.markdown("### 📋 Recent Signals")
                
                recent_signals = df[df['Signal'] != ''].tail(10).copy()
                if not recent_signals.empty:
                    recent_signals = recent_signals[['Close', 'RSI_14', 'SMA_50', 'SMA_200', 'Signal']].copy()
                    recent_signals.columns = ['Price', 'RSI', 'SMA 50', 'SMA 200', 'Signal']
                    recent_signals = recent_signals.round(2)
                    recent_signals.index = recent_signals.index.strftime('%Y-%m-%d')
                    recent_signals = recent_signals.iloc[::-1]  # Most recent first
                    
                    st.dataframe(
                        recent_signals,
                        use_container_width=True,
                        column_config={
                            "Signal": st.column_config.TextColumn("Signal", width="medium"),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                            "SMA 50": st.column_config.NumberColumn("SMA 50", format="$%.2f"),
                            "SMA 200": st.column_config.NumberColumn("SMA 200", format="$%.2f"),
                        }
                    )
                else:
                    st.info("No trading signals generated in the selected period.")
                
                # === VALUATION CHART ===
                st.markdown("---")
                st.markdown("### 💰 Valuation Analysis")
                st.caption("P/E Multiple and Revenue Growth trends help assess if a stock is fairly valued relative to its growth.")
                
                # Time range selector and optional refetch for valuation chart
                val_time_col1, val_time_col2 = st.columns([1, 4])
                with val_time_col1:
                    valuation_years = st.selectbox(
                        "Time Range",
                        options=[2, 5, 10],
                        index=0,
                        format_func=lambda x: f"{x} Years",
                        key="valuation_time_range",
                        help="Select historical time range for valuation analysis"
                    )
                    # Force refetch from API (skip DB) to get best-available P/E from all sources
                    refetch_key = f"valuation_refetch_{ticker_input}"
                    if st.button("🔄 Refetch from API", key=refetch_key, help="Skip database and fetch fresh P/E and revenue from APIs (best available data)"):
                        st.session_state["valuation_skip_db"] = True
                
                skip_db = st.session_state.pop("valuation_skip_db", False)
                with st.spinner(f"Loading {valuation_years}-year valuation data..." + (" (from API)" if skip_db else "")):
                    valuation_data = _cached_get_valuation_chart_data(ticker_input, valuation_years, skip_db)
                
                if valuation_data:
                    # Show key valuation metrics
                    val_col1, val_col2, val_col3, val_col4, val_col5 = st.columns(5)
                    
                    with val_col1:
                        pe_val = valuation_data['current_pe']
                        if pe_val:
                            # Color code P/E
                            pe_color = "🟢" if pe_val < 20 else "🟡" if pe_val < 35 else "🔴"
                            st.metric("Current P/E", f"{pe_color} {pe_val:.1f}x")
                        else:
                            st.metric("Current P/E", "N/A")
                    
                    with val_col2:
                        fwd_pe = valuation_data['forward_pe']
                        if fwd_pe:
                            st.metric("Forward P/E", f"{fwd_pe:.1f}x")
                        else:
                            st.metric("Forward P/E", "N/A")
                    
                    with val_col3:
                        peg = valuation_data.get('peg_ratio')
                        earnings_growth = valuation_data.get('earnings_growth')
                        
                        if peg and peg > 0:
                            # Color code PEG ratio
                            # < 1 = potentially undervalued, 1-2 = fair, > 2 = expensive
                            if peg < 1:
                                peg_color = "🟢"
                                peg_label = "Undervalued"
                            elif peg < 2:
                                peg_color = "🟡"
                                peg_label = "Fair"
                            else:
                                peg_color = "🔴"
                                peg_label = "Expensive"
                            st.metric(
                                "PEG Ratio", 
                                f"{peg_color} {peg:.2f}",
                                peg_label,
                                delta_color="off"
                            )
                        elif earnings_growth:
                            # Show earnings growth instead if PEG unavailable (no delta = no arrow)
                            growth_pct = earnings_growth * 100
                            growth_color = "🟢" if growth_pct > 20 else "🟡" if growth_pct > 0 else "🔴"
                            st.metric(
                                "Earnings Growth (YoY)",
                                f"{growth_color} {growth_pct:+.1f}%",
                                delta=None,
                                delta_color="off"
                            )
                        else:
                            st.metric("PEG Ratio", "N/A")
                    
                    with val_col4:
                        sector = valuation_data.get('sector')
                        st.metric("Sector", sector if sector else "N/A")
                    
                    with val_col5:
                        industry = valuation_data.get('industry')
                        st.metric("Industry", industry[:18] + "..." if industry and len(industry) > 18 else (industry or "N/A"))
                    
                    # Second row: Market cap, 52w high, 52w low (from summary)
                    def _fmt_market_cap(mc):
                        if mc is None or mc <= 0:
                            return "N/A"
                        if mc >= 1e12:
                            return f"${mc / 1e12:.2f}T"
                        if mc >= 1e9:
                            return f"${mc / 1e9:.2f}B"
                        if mc >= 1e6:
                            return f"${mc / 1e6:.2f}M"
                        return f"${mc:,.0f}"
                    v2_1, v2_2, v2_3, v2_4, v2_5 = st.columns(5)
                    with v2_1:
                        st.metric("Market cap", _fmt_market_cap(summary.get("market_cap")))
                    with v2_2:
                        h52 = summary.get("high_52w")
                        st.metric("52w high", f"${h52:,.2f}" if h52 is not None else "N/A")
                    with v2_3:
                        l52 = summary.get("low_52w")
                        st.metric("52w low", f"${l52:,.2f}" if l52 is not None else "N/A")
                    with v2_4:
                        pct_52 = summary.get("pct_from_52w_high")
                        st.metric("% from 52w high", f"{pct_52:+.1f}%" if pct_52 is not None else "N/A")
                    with v2_5:
                        st.metric("", "", help="")
                    
                    # Create and display valuation chart (scroll to zoom, double-click to reset)
                    valuation_fig = create_valuation_chart(valuation_data)
                    st.plotly_chart(valuation_fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{ticker_input}_valuation',
                            'height': 600,
                            'width': 1200,
                            'scale': 2
                        }
                    })
                    
                    # Database save button and status
                    from market_data import save_valuation_to_db, check_ticker_in_db
                    
                    db_status = check_ticker_in_db(ticker_input)
                    
                    save_col1, save_col2 = st.columns([3, 1])
                    with save_col1:
                        if db_status['has_data']:
                            status_icon = "✅" if db_status['is_fresh'] else "⚠️"
                            st.caption(f"{status_icon} **{ticker_input}** saved in database: {db_status['quarters']} quarters (most recent: {db_status['most_recent']})")
                        else:
                            st.caption(f"💾 **{ticker_input}** not saved in database yet")
                    
                    with save_col2:
                        # Check if data came from database
                        from_db = valuation_data.get('from_database', False)
                        
                        if from_db:
                            st.success("📂 From DB")
                        else:
                            if st.button("💾 Save", key=f"save_valuation_{ticker_input}", help="Save this ticker's valuation data to your database"):
                                # Get the raw data to save
                                pe_history = valuation_data.get('pe_data', pd.DataFrame())
                                revenue_data = valuation_data.get('revenue_data', pd.DataFrame())
                                
                                # Convert DataFrames back to list format
                                pe_list = []
                                if not pe_history.empty:
                                    for _, row in pe_history.iterrows():
                                        pe_list.append({
                                            'date': row['date'],
                                            'pe': row['pe'],
                                            'ttm_eps': row.get('ttm_eps'),
                                            'price': row.get('price'),
                                            'source': row.get('source', 'api')
                                        })
                                
                                revenue_list = []
                                if not revenue_data.empty:
                                    for _, row in revenue_data.iterrows():
                                        revenue_list.append({
                                            'date': row['date'],
                                            'revenue': row.get('revenue'),
                                            'yoy_growth': row.get('growth') if row.get('growth_type') == 'yoy' else None,
                                            'qoq_growth': row.get('growth') if row.get('growth_type') == 'qoq' else None
                                        })
                                
                                if save_valuation_to_db(ticker_input, pe_list, revenue_list):
                                    st.success(f"✅ Saved {ticker_input} to database!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save. Check database connection.")
                    
                    # Valuation interpretation guide
                    with st.expander("📖 How to Interpret This Chart"):
                        st.markdown("""
                        **P/E Multiple (Top Panel):**
                        - Shows how much investors are willing to pay per dollar of earnings
                        - **Lower P/E** (< 15-20x): May indicate undervaluation or slower growth expectations
                        - **Higher P/E** (> 30-40x): May indicate overvaluation or high growth expectations
                        - **Rising P/E**: Investors becoming more optimistic
                        - **Falling P/E**: Investors becoming more cautious
                        
                        **Revenue Growth YoY % (Bottom Panel):**
                        - Shows the year-over-year revenue growth rate each quarter
                        - **Solid line**: Actual reported results
                        - **Dashed line**: Analyst projections (if available)
                        - **Accelerating growth**: Revenue growth rate increasing → bullish
                        - **Decelerating growth**: Revenue growth rate decreasing → watch for multiple compression
                        
                        ---
                        
                        **🎯 PEG Ratio (P/E to Growth):**
                        
                        The PEG ratio divides P/E by expected annual earnings growth rate. It answers: *"Am I paying a fair price for this growth?"*
                        
                        | PEG Range | Interpretation | Color |
                        |-----------|----------------|-------|
                        | **< 1.0** | Potentially **undervalued** relative to growth | 🟢 |
                        | **1.0 - 2.0** | **Fairly valued** | 🟡 |
                        | **> 2.0** | Potentially **expensive** relative to growth | 🔴 |
                        
                        *Example:* A stock with P/E of 30x and 30% earnings growth has PEG = 1.0 (fair). The same P/E with only 15% growth has PEG = 2.0 (expensive).
                        
                        ---
                        
                        **The Key Relationship:**
                        | P/E Trend | Revenue Trend | Interpretation |
                        |-----------|---------------|----------------|
                        | ↑ Rising | ↑ Accelerating | Growth justified premium |
                        | ↑ Rising | ↓ Decelerating | Potential overvaluation ⚠️ |
                        | ↓ Falling | ↑ Accelerating | Potential opportunity 🎯 |
                        | ↓ Falling | ↓ Decelerating | Fundamentals weakening |
                        """)
                else:
                    st.info(f"📊 Valuation data not available for {ticker_input}. This may be due to limited financial disclosures or the stock being too new.")
                
                # === FUNDAMENTALS AND RATIOS (expander) ===
                try:
                    fundamentals = _cached_get_fundamentals_ratios(ticker_input)
                    with st.expander("Fundamentals and ratios", expanded=False):
                        if fundamentals is None:
                            st.caption("Fundamentals unavailable.")
                        else:
                            rev = fundamentals.get("revenue_ttm")
                            vals = [rev, fundamentals.get("gross_margin"), fundamentals.get("operating_margin"), fundamentals.get("net_margin"), fundamentals.get("roe"), fundamentals.get("roa")]
                            if all(v is None for v in vals):
                                st.caption("Fundamentals unavailable.")
                            else:
                                st.caption(f"**Revenue (TTM):** ${rev:,.0f}" if rev is not None and rev else "**Revenue (TTM):** N/A")
                                for label, key in [
                                    ("Gross margin", "gross_margin"),
                                    ("Operating margin", "operating_margin"),
                                    ("Net margin", "net_margin"),
                                    ("ROE", "roe"),
                                    ("ROA", "roa"),
                                ]:
                                    val = fundamentals.get(key)
                                    if val is not None:
                                        if "margin" in key:
                                            st.caption(f"**{label}:** {val * 100:.2f}%" if abs(val) <= 1 else f"**{label}:** {val:.2f}%")
                                        else:
                                            st.caption(f"**{label}:** {val * 100:.2f}%" if abs(val) <= 1 else f"**{label}:** {val:.2f}")
                                    else:
                                        st.caption(f"**{label}:** N/A")
                except Exception:
                    with st.expander("Fundamentals and ratios", expanded=False):
                        st.caption("Fundamentals unavailable.")
                
                # === TRADINGVIEW-STYLE CHART ===
                st.markdown("---")
                st.markdown("### 📊 TradingView Analysis")
                st.caption("Multi-panel momentum analysis with supply/demand zones and trading signals.")
                
                # Timeframe selector for TradingView chart
                tv_col1, tv_col2 = st.columns([1, 4])
                with tv_col1:
                    tv_timeframe = st.selectbox(
                        "Timeframe",
                        options=['1W', '1D', '4H'],
                        index=0,
                        key="tv_timeframe",
                        help="Select chart timeframe"
                    )
                
                # Determine data to display based on timeframe
                # For weekly, resample daily data to weekly
                if tv_timeframe == '1W':
                    # Resample to weekly data
                    df_tv = df.copy()
                    df_tv_resampled = df_tv.resample('W').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                    # Take last 52 weeks (1 year)
                    df_tv_display = df_tv_resampled.tail(52)
                    timeframe_label = '1W'
                elif tv_timeframe == '4H':
                    # For 4H, use daily data but show more recent period
                    df_tv_display = df.tail(60)  # ~60 days
                    timeframe_label = '4H (Daily proxy)'
                else:  # 1D
                    df_tv_display = df.tail(126)  # 6 months daily
                    timeframe_label = '1D'
                
                # Use cached TradingView signals if fresh; otherwise compute and auto-save
                df_tv_with_signals = load_tv_signals_from_cache(ticker_input, tv_timeframe)
                if df_tv_with_signals is None:
                    df_tv_with_signals = calculate_tradingview_signals(df_tv_display)
                    save_tv_signals_to_cache(ticker_input, df_tv_with_signals, tv_timeframe)
                
                # Create and display TradingView chart
                tv_fig = create_tradingview_chart(df_tv_with_signals, ticker_input, timeframe_label)
                st.plotly_chart(tv_fig, use_container_width=True, config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{ticker_input}_tradingview_{tv_timeframe}',
                        'height': 900,
                        'width': 1400,
                        'scale': 2
                    }
                })
                
                # Show current signal status
                tv_signal_col1, tv_signal_col2, tv_signal_col3 = st.columns(3)
                
                with tv_signal_col1:
                    current_momentum = df_tv_with_signals['TV_Momentum'].iloc[-1] if 'TV_Momentum' in df_tv_with_signals.columns else 0
                    momentum_color = "🟢" if current_momentum > 10 else "🔴" if current_momentum < -10 else "⚪"
                    st.metric("Momentum", f"{momentum_color} {current_momentum:.1f}")
                
                with tv_signal_col2:
                    current_osc = df_tv_with_signals['TV_Oscillator'].iloc[-1] if 'TV_Oscillator' in df_tv_with_signals.columns else 0
                    osc_color = "🟢" if current_osc > 10 else "🔴" if current_osc < -10 else "⚪"
                    st.metric("Oscillator", f"{osc_color} {current_osc:.1f}")
                
                with tv_signal_col3:
                    # Get latest TV signal
                    tv_signals = df_tv_with_signals[df_tv_with_signals['TV_Signal'] != '']
                    if not tv_signals.empty:
                        latest_tv_signal = tv_signals['TV_Signal'].iloc[-1]
                        latest_tv_strength = tv_signals['TV_Signal_Strength'].iloc[-1]
                        signal_emoji = "💎" if latest_tv_signal == "BUY" else "🔻" if latest_tv_signal == "SELL" else "➖"
                        st.metric("Latest Signal", f"{signal_emoji} {latest_tv_signal} ({latest_tv_strength})")
                    else:
                        st.metric("Latest Signal", "➖ None")
                
                # TradingView chart interpretation guide
                with st.expander("📖 TradingView Chart Guide"):
                    st.markdown("""
                    **Top Panel - Momentum Oscillator (-50 to +50):**
                    - Combines Stochastic RSI and Rate of Change
                    - **Cyan fill**: Positive momentum (bullish)
                    - **Pink fill**: Negative momentum (bearish)
                    - **Pink diamonds**: Potential sell signals at peaks
                    
                    **Middle Panel - Price Action:**
                    - **Cyan/Teal candles**: Bullish (close > open)
                    - **Pink/Red candles**: Bearish (close < open)
                    - **Pink shaded zones**: Supply (upper) and Demand (lower) zones
                    - **Gold line**: 50-period moving average support
                    - **Cyan diamonds**: Buy signals
                    - **Pink diamonds**: Sell signals
                    
                    **Bottom Panel - Secondary Oscillator (-40 to +40):**
                    - MACD-style momentum histogram
                    - Confirms signals from top panel
                    - **Cyan diamonds**: Buy confirmation
                    
                    **Signal Logic:**
                    | Condition | Signal Type |
                    |-----------|-------------|
                    | Price at demand zone + both oscillators negative turning up | **STRONG BUY** |
                    | Price near lower band + momentum turning up | **MODERATE BUY** |
                    | Price at supply zone + both oscillators positive turning down | **STRONG SELL** |
                    | Price near upper band + momentum turning down | **MODERATE SELL** |
                    
                    **"Triple Blue" Alignment:**
                    When price, momentum, and oscillator are all positive/bullish simultaneously, this indicates strong upward momentum.
                    """)
                
                # Save button for TradingView signals
                tv_cache_status = check_tv_signals_in_cache(ticker_input, tv_timeframe)
                
                tv_save_col1, tv_save_col2 = st.columns([3, 1])
                with tv_save_col1:
                    if tv_cache_status['has_data']:
                        status_icon = "✅" if tv_cache_status['is_fresh'] else "⚠️"
                        cache_time = tv_cache_status['timestamp'].strftime('%Y-%m-%d %H:%M') if tv_cache_status['timestamp'] else 'Unknown'
                        st.caption(f"{status_icon} **{ticker_input}** TradingView signals cached (saved: {cache_time})")
                    else:
                        st.caption(f"💾 **{ticker_input}** TradingView signals not saved yet")
                
                with tv_save_col2:
                    if tv_cache_status['is_fresh']:
                        st.success("📂 Cached")
                    else:
                        if st.button("💾 Save TV", key=f"save_tv_{ticker_input}_{tv_timeframe}", help="Save TradingView signals to cache"):
                            if save_tv_signals_to_cache(ticker_input, df_tv_with_signals, tv_timeframe):
                                st.success(f"✅ Saved TradingView signals for {ticker_input}!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to save TradingView signals.")
                
                # === COMPETITORS ===
                st.markdown("---")
                st.markdown("### Competitors")
                st.caption("Same industry and similar market cap; optional sort by description match.")
                with st.expander("How we pick peers", expanded=False):
                    st.markdown("""
                    Peers are chosen using **industry** and **similar market cap** (via FMP screener).
                    When your company's industry doesn't match our data provider's list, we use sector-level peers and note it above the table.
                    You can sort by **description match** to rank by text similarity to the company description (same candidate set).
                    Matching uses the **full** company description (not the shortened preview in the profile box above).
                    Description match ranks by text similarity, not verified competitive relationship; results can include companies that sound similar but operate in different segments.
                    """)
                comp_sort = st.radio(
                    "Sort",
                    options=["By industry & size", "By description match"],
                    index=0,
                    key=f"competitors_sort_{ticker_input}",
                    horizontal=True,
                )
                sort_by_val = "description" if comp_sort == "By description match" else "industry_size"
                if "peers_refresh_key" not in st.session_state:
                    st.session_state["peers_refresh_key"] = 0
                ref_col1, ref_col2 = st.columns([4, 1])
                with ref_col2:
                    if st.button("Refresh", key=f"competitors_refresh_{ticker_input}", help="Clear peers cache and reload"):
                        clear_peers_cache(ticker_input)
                        st.session_state["peers_refresh_key"] = st.session_state.get("peers_refresh_key", 0) + 1
                        st.rerun()
                try:
                    comp_result = get_competitors(ticker_input, sort_by_val, max_peers=5)
                    peers_list = comp_result.get("peers") or []
                    fallback_used = comp_result.get("fallback_used", "industry")
                    used_sector_fallback = comp_result.get("used_sector_fallback", False)
                    if fallback_used == "sector":
                        st.caption("Few peers in this industry; showing sector-level (and similar size) peers.")
                    elif fallback_used == "sector_wide_cap":
                        st.caption("Showing sector-level peers with wider size range.")
                    elif fallback_used == "stock_peers":
                        st.caption("Using FMP stock-peers (company-screener not available on your plan). Same sector and similar market cap.")
                    elif used_sector_fallback:
                        st.caption("Showing sector-level peers (industry not matched).")
                    if not peers_list:
                        st.info("No peers found for this industry/market cap. Try another ticker or refresh.")
                    else:
                        def _fmt_mc(mc):
                            if mc is None or mc <= 0:
                                return "—"
                            if mc >= 1e12:
                                return f"${mc / 1e12:.2f}T"
                            if mc >= 1e9:
                                return f"${mc / 1e9:.2f}B"
                            if mc >= 1e6:
                                return f"${mc / 1e6:.2f}M"
                            return f"${mc:,.0f}"
                        rows = []
                        for p in peers_list:
                            ticker_display = p.get("ticker") or "—"
                            name_display = (p.get("name") or ticker_display)[:50]
                            sector_display = (p.get("sector") or "—")[:20]
                            industry_display = (p.get("industry") or "—")[:25]
                            mc_display = _fmt_mc(p.get("market_cap"))
                            pe_display = f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") is not None else "—"
                            rev = p.get("revenue_ttm")
                            rev_display = f"${rev / 1e9:.2f}B" if rev and rev >= 1e9 else (f"${rev / 1e6:.0f}M" if rev and rev >= 1e6 else ("—" if rev is None else f"${rev:,.0f}"))
                            match_display = str(p.get("description_match_score")) if p.get("description_match_score") is not None else "—"
                            rows.append({
                                "Ticker": ticker_display,
                                "Name": name_display,
                                "Sector": sector_display,
                                "Industry": industry_display,
                                "Market cap": mc_display,
                                "P/E": pe_display,
                                "Revenue (TTM)": rev_display,
                                "Match": match_display,
                            })
                        df_comp = pd.DataFrame(rows)
                        if sort_by_val != "description":
                            df_comp = df_comp.drop(columns=["Match"], errors="ignore")
                        st.dataframe(df_comp, use_container_width=True, hide_index=True)
                        st.caption("Analyze a peer:")
                        peer_btns = st.columns(min(len(peers_list), 8))
                        for i, p in enumerate(peers_list[:8]):
                            pt = p.get("ticker")
                            if pt and i < len(peer_btns):
                                with peer_btns[i]:
                                    if st.button(f"→ {pt}", key=f"analyze_peer_{ticker_input}_{pt}", help=f"Load {pt} in search"):
                                        st.session_state["market_analysis_ticker"] = pt
                                        st.rerun()
                        with st.expander("Edit peers", expanded=False):
                            st.caption("Custom additions/removals are saved and applied to future loads.")
                            add_peer = st.text_input("Add ticker to always include", placeholder="e.g. MSFT", key=f"add_peer_{ticker_input}").upper().strip()
                            if st.button("Add", key=f"add_peer_btn_{ticker_input}") and add_peer:
                                try:
                                    db = get_db_session()
                                    from models import PeerOverride
                                    existing = db.query(PeerOverride).filter(
                                        PeerOverride.focus_ticker == ticker_input,
                                        PeerOverride.peer_ticker == add_peer,
                                    ).first()
                                    if not existing:
                                        db.add(PeerOverride(focus_ticker=ticker_input, peer_ticker=add_peer, is_excluded=0))
                                        db.commit()
                                        st.success(f"Added {add_peer} to peers for {ticker_input}.")
                                    else:
                                        existing.is_excluded = 0
                                        db.commit()
                                        st.success(f"{add_peer} is already in peers.")
                                    db.close()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to add: {e}")
                            for p in peers_list:
                                pt = p.get("ticker")
                                if not pt:
                                    continue
                                c1, c2 = st.columns([3, 1])
                                with c2:
                                    if st.button("Remove", key=f"remove_peer_{ticker_input}_{pt}"):
                                        try:
                                            db = get_db_session()
                                            from models import PeerOverride
                                            row = db.query(PeerOverride).filter(
                                                PeerOverride.focus_ticker == ticker_input,
                                                PeerOverride.peer_ticker == pt,
                                            ).first()
                                            if row:
                                                row.is_excluded = 1
                                            else:
                                                db.add(PeerOverride(focus_ticker=ticker_input, peer_ticker=pt, is_excluded=1))
                                            db.commit()
                                            db.close()
                                            clear_peers_cache(ticker_input)
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to remove: {e}")
                                with c1:
                                    st.caption(f"**{pt}** — {p.get('name', '')[:40]}")
                except Exception as e:
                    st.warning(f"Could not load competitors: {e}")
                
                # === NEWS (expander) ===
                try:
                    news_items = _cached_fetch_company_news(ticker_input, 10)
                    with st.expander("News", expanded=False):
                        if not news_items:
                            st.caption("No recent headlines.")
                        else:
                            for n in news_items:
                                headline = n.get("headline") or "No title"
                                url = n.get("url") or "#"
                                source = n.get("source") or ""
                                dt = n.get("datetime")
                                if isinstance(dt, (int, float)):
                                    from datetime import datetime as _dt
                                    try:
                                        dt = _dt.fromtimestamp(dt).strftime("%Y-%m-%d %H:%M") if dt else ""
                                    except Exception:
                                        dt = str(dt) if dt else ""
                                else:
                                    dt = str(dt) if dt else ""
                                if url:
                                    st.markdown(f"- [{headline[:80]}{'...' if len(headline) > 80 else ''}]({url})")
                                else:
                                    st.markdown(f"- {headline[:80]}{'...' if len(headline) > 80 else ''}")
                                if source or dt:
                                    st.caption(f"  {source} {dt}".strip())
                except Exception:
                    with st.expander("News", expanded=False):
                        st.caption("News unavailable.")
    
    # === WATCHLIST SECTION ===
    st.markdown("---")
    st.markdown("## 👁️ Watchlist Monitor")
    
    # Watchlist management
    with st.expander("➕ Manage Watchlist", expanded=False):
        add_col, remove_col = st.columns(2)
        
        with add_col:
            new_ticker = st.text_input(
                "Add Ticker to Watchlist",
                placeholder="e.g., NVDA",
                key="add_watchlist_ticker"
            ).upper().strip()
            
            alert_price = st.number_input(
                "Alert Price (optional)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="alert_price"
            )
            
            if st.button("➕ Add to Watchlist", use_container_width=True):
                if new_ticker:
                    try:
                        db = get_db_session()
                        existing = db.query(Watchlist).filter(Watchlist.ticker == new_ticker).first()
                        
                        if existing:
                            st.warning(f"{new_ticker} is already in your watchlist.")
                        else:
                            new_watch = Watchlist(
                                ticker=new_ticker,
                                alert_price=Decimal(str(alert_price)) if alert_price > 0 else None
                            )
                            db.add(new_watch)
                            db.commit()
                            st.success(f"✅ Added {new_ticker} to watchlist!")
                            st.rerun()
                        
                        db.close()
                    except Exception as e:
                        st.error(f"Error adding to watchlist: {e}")
        
        with remove_col:
            try:
                db = get_db_session()
                watchlist_items = db.query(Watchlist).all()
                db.close()
                
                if watchlist_items:
                    ticker_to_remove = st.selectbox(
                        "Remove Ticker from Watchlist",
                        options=[w.ticker for w in watchlist_items],
                        key="remove_watchlist_ticker"
                    )
                    
                    if st.button("🗑️ Remove from Watchlist", use_container_width=True):
                        try:
                            db = get_db_session()
                            db.query(Watchlist).filter(Watchlist.ticker == ticker_to_remove).delete()
                            db.commit()
                            db.close()
                            st.success(f"✅ Removed {ticker_to_remove} from watchlist!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing from watchlist: {e}")
            except Exception as e:
                st.info("Watchlist is empty. Add some tickers above.")
    
    # Display watchlist summary table
    try:
        db = get_db_session()
        watchlist_items = db.query(Watchlist).all()
        db.close()
        
        if watchlist_items:
            # Header with refresh button
            header_col1, header_col2 = st.columns([4, 1])
            with header_col1:
                st.markdown("### 📊 Watchlist Summary")
            with header_col2:
                if st.button("🔄 Refresh", help="Clear cached data and fetch fresh info", key="refresh_watchlist"):
                    # Clear ticker info cache for all watchlist items
                    cache_dir = _PROJECT_ROOT / ".market_cache"
                    if cache_dir.exists():
                        for item in watchlist_items:
                            info_cache = cache_dir / f"{item.ticker.upper()}_info.json"
                            if info_cache.exists():
                                info_cache.unlink()
                                st.toast(f"Cleared cache for {item.ticker}")
                    st.rerun()
            
            st.caption("Sorted by importance score (technical signals, RSI extremes, earnings proximity)")
            
            watchlist_data = []
            
            with st.spinner("Loading watchlist data..."):
                for item in watchlist_items:
                    alert_price = float(item.alert_price) if item.alert_price else None
                    summary = _cached_get_ticker_summary(item.ticker, alert_price=alert_price)
                    
                    if summary:
                        # Format earnings display
                        earnings_display = None
                        if summary.get('days_to_earnings') is not None:
                            days = summary['days_to_earnings']
                            if days <= 0:
                                earnings_display = "Today!"
                            elif days <= 7:
                                earnings_display = f"{days}d ⚠️"
                            elif days <= 14:
                                earnings_display = f"{days}d"
                            else:
                                earnings_display = f"{days}d"
                        elif summary.get('earnings_date'):
                            earnings_display = summary['earnings_date'][:10]  # Just the date part
                        
                        # Format volume vs average
                        vol_display = None
                        if summary.get('vol_vs_avg'):
                            vol_display = f"{summary['vol_vs_avg']:.1f}x"
                        
                        # Format short percent
                        short_display = None
                        if summary.get('short_percent'):
                            short_display = f"{summary['short_percent']:.1f}%"
                        
                        watchlist_data.append({
                            'Score': summary.get('importance_score', 0),
                            'Ticker': summary['ticker'],
                            'Price': summary['current_price'],
                            '3M %': summary.get('change_3m_pct'),
                            'RSI': summary.get('rsi'),
                            'BB%': summary.get('bb_pct'),
                            'Trend': summary.get('trend', 'N/A'),
                            'Signal': summary.get('signal', 'N/A'),
                            'P/E': summary.get('pe_ratio'),
                            'Fwd P/E': summary.get('forward_pe'),
                            'PEG': summary.get('peg_ratio'),
                            '52W %': summary.get('pct_from_52w_high'),
                            'Vol': vol_display,
                            'Cap': summary.get('market_cap_category'),
                            'Earnings': earnings_display,
                            'Short%': short_display,
                            'Target': summary.get('analyst_target'),
                            'Yield %': summary.get('dividend_yield'),
                            'Alert': alert_price,
                            # Store extra data for alerts (not displayed in table)
                            '_signal': summary.get('signal', 'N/A'),
                        })
                    else:
                        watchlist_data.append({
                            'Score': 0,
                            'Ticker': item.ticker,
                            'Price': None,
                            '3M %': None,
                            'RSI': None,
                            'BB%': None,
                            'Trend': 'N/A',
                            'Signal': 'N/A',
                            'P/E': None,
                            'Fwd P/E': None,
                            'PEG': None,
                            '52W %': None,
                            'Vol': None,
                            'Cap': None,
                            'Earnings': None,
                            'Short%': None,
                            'Target': None,
                            'Yield %': None,
                            'Alert': float(item.alert_price) if item.alert_price else None,
                            '_signal': 'N/A',
                        })
            
            if watchlist_data:
                # Sort by importance score (descending)
                watchlist_data.sort(key=lambda x: x.get('Score', 0), reverse=True)
                
                df_watchlist = pd.DataFrame(watchlist_data)
                
                # Remove internal columns before display
                display_columns = ['Score', 'Ticker', 'Price', '3M %', 'RSI', 'BB%', 'Trend', 'Signal', 
                                   'P/E', 'Fwd P/E', 'PEG', '52W %', 'Vol', 'Cap', 'Earnings', 
                                   'Short%', 'Target', 'Yield %', 'Alert']
                df_display = df_watchlist[display_columns]
                
                # Style the dataframe
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            help="Importance score based on technical signals, RSI, earnings proximity",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "3M %": st.column_config.NumberColumn(
                            "3M %",
                            help="3-month price change",
                            format="%.1f%%"
                        ),
                        "RSI": st.column_config.NumberColumn(
                            "RSI",
                            help="Relative Strength Index (14-day). <30 oversold, >70 overbought",
                            format="%.1f"
                        ),
                        "BB%": st.column_config.NumberColumn(
                            "BB%",
                            help="Bollinger Band position. <10 near lower band (oversold), >90 near upper (overbought)",
                            format="%.0f"
                        ),
                        "Trend": st.column_config.TextColumn(
                            "Trend",
                            help="Overall trend: BULLISH (SMA50 > SMA200), BEARISH, or NEUTRAL",
                            width="small"
                        ),
                        "Signal": st.column_config.TextColumn(
                            "Signal",
                            help="Technical signal: BUY, SELL, GOLDEN CROSS, DEATH CROSS, or HOLD",
                            width="medium"
                        ),
                        "P/E": st.column_config.NumberColumn(
                            "P/E",
                            help="Trailing Price-to-Earnings ratio",
                            format="%.1f"
                        ),
                        "Fwd P/E": st.column_config.NumberColumn(
                            "Fwd P/E",
                            help="Forward P/E based on expected earnings",
                            format="%.1f"
                        ),
                        "PEG": st.column_config.NumberColumn(
                            "PEG",
                            help="Price/Earnings to Growth ratio. <1 may be undervalued, >2 expensive",
                            format="%.2f"
                        ),
                        "52W %": st.column_config.NumberColumn(
                            "52W %",
                            help="Percent from 52-week high (negative = below high)",
                            format="%.1f%%"
                        ),
                        "Vol": st.column_config.TextColumn(
                            "Vol",
                            help="Today's volume vs 20-day average (e.g., 1.5x = 50% above normal)",
                            width="small"
                        ),
                        "Cap": st.column_config.TextColumn(
                            "Cap",
                            help="Market cap category: Mega ($200B+), Large ($10B+), Mid ($2B+), Small ($300M+), Micro",
                            width="small"
                        ),
                        "Earnings": st.column_config.TextColumn(
                            "Earnings",
                            help="Days until next earnings report",
                            width="small"
                        ),
                        "Short%": st.column_config.TextColumn(
                            "Short%",
                            help="Short interest as % of float. High (>10%) may indicate bearish sentiment or squeeze potential",
                            width="small"
                        ),
                        "Target": st.column_config.NumberColumn(
                            "Target",
                            help="Analyst consensus price target",
                            format="$%.2f"
                        ),
                        "Yield %": st.column_config.NumberColumn(
                            "Yield %",
                            help="Annual dividend yield",
                            format="%.2f%%"
                        ),
                        "Alert": st.column_config.NumberColumn(
                            "Alert",
                            help="Your price alert target",
                            format="$%.2f"
                        ),
                    }
                )
                
                # Alert notifications
                st.markdown("---")
                st.markdown("#### Alerts & Signals")
                
                has_alerts = False
                for item in watchlist_data:
                    alert_price = item.get('Alert')
                    current_price = item.get('Price')
                    signal = item.get('_signal', item.get('Signal', 'N/A'))
                    ticker = item.get('Ticker')
                    
                    # Price alert
                    if alert_price and current_price:
                        if current_price <= alert_price:
                            st.success(f"🔔 **{ticker}** has reached your alert price! Current: ${current_price:.2f}, Alert: ${alert_price:.2f}")
                            has_alerts = True
                    
                    # Signal alerts
                    if signal in ['BUY', 'GOLDEN CROSS']:
                        st.info(f"📈 **{ticker}** - {signal} signal detected!")
                        has_alerts = True
                    elif signal in ['SELL', 'DEATH CROSS']:
                        st.warning(f"📉 **{ticker}** - {signal} signal detected!")
                        has_alerts = True
                    
                    # Earnings imminent alert
                    earnings = item.get('Earnings')
                    if earnings and ('⚠️' in str(earnings) or earnings == 'Today!'):
                        st.warning(f"📅 **{ticker}** - Earnings {earnings.replace('⚠️', '').strip()}!")
                        has_alerts = True
                
                if not has_alerts:
                    st.caption("No active alerts at this time.")
                    
        else:
            st.info("📋 Your watchlist is empty. Add tickers above to monitor them.")
            
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")


def ipo_tracker_page():
    """Display IPO Vintage Tracker page with calendar, alerts, and performance analysis."""
    st.title("🚀 IPO Vintage Tracker")
    
    # Custom CSS for IPO tracker styling
    st.markdown("""
        <style>
        .ipo-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }
        .vintage-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 12px;
        }
        .vintage-1y { background-color: #3498db; color: white; }
        .vintage-2y { background-color: #9b59b6; color: white; }
        .vintage-3y { background-color: #f39c12; color: white; }
        .pending-badge { background-color: #7f8c8d; color: white; }
        .positive-return { color: #00ff41; }
        .negative-return { color: #ff073a; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Refresh button
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data"):
            clear_ipo_cache()
            st.cache_data.clear()
            st.rerun()
    
    # ==========================================
    # SECTION 1: THE HORIZON - Upcoming IPOs
    # ==========================================
    with st.expander("📅 **Section 1: The Horizon** - Upcoming IPOs", expanded=True):
        st.markdown("### Upcoming IPO Calendar (Next 30 Days)")
        
        with st.spinner("Loading IPO calendar..."):
            upcoming_ipos = _cached_fetch_ipo_calendar(30)
        
        if upcoming_ipos:
            # Create DataFrame for display
            ipo_data = []
            for ipo in upcoming_ipos:
                price_range = "TBD"
                if ipo.price_range_low and ipo.price_range_high:
                    price_range = f"${ipo.price_range_low:.2f} - ${ipo.price_range_high:.2f}"
                elif ipo.ipo_price:
                    price_range = f"${ipo.ipo_price:.2f}"
                
                shares_str = f"{ipo.shares_offered:,}" if ipo.shares_offered else "TBD"
                days_until = (ipo.ipo_date - date.today()).days
                
                ipo_data.append({
                    'Ticker': ipo.ticker,
                    'Company': ipo.name[:40] + '...' if len(ipo.name) > 40 else ipo.name,
                    'Exchange': ipo.exchange,
                    'Date': ipo.ipo_date.strftime('%Y-%m-%d'),
                    'Days Until': days_until,
                    'Price Range': price_range,
                    'Shares': shares_str,
                    'Status': ipo.status.title()
                })
            
            st.dataframe(
                ipo_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Exchange": st.column_config.TextColumn("Exchange", width="small"),
                    "Date": st.column_config.TextColumn("Listing Date", width="small"),
                    "Days Until": st.column_config.NumberColumn("Days Until", width="small"),
                    "Price Range": st.column_config.TextColumn("Price Range", width="small"),
                    "Shares": st.column_config.TextColumn("Shares Offered", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                }
            )
            
            # Follow IPO functionality
            st.markdown("---")
            st.markdown("#### ⭐ Follow an Upcoming IPO")
            st.caption("Save an IPO to your registry for vintage tracking once it goes public.")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                available_tickers = [ipo.ticker for ipo in upcoming_ipos if ipo.ticker != 'N/A']
                selected_ipo_ticker = st.selectbox(
                    "Select IPO to Follow",
                    options=available_tickers if available_tickers else ["No IPOs available"],
                    key="follow_ipo_select"
                )
            
            with col2:
                # Show details of selected IPO
                selected_ipo = next((ipo for ipo in upcoming_ipos if ipo.ticker == selected_ipo_ticker), None)
                if selected_ipo:
                    st.text_input("Company Name", value=selected_ipo.name, disabled=True, key="ipo_name_display")
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⭐ Follow IPO", use_container_width=True, type="primary"):
                    if selected_ipo:
                        try:
                            db = get_db_session()
                            
                            # Check if already following
                            existing = db.query(IPO_Registry).filter(
                                IPO_Registry.ticker == selected_ipo.ticker
                            ).first()
                            
                            if existing:
                                st.warning(f"Already following {selected_ipo.ticker}")
                            else:
                                # Determine IPO price
                                ipo_price = None
                                if selected_ipo.ipo_price:
                                    ipo_price = selected_ipo.ipo_price
                                elif selected_ipo.price_range_low and selected_ipo.price_range_high:
                                    ipo_price = (selected_ipo.price_range_low + selected_ipo.price_range_high) / 2
                                
                                new_registry = IPO_Registry(
                                    ticker=selected_ipo.ticker,
                                    company_name=selected_ipo.name,
                                    ipo_date=selected_ipo.ipo_date,
                                    ipo_price=ipo_price,
                                    exchange=selected_ipo.exchange,
                                    shares_offered=selected_ipo.shares_offered,
                                    is_following=1,
                                    created_at=date.today()
                                )
                                db.add(new_registry)
                                db.commit()
                                st.success(f"✅ Now following {selected_ipo.ticker}!")
                                st.rerun()
                            
                            db.close()
                        except Exception as e:
                            st.error(f"Error following IPO: {e}")
        else:
            st.info("No upcoming IPOs found in the next 30 days, or API key not configured.")
            st.caption("Set FINNHUB_API_KEY in your .env file to fetch real IPO data.")
    
    # ==========================================
    # SECTION 2: VINTAGE ALERTS
    # ==========================================
    with st.expander("🔔 **Section 2: Vintage Alerts** - Anniversary Notifications", expanded=True):
        st.markdown("### IPOs Approaching Vintage Milestones")
        st.caption("Companies within ±10 days of their 1, 2, or 3-year IPO anniversary.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(IPO_Registry.is_following == 1).all()
            db.close()
            
            if ipo_registries:
                # Convert to dicts for anniversary check
                registry_dicts = [
                    {
                        'ticker': r.ticker,
                        'ipo_date': r.ipo_date,
                        'company_name': r.company_name or r.ticker
                    }
                    for r in ipo_registries
                ]
                
                alerts = check_vintage_anniversaries(registry_dicts, days_threshold=10)
                
                if alerts:
                    for alert in alerts:
                        # Determine styling based on anniversary type
                        if alert['anniversary_years'] == 1:
                            badge_class = "vintage-1y"
                            emoji = "🥇"
                        elif alert['anniversary_years'] == 2:
                            badge_class = "vintage-2y"
                            emoji = "🥈"
                        else:
                            badge_class = "vintage-3y"
                            emoji = "🥉"
                        
                        # Format message
                        days = alert['days_diff']
                        if days > 0:
                            time_msg = f"in {days} day{'s' if days != 1 else ''}"
                        elif days < 0:
                            time_msg = f"{abs(days)} day{'s' if abs(days) != 1 else ''} ago"
                        else:
                            time_msg = "TODAY!"
                        
                        # Display alert card
                        if alert['status'] == 'today':
                            st.success(f"""
                            {emoji} **{alert['ticker']}** ({alert['company_name']}) 
                            — Year {alert['anniversary_years']} Anniversary is **{time_msg}** 
                            (IPO: {alert['ipo_date'].strftime('%Y-%m-%d')})
                            """)
                        elif alert['status'] == 'upcoming':
                            st.info(f"""
                            {emoji} **{alert['ticker']}** ({alert['company_name']}) 
                            — Year {alert['anniversary_years']} Anniversary {time_msg}
                            (IPO: {alert['ipo_date'].strftime('%Y-%m-%d')})
                            """)
                        else:
                            st.warning(f"""
                            {emoji} **{alert['ticker']}** ({alert['company_name']}) 
                            — Year {alert['anniversary_years']} Anniversary was {time_msg}
                            (IPO: {alert['ipo_date'].strftime('%Y-%m-%d')})
                            """)
                else:
                    st.info("🎉 No vintage anniversaries within the next 10 days.")
            else:
                st.info("No IPOs in your registry. Follow some IPOs to receive vintage alerts!")
                
        except Exception as e:
            st.error(f"Error loading vintage alerts: {e}")
    
    # ==========================================
    # SECTION 3: PERFORMANCE REVIEW - Leaderboard
    # ==========================================
    with st.expander("🏆 **Section 3: Performance Review** - Vintage Leaderboard", expanded=True):
        st.markdown("### IPO Performance Leaderboard")
        st.caption("Ranking your followed IPOs by total return since debut.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(IPO_Registry.is_following == 1).all()
            db.close()
            
            if ipo_registries:
                leaderboard_data = []
                
                with st.spinner("Calculating vintage performance..."):
                    for registry in ipo_registries:
                        # Skip future IPOs
                        if registry.ipo_date > date.today():
                            continue
                        
                        ipo_price = float(registry.ipo_price) if registry.ipo_price else None
                        vintage = get_vintage_performance(
                            registry.ticker,
                            registry.ipo_date,
                            ipo_price
                        )
                        
                        if vintage:
                            leaderboard_data.append({
                                'Ticker': vintage.ticker,
                                'Company': registry.company_name or vintage.ticker,
                                'IPO Date': vintage.ipo_date.strftime('%Y-%m-%d'),
                                'IPO Price': vintage.ipo_price,
                                'Current Price': vintage.current_price,
                                'Total Return %': vintage.total_return,
                                '1Y Return %': vintage.year_1_return if vintage.year_1_status == "Calculated" else None,
                                '2Y Return %': vintage.year_2_return if vintage.year_2_status == "Calculated" else None,
                                '3Y Return %': vintage.year_3_return if vintage.year_3_status == "Calculated" else None,
                                '1Y Status': vintage.year_1_status,
                                '2Y Status': vintage.year_2_status,
                                '3Y Status': vintage.year_3_status,
                            })
                
                if leaderboard_data:
                    # Sort by total return descending
                    leaderboard_data.sort(key=lambda x: x['Total Return %'] or 0, reverse=True)
                    
                    # Add rank
                    for i, entry in enumerate(leaderboard_data, 1):
                        entry['Rank'] = i
                    
                    # Create display DataFrame
                    display_data = []
                    for entry in leaderboard_data:
                        # Format returns with color indicators
                        total_ret = entry['Total Return %']
                        total_str = f"{total_ret:+.2f}%" if total_ret is not None else "N/A"
                        
                        def format_vintage(ret, status):
                            if status == "Pending":
                                return "⏳ Pending"
                            elif ret is not None:
                                return f"{ret:+.2f}%"
                            return "N/A"
                        
                        # Rank emoji
                        rank_emoji = "🥇" if entry['Rank'] == 1 else "🥈" if entry['Rank'] == 2 else "🥉" if entry['Rank'] == 3 else f"#{entry['Rank']}"
                        
                        display_data.append({
                            'Rank': rank_emoji,
                            'Ticker': entry['Ticker'],
                            'Company': entry['Company'][:25] + '...' if len(entry['Company']) > 25 else entry['Company'],
                            'IPO Date': entry['IPO Date'],
                            'IPO Price': f"${entry['IPO Price']:.2f}",
                            'Current': f"${entry['Current Price']:.2f}",
                            'Total Return': total_str,
                            '1Y': format_vintage(entry['1Y Return %'], entry['1Y Status']),
                            '2Y': format_vintage(entry['2Y Return %'], entry['2Y Status']),
                            '3Y': format_vintage(entry['3Y Return %'], entry['3Y Status']),
                        })
                    
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.TextColumn("🏆", width="small"),
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Company": st.column_config.TextColumn("Company", width="medium"),
                            "IPO Date": st.column_config.TextColumn("IPO Date", width="small"),
                            "IPO Price": st.column_config.TextColumn("IPO $", width="small"),
                            "Current": st.column_config.TextColumn("Now $", width="small"),
                            "Total Return": st.column_config.TextColumn("Total %", width="small"),
                            "1Y": st.column_config.TextColumn("1Y", width="small"),
                            "2Y": st.column_config.TextColumn("2Y", width="small"),
                            "3Y": st.column_config.TextColumn("3Y", width="small"),
                        }
                    )
                    
                    # Summary stats
                    st.markdown("---")
                    st.markdown("#### 📊 Portfolio Summary")
                    
                    total_returns = [e['Total Return %'] for e in leaderboard_data if e['Total Return %'] is not None]
                    if total_returns:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best Performer", f"{leaderboard_data[0]['Ticker']}", f"{leaderboard_data[0]['Total Return %']:+.2f}%")
                        with col2:
                            avg_return = sum(total_returns) / len(total_returns)
                            st.metric("Average Return", f"{avg_return:+.2f}%")
                        with col3:
                            winners = len([r for r in total_returns if r > 0])
                            st.metric("Winners", f"{winners}/{len(total_returns)}")
                        with col4:
                            st.metric("Total IPOs Tracked", len(leaderboard_data))
                else:
                    st.info("No performance data available yet. IPOs may not have started trading.")
            else:
                st.info("No IPOs in your registry. Follow some IPOs to track their vintage performance!")
                
        except Exception as e:
            st.error(f"Error loading leaderboard: {e}")
    
    # ==========================================
    # SECTION 4: VIBE CHART - IPO Trajectory Comparison
    # ==========================================
    with st.expander("📈 **Vibe Chart** - IPO Debut Trajectory Comparison", expanded=True):
        st.markdown("### Compare IPO Price Trajectories")
        st.caption("Overlay multiple IPOs aligned by their 'Day 0' (listing date) to compare debut performance.")
        
        try:
            db = get_db_session()
            ipo_registries = db.query(IPO_Registry).filter(
                IPO_Registry.is_following == 1,
                IPO_Registry.ipo_date <= date.today()
            ).all()
            db.close()
            
            if ipo_registries and len(ipo_registries) >= 1:
                # Let user select up to 3 IPOs to compare
                available_tickers = [r.ticker for r in ipo_registries]
                
                selected_tickers = st.multiselect(
                    "Select IPOs to Compare (max 3)",
                    options=available_tickers,
                    default=available_tickers[:min(3, len(available_tickers))],
                    max_selections=3,
                    key="vibe_chart_select"
                )
                
                # Days to show
                days_to_show = st.slider(
                    "Trading Days to Display",
                    min_value=30,
                    max_value=365,
                    value=90,
                    step=30,
                    key="vibe_days_slider"
                )
                
                if selected_tickers:
                    with st.spinner("Loading price trajectories..."):
                        # Collect data for selected IPOs
                        chart_data = []
                        colors = ['#00ff41', '#58a6ff', '#f78166', '#d2a8ff']
                        
                        for i, ticker in enumerate(selected_tickers):
                            registry = next((r for r in ipo_registries if r.ticker == ticker), None)
                            if registry:
                                df = get_ipo_price_history(ticker, registry.ipo_date, days=days_to_show)
                                if df is not None:
                                    df['Color'] = colors[i % len(colors)]
                                    chart_data.append(df)
                        
                        if chart_data:
                            # Create Plotly figure
                            fig = go.Figure()
                            
                            for i, df in enumerate(chart_data):
                                ticker = df['Ticker'].iloc[0]
                                
                                fig.add_trace(go.Scatter(
                                    x=df['Day'],
                                    y=df['Normalized'],
                                    mode='lines',
                                    name=ticker,
                                    line=dict(width=2.5),
                                    hovertemplate=(
                                        f"<b>{ticker}</b><br>"
                                        "Day %{x}<br>"
                                        "Performance: %{y:.1f}%<br>"
                                        "<extra></extra>"
                                    )
                                ))
                            
                            # Add baseline at 100 (IPO price)
                            fig.add_hline(
                                y=100,
                                line_dash="dash",
                                line_color="#8b949e",
                                annotation_text="IPO Price (Day 0)",
                                annotation_position="right"
                            )
                            
                            # Styling
                            fig.update_layout(
                                title=dict(
                                    text="<b>IPO Vibe Chart</b> - Debut Trajectory Comparison",
                                    font=dict(size=20, color='#c9d1d9'),
                                    x=0.5
                                ),
                                paper_bgcolor='#0d1117',
                                plot_bgcolor='#0d1117',
                                font=dict(color='#c9d1d9', family='Monaco, monospace'),
                                legend=dict(
                                    bgcolor='rgba(0,0,0,0.5)',
                                    bordercolor='#21262d',
                                    borderwidth=1,
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='center',
                                    x=0.5
                                ),
                                xaxis=dict(
                                    title="Trading Days Since IPO",
                                    gridcolor='#21262d',
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    title="Normalized Price (100 = IPO Price)",
                                    gridcolor='#21262d',
                                    zeroline=False
                                ),
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show legend/interpretation
                            st.markdown("""
                            **How to Read This Chart:**
                            - All IPOs are **normalized** to start at 100 (their IPO price)
                            - A value of **150** means the stock is up **50%** from IPO
                            - A value of **75** means the stock is down **25%** from IPO
                            - Compare how different IPOs performed in their first days/months of trading
                            """)
                        else:
                            st.warning("Could not load price data for selected IPOs.")
                else:
                    st.info("Select at least one IPO to view the trajectory chart.")
            else:
                st.info("No historical IPOs in your registry. Follow IPOs that have already listed to compare their trajectories.")
                
        except Exception as e:
            st.error(f"Error loading vibe chart: {e}")
    
    # ==========================================
    # MANAGE REGISTRY
    # ==========================================
    st.markdown("---")
    st.markdown("### 📋 Manage IPO Registry")
    
    with st.expander("View & Edit Your IPO Registry", expanded=False):
        try:
            db = get_db_session()
            all_registries = db.query(IPO_Registry).all()
            db.close()
            
            if all_registries:
                # Display registry
                registry_data = []
                for r in all_registries:
                    registry_data.append({
                        'ID': r.id,
                        'Ticker': r.ticker,
                        'Company': r.company_name or 'Unknown',
                        'IPO Date': r.ipo_date.strftime('%Y-%m-%d') if r.ipo_date else 'N/A',
                        'IPO Price': f"${float(r.ipo_price):.2f}" if r.ipo_price else 'N/A',
                        'Exchange': r.exchange or 'N/A',
                        'Following': '✅' if r.is_following else '❌'
                    })
                
                st.dataframe(registry_data, use_container_width=True, hide_index=True)
                
                # Add manual entry form
                st.markdown("---")
                st.markdown("#### ➕ Add IPO Manually")
                
                with st.form("manual_ipo_form", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        manual_ticker = st.text_input("Ticker Symbol", placeholder="e.g., UBER").upper()
                        manual_name = st.text_input("Company Name", placeholder="e.g., Uber Technologies Inc.")
                        manual_ipo_date = st.date_input("IPO Date")
                    
                    with col2:
                        manual_ipo_price = st.number_input("IPO Price ($)", min_value=0.01, step=0.01, format="%.2f")
                        manual_exchange = st.selectbox("Exchange", options=["NYSE", "NASDAQ", "AMEX", "Other"])
                    
                    if st.form_submit_button("Add to Registry", use_container_width=True):
                        if manual_ticker and manual_ipo_date:
                            try:
                                db = get_db_session()
                                
                                # Check if exists
                                existing = db.query(IPO_Registry).filter(IPO_Registry.ticker == manual_ticker).first()
                                if existing:
                                    st.warning(f"{manual_ticker} already exists in registry.")
                                else:
                                    new_entry = IPO_Registry(
                                        ticker=manual_ticker,
                                        company_name=manual_name if manual_name else None,
                                        ipo_date=manual_ipo_date,
                                        ipo_price=manual_ipo_price if manual_ipo_price > 0 else None,
                                        exchange=manual_exchange,
                                        is_following=1,
                                        created_at=date.today()
                                    )
                                    db.add(new_entry)
                                    db.commit()
                                    st.success(f"✅ Added {manual_ticker} to registry!")
                                    st.rerun()
                                
                                db.close()
                            except Exception as e:
                                st.error(f"Error adding IPO: {e}")
                        else:
                            st.error("Ticker and IPO Date are required.")
                
                # Remove entry
                st.markdown("---")
                st.markdown("#### 🗑️ Remove from Registry")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    ticker_to_remove = st.selectbox(
                        "Select IPO to Remove",
                        options=[r['Ticker'] for r in registry_data],
                        key="remove_ipo_select"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑️ Remove", use_container_width=True):
                        try:
                            db = get_db_session()
                            db.query(IPO_Registry).filter(IPO_Registry.ticker == ticker_to_remove).delete()
                            db.commit()
                            db.close()
                            st.success(f"Removed {ticker_to_remove} from registry.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing: {e}")
            else:
                st.info("Your IPO registry is empty. Follow upcoming IPOs or add them manually.")
                
                # Quick add form for empty state
                st.markdown("#### ➕ Add Your First IPO")
                with st.form("first_ipo_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        first_ticker = st.text_input("Ticker", placeholder="e.g., RIVN").upper()
                        first_date = st.date_input("IPO Date")
                    with col2:
                        first_name = st.text_input("Company Name", placeholder="e.g., Rivian Automotive")
                        first_price = st.number_input("IPO Price ($)", min_value=0.01, value=78.00, step=0.01)
                    
                    if st.form_submit_button("Add IPO"):
                        if first_ticker:
                            try:
                                db = get_db_session()
                                new_entry = IPO_Registry(
                                    ticker=first_ticker,
                                    company_name=first_name,
                                    ipo_date=first_date,
                                    ipo_price=first_price,
                                    is_following=1,
                                    created_at=date.today()
                                )
                                db.add(new_entry)
                                db.commit()
                                db.close()
                                st.success(f"✅ Added {first_ticker}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
        except Exception as e:
            st.error(f"Error loading registry: {e}")


def partnerships_page():
    """Display Partnerships (8-K Watch) page: SEC EDGAR Item 1.01 filings and counterparties."""
    st.title("Partnerships (8-K Watch)")
    st.markdown(
        "We watch SEC 8-K **Item 1.01** filings for your watched companies and surface only "
        "**partnership- and strategic-type** events (e.g. strategic partnerships, collaborations, "
        "joint ventures, supply/license agreements, M&A). Routine financing (credit agreements, "
        "indentures, notes offerings) is filtered out so what appears here is more likely to move the stock."
    )
    st.markdown("---")

    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", key="partnerships_refresh", help="Fetch latest 8-K filings from SEC"):
            with st.spinner("Fetching SEC EDGAR data..."):
                try:
                    refresh_edgar_data(limit=50)
                    st.success("Data refreshed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    try:
        events = _cached_get_partnership_events(50)
    except Exception as e:
        st.error(f"Error loading partnership events: {e}")
        return

    if not events:
        st.info(
            "No partnership events yet. Click **Refresh** to fetch 8-K Item 1.01 filings "
            "from the SEC for your watched companies. The first run may take a minute."
        )
        return

    # Build table rows: Filer, Filing date, Type, Counterparty(ies), Interest, Link
    rows = []
    for ev in events:
        filer = f"{ev.get('filer_ticker', '')} ({ev.get('filer_name', '')[:30]}{'...' if len(ev.get('filer_name', '') or '') > 30 else ''})"
        filing_date = ev.get("filing_date") or ""
        relevance = ev.get("relevance_type") or "other"
        type_label = "Partnership" if relevance == "partnership" else "Other"
        counterparties = ev.get("counterparties") or []
        cp_display = ", ".join(c.get("name", "") for c in counterparties) if counterparties else "—"
        has_interest = any(c.get("is_interest") for c in counterparties)
        interest_badge = "Yes" if has_interest else "—"
        sec_url = ev.get("sec_url") or ""

        rows.append({
            "Filer (ticker)": filer,
            "Filing date": filing_date,
            "Type": type_label,
            "Counterparty(ies)": cp_display,
            "Interest": interest_badge,
            "Link": sec_url,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Filer (ticker)": st.column_config.TextColumn("Filer (ticker)", width="medium"),
            "Filing date": st.column_config.TextColumn("Filing date", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Counterparty(ies)": st.column_config.TextColumn("Counterparty(ies)", width="large"),
            "Interest": st.column_config.TextColumn("Interest", width="small"),
            "Link": st.column_config.LinkColumn("SEC filing", display_text="View", width="small"),
        }
    )

    with st.expander("About this data"):
        st.caption(
            "Data from SEC EDGAR. We only show 8-K Item 1.01 filings that look like **partnerships or "
            "strategic deals** (e.g. strategic partnership, collaboration, joint venture, supply/license, "
            "M&A). Routine financing (credit agreements, indentures, notes) is excluded. "
            "'Type' = Partnership (keyword match) or Other (ambiguous). "
            "'Interest' = counterparty matches your configured private-company list."
        )


def thirteenf_page():
    """Display 13F Institutional Holdings page: 13f.info-style data, compare quarters, by-CUSIP, overlap."""
    st.title("13F Institutional Holdings")
    st.markdown(
        "Quarterly 13F filings for selected institutions. View holdings (13f.info-style), "
        "compare two quarters to see adds/subtracts, find who holds a given CUSIP, or see which holdings are common across funds."
    )
    st.markdown("---")

    institution_options = [inst["name"] for inst in THIRTEENF_INSTITUTIONS]
    name_to_cik = {inst["name"]: inst["cik"] for inst in THIRTEENF_INSTITUTIONS}

    # Build quarter list: last 8 quarters (e.g. 2025 Q4 down to 2024 Q1)
    today = date.today()
    quarter_choices = []
    y, q = today.year, (today.month - 1) // 3 + 1
    for _ in range(8):
        quarter_choices.append((y, q, f"{y} Q{q}"))
        q -= 1
        if q < 1:
            q = 4
            y -= 1

    # Controls
    selected_names = st.multiselect(
        "Institutions",
        options=institution_options,
        default=institution_options[: min(5, len(institution_options))],
        key="thirteenf_institutions",
    )
    selected_ciks = [name_to_cik[n] for n in selected_names]
    quarter_labels = [q[2] for q in quarter_choices]
    quarter_by_label = {q[2]: (q[0], q[1]) for q in quarter_choices}

    if not selected_names:
        st.info("Select at least one institution.")
        return

    # Section 1: Single filing (13f.info-style)
    st.markdown("### Single filing")
    inst_single = st.selectbox("Institution", selected_names, key="thirteenf_single_inst")
    q_single = st.selectbox("Quarter", quarter_labels, key="thirteenf_single_q")
    cik_single = name_to_cik[inst_single]
    year_single, qtr_single = quarter_by_label[q_single]
    try:
        data_single = _cached_get_13f_holdings_by_quarter(cik_single, year_single, qtr_single)
        if data_single:
            st.markdown(f"**{data_single.get('filer_name', inst_single)}** — {q_single}")
            st.caption(
                f"Holdings as of {data_single.get('period_end', 'N/A')} | "
                f"Value ($000): {data_single.get('value_thousands', 0):,.0f} | "
                f"Num holdings: {data_single.get('num_holdings', 0)} | "
                f"Date filed: {data_single.get('filing_date', 'N/A')} | "
                f"Form: {data_single.get('form_type', 'N/A')}"
            )
            if data_single.get("sec_url"):
                st.markdown(f"[View on SEC]({data_single['sec_url']})")
            rows = []
            for h in data_single.get("holdings", []):
                rows.append({
                    "Sym": h.get("sym", "—"),
                    "Issuer Name": (h.get("issuer_name") or "")[:60],
                    "CUSIP": h.get("cusip", ""),
                    "Value ($000)": h.get("value_thousands", 0),
                    "%": h.get("pct", 0),
                    "Shares": h.get("shares", 0),
                    "Principal": h.get("principal_type", "SH"),
                    "Option": h.get("option_type", "") or "—",
                })
            if rows:
                df1 = pd.DataFrame(rows)
                st.dataframe(df1, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No 13F data for {inst_single} {q_single}.")
    except Exception as e:
        st.error(f"Error loading single filing: {e}")

    st.markdown("---")

    # Section 2: Compare two quarters
    with st.expander("Compare two quarters (adds/subtracts)", expanded=True):
        inst_compare = st.selectbox("Institution", selected_names, key="thirteenf_compare_inst")
        q_a = st.selectbox("Period A", quarter_labels, key="thirteenf_q_a")
        q_b = st.selectbox("Period B", quarter_labels, key="thirteenf_q_b")
        cik_c = name_to_cik[inst_compare]
        filings_c = _cached_get_13f_filings_for_institution(cik_c)
        acc_a = next((f["accession_number"] for f in filings_c if f["year"] == quarter_by_label[q_a][0] and f["quarter"] == quarter_by_label[q_a][1]), None)
        acc_b = next((f["accession_number"] for f in filings_c if f["year"] == quarter_by_label[q_b][0] and f["quarter"] == quarter_by_label[q_b][1]), None)
        if acc_a and acc_b and st.button("Compare", key="thirteenf_do_compare"):
            try:
                compare_data = _cached_get_13f_compare(cik_c, acc_a, acc_b)
                if compare_data:
                    st.caption(f"**{compare_data.get('filer_name', '')}** — Value ($000): {compare_data.get('value_a', 0):,.0f} → {compare_data.get('value_b', 0):,.0f}")
                    rows_c = []
                    for r in compare_data.get("rows", []):
                        rows_c.append({
                            "CUSIP": r.get("cusip", ""),
                            "Issuer Name": (r.get("issuer_name") or "")[:50],
                            "Opt": r.get("option_type", "") or "—",
                            "Shares A": r.get("shares_a", 0),
                            "Shares B": r.get("shares_b", 0),
                            "Diff Sh": r.get("diff_shares", 0),
                            "Chg %": r.get("chg_pct_shares", 0),
                            "Val A ($000)": r.get("value_a", 0),
                            "Val B ($000)": r.get("value_b", 0),
                            "Diff Val": r.get("diff_value", 0),
                        })
                    if rows_c:
                        st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not load compare data.")
            except Exception as e:
                st.error(f"Error: {e}")
        elif not acc_a or not acc_b:
            st.caption("Select different quarters that have filings.")

    # Section 3: By CUSIP (who holds)
    with st.expander("By CUSIP (who holds this security)"):
        cusip_input = st.text_input("CUSIP", placeholder="e.g. 037833100", key="thirteenf_cusip").strip().upper()
        q_cusip = st.selectbox("Quarter", quarter_labels, key="thirteenf_cusip_q")
        if cusip_input:
            year_c, qtr_c = quarter_by_label[q_cusip]
            try:
                holders = _cached_get_holders_by_cusip(cusip_input, tuple(selected_ciks), year_c, qtr_c)
                if holders:
                    rows_h = [{"Institution": h["filer_name"], "Shares": h["shares"], "Value ($000)": h["value_thousands"], "%": h["pct"]} for h in holders]
                    st.dataframe(pd.DataFrame(rows_h), use_container_width=True, hide_index=True)
                else:
                    st.info("No tracked institution holds this CUSIP in that quarter.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Section 4: Overlap
    with st.expander("Overlap (holdings common to selected funds)"):
        overlap_names = st.multiselect("Institutions to compare (2+)", selected_names, default=selected_names[:2], key="thirteenf_overlap")
        q_overlap = st.selectbox("Quarter", quarter_labels, key="thirteenf_overlap_q")
        if len(overlap_names) >= 2:
            year_o, qtr_o = quarter_by_label[q_overlap]
            overlap_ciks = [name_to_cik[n] for n in overlap_names]
            try:
                overlap_list = _cached_get_overlap_holdings(tuple(overlap_ciks), year_o, qtr_o)
                if overlap_list:
                    rows_o = [{"CUSIP": o["cusip"], "Issuer Name": (o.get("issuer_name") or "")[:60]} for o in overlap_list]
                    st.dataframe(pd.DataFrame(rows_o), use_container_width=True, hide_index=True)
                else:
                    st.info("No common holdings for these institutions in that quarter.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.caption("Select at least 2 institutions.")

    with st.expander("About this data"):
        st.caption(
            "Data from SEC EDGAR 13F-HR filings. Holdings are reported by CUSIP; ticker (Sym) is not provided by the SEC. "
            "Compare view shows quarter-over-quarter changes. Overlap shows securities held by all selected institutions in the chosen quarter."
        )


def main():
    """Main application entry point."""
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
        if initialize_database():
            st.session_state.db_initialized = True
    
    # Sidebar navigation
    st.sidebar.title("📈 Gavin Financial Terminal")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Portfolio & Taxes", "Market Analysis", "IPO Vintage Tracker", "Partnerships", "13F Holdings"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Personal financial intelligence platform for portfolio tracking, "
        "tax optimization, and market analysis."
    )
    
    # Route to appropriate page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Portfolio & Taxes":
        portfolio_taxes_page()
    elif page == "Market Analysis":
        market_analysis_page()
    elif page == "IPO Vintage Tracker":
        ipo_tracker_page()
    elif page == "Partnerships":
        partnerships_page()
    elif page == "13F Holdings":
        thirteenf_page()


if __name__ == "__main__":
    main()
