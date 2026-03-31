"""
Portfolio summary dict for Streamlit cache and REST API (no Streamlit dependency).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from db import get_db_session
from tax_engine import TaxEngine


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, Decimal):
            if value.is_nan() or value.is_infinite():
                return default
        return float(value)
    except Exception:
        return default


def fetch_portfolio_snapshot_dict() -> Optional[Dict[str, Any]]:
    """
    Same shape as the Streamlit dashboard `get_portfolio_data()` return value.
    Opens one DB session per call; close in finally.
    """
    try:
        db = get_db_session()
        try:
            engine = TaxEngine(db)
            summary = engine.get_portfolio_summary()
        finally:
            db.close()

        return {
            "total_value": safe_float(summary.total_value),
            "total_cost_basis": safe_float(summary.total_cost_basis),
            "total_unrealized_gain": safe_float(summary.total_unrealized_gain),
            "total_unrealized_gain_pct": safe_float(summary.total_unrealized_gain_pct),
            "short_term_gain": safe_float(summary.short_term_gain),
            "long_term_gain": safe_float(summary.long_term_gain),
            "positions": [
                {
                    "ticker": p.ticker,
                    "total_shares": safe_float(p.total_shares),
                    "total_cost_basis": safe_float(p.total_cost_basis),
                    "current_price": safe_float(p.current_price),
                    "current_value": safe_float(p.current_value),
                    "unrealized_gain": safe_float(p.unrealized_gain),
                    "unrealized_gain_pct": safe_float(p.unrealized_gain_pct),
                    "short_term_shares": safe_float(p.short_term_shares),
                    "short_term_gain": safe_float(p.short_term_gain),
                    "long_term_shares": safe_float(p.long_term_shares),
                    "long_term_gain": safe_float(p.long_term_gain),
                    "tax_lots": [
                        {
                            "trade_id": lot.trade_id,
                            "ticker": lot.ticker,
                            "purchase_date": lot.purchase_date.isoformat(),
                            "shares": safe_float(lot.shares),
                            "cost_basis": safe_float(lot.cost_basis),
                            "holding_days": lot.holding_days,
                            "is_long_term": lot.is_long_term,
                            "tax_status": lot.tax_status,
                            "days_until_long_term": lot.days_until_long_term,
                            "is_near_long_term": lot.is_near_long_term,
                        }
                        for lot in p.tax_lots
                    ],
                    "lots_near_long_term": [
                        {
                            "ticker": lot.ticker,
                            "shares": safe_float(lot.shares),
                            "purchase_date": lot.purchase_date.isoformat(),
                            "cost_basis": safe_float(lot.cost_basis),
                            "days_until_long_term": lot.days_until_long_term,
                        }
                        for lot in p.lots_near_long_term
                    ],
                }
                for p in summary.positions
            ],
            "urgent_lots": [
                {
                    "ticker": lot.ticker,
                    "shares": safe_float(lot.shares),
                    "purchase_date": lot.purchase_date.isoformat(),
                    "cost_basis": safe_float(lot.cost_basis),
                    "days_until_long_term": lot.days_until_long_term,
                }
                for lot in summary.urgent_lots
            ],
        }
    except Exception as e:
        import traceback

        print(f"Error fetching portfolio snapshot: {e}")
        print(traceback.format_exc())
        return None


def positions_for_insights(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """PORT / factor helpers: [{'ticker', 'current_value'}, ...]."""
    out: List[Dict[str, Any]] = []
    for p in snap.get("positions") or []:
        t = (p.get("ticker") or "").strip().upper()
        v = float(p.get("current_value") or 0)
        if t and v > 0:
            out.append({"ticker": t, "current_value": v})
    return out
