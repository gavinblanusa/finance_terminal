"""
Compose dashboard-style analytics dicts for REST (no Streamlit, no session TCA).
"""

from __future__ import annotations

from typing import Any, Dict

from data_schemas import build_dashboard_export_payload
from factor_exposure import build_factor_exposure
from macro_context import build_macro_context
from market_data import fetch_ohlcv, get_company_profile
from portfolio_insights import build_portfolio_insights
from portfolio_snapshot import fetch_portfolio_snapshot_dict, positions_for_insights


def build_rest_dashboard_payload(include_factors: bool = True) -> Dict[str, Any]:
    """
    Same logical content as the Dashboard JSON download (macro + insights + factors),
    without `tca_estimate` (no browser session). Portfolio prices refresh per request.
    """
    macro = build_macro_context()
    snap = fetch_portfolio_snapshot_dict()
    if snap is None:
        return {
            **build_dashboard_export_payload(macro, None, None, None),
            "portfolio_error": "database_or_engine_unavailable",
        }
    pos_list = positions_for_insights(snap)
    if not pos_list:
        return build_dashboard_export_payload(macro, None, None, None)
    insights = build_portfolio_insights(pos_list, get_company_profile, fetch_ohlcv)
    factors = (
        build_factor_exposure(pos_list, fetch_ohlcv, period_years=3)
        if include_factors
        else None
    )
    return build_dashboard_export_payload(macro, insights, factors, None)
