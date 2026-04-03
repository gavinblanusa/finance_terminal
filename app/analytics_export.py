"""
Compose dashboard-style analytics dicts for REST (no Streamlit, no session TCA).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Dict, Optional

from data_schemas import build_dashboard_export_payload
from factor_exposure import (
    FactorAttributionResult,
    build_factor_attribution,
    build_factor_exposure,
    load_ff5_factors,
    resolve_attribution_window,
)
from macro_context import build_macro_context
from market_data import fetch_ohlcv, get_company_profile
from portfolio_insights import build_portfolio_insights
from portfolio_snapshot import fetch_portfolio_snapshot_dict, positions_for_insights


def build_rest_dashboard_payload(
    include_factors: bool = True,
    attribution_preset: str = "21",
    attribution_start: Optional[date] = None,
    attribution_end: Optional[date] = None,
) -> Dict[str, Any]:
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
    factor_attr: Optional[FactorAttributionResult] = None
    preset_used: Optional[str] = None
    if include_factors:
        ff, w_ff = load_ff5_factors()
        if attribution_start is not None and attribution_end is not None:
            a0, a1, w_bounds = resolve_attribution_window(
                ff, "custom", attribution_start, attribution_end
            )
            preset_used = "custom"
        else:
            a0, a1, w_bounds = resolve_attribution_window(ff, attribution_preset)
            preset_used = attribution_preset.strip().lower() if attribution_preset else "21"
        if a0 is not None and a1 is not None:
            factor_attr = build_factor_attribution(pos_list, fetch_ohlcv, a0, a1, period_years=3)
            merged = list(w_ff) + list(w_bounds) + list(factor_attr.data_warnings)
            factor_attr = replace(factor_attr, data_warnings=merged)
    return build_dashboard_export_payload(
        macro,
        insights,
        factors,
        None,
        factor_attribution=factor_attr,
        factor_attribution_preset=preset_used,
    )
