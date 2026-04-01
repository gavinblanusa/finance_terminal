"""
Single source of truth for OpenBB provider try-order (app + docs drift check).

Keep ``docs/OPENBB_COVERAGE.md`` aligned: run ``python scripts/verify_openbb_coverage_doc.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

# dataset_id -> providers to try in order (OpenBB provider names)
OPENBB_PROVIDER_CHAINS: Dict[str, Tuple[str, ...]] = {
    "equity.price.historical": ("polygon", "yfinance", "fmp"),
    "equity.price.quote": ("yfinance", "fmp", "polygon"),
    "equity.price.historical_point": ("polygon", "yfinance", "fmp"),
    "equity.profile": ("fmp", "yfinance", "intrinio"),
    "equity.fundamental.ratios": ("fmp", "intrinio"),
    "equity.fundamental.income": ("fmp", "intrinio", "polygon", "yfinance"),
    "news.company": ("fmp", "polygon", "yfinance", "tiingo", "intrinio"),
    "economy.fred_series": ("fred",),
}


def chain_arrow(providers: Tuple[str, ...]) -> str:
    """Human-readable chain as in OPENBB_COVERAGE.md (spaces around arrows)."""
    return " → ".join(providers)
