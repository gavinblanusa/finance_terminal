"""
FinanceToolkit adapter for Gavin Financial Terminal.

Provides fundamentals (profitability ratios, revenue TTM) and optional current
valuation ratios (P/E, PEG) via FinanceToolkit. Returns None on any exception
or when FinanceToolkit is not installed so callers can fall back to OpenBB / FMP.

Feature flag: set USE_FINANCETOOLKIT=false to disable and use fallbacks only.
Uses FMP_API_KEY from environment when available.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Project root (app lives in Invest/app/)
_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    _env_path = _ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

USE_FINANCETOOLKIT = os.environ.get("USE_FINANCETOOLKIT", "true").lower() in ("true", "1", "yes")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

_Toolkit = None


def _get_toolkit(ticker: str):
    """Lazy load FinanceToolkit for one ticker. Returns Toolkit instance or None."""
    global _Toolkit
    if not USE_FINANCETOOLKIT:
        return None
    try:
        from financetoolkit import Toolkit
    except ImportError:
        return None
    try:
        # Per-ticker Toolkit with quarterly data for TTM-style ratios
        return Toolkit(
            tickers=[ticker.upper().strip()],
            api_key=FMP_API_KEY or None,
            quarterly=True,
        )
    except Exception as e:
        print(f"[FinanceToolkit] Toolkit init error for {ticker}: {e}")
        return None


def _safe_float(val: Any, round_to: int = 4) -> Optional[float]:
    """Convert to float for DB/UI; return None for NaN or invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val != val:  # NaN
            return None
        try:
            f = float(val)
            return round(f, round_to) if f == f else None
        except (TypeError, ValueError):
            return None
    return None


def fetch_fundamentals_financetoolkit(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamentals (revenue_ttm, margins, ROE, ROA) from FinanceToolkit.

    Returns dict with keys: ticker, revenue_ttm, gross_margin, operating_margin,
    net_margin, roe, roa, data_source. All ratios as decimals (0-1). None on failure.
    """
    if not USE_FINANCETOOLKIT:
        return None
    ticker = ticker.upper().strip()
    try:
        toolkit = _get_toolkit(ticker)
        if toolkit is None:
            return None

        # Profitability ratios (TTM with trailing=4 for quarterly)
        prof = toolkit.ratios.collect_profitability_ratios(trailing=4)
        if prof is None or (hasattr(prof, 'empty') and prof.empty):
            return None

        # FT doc: collect_profitability_ratios() returns DataFrame; .loc[ticker] gives
        # DataFrame with index = ratio names, columns = periods (years/quarters)
        import pandas as pd
        out = {
            "ticker": ticker,
            "revenue_ttm": None,
            "gross_margin": None,
            "operating_margin": None,
            "net_margin": None,
            "roe": None,
            "roa": None,
            "data_source": "financetoolkit",
        }

        ratio_map = [
            ("Gross Margin", "gross_margin"),
            ("Operating Margin", "operating_margin"),
            ("Net Profit Margin", "net_margin"),
            ("Return on Equity (ROE)", "roe"),
            ("Return on Assets (ROA)", "roa"),
        ]

        if isinstance(prof, pd.DataFrame) and not prof.empty:
            # Get slice for this ticker (doc: profitability_ratios.loc['AAPL'])
            ticker_data = prof.loc[ticker] if ticker in prof.index else None
            if ticker_data is None and len(prof.index) > 0:
                # Single-ticker Toolkit may return index = ratio names, columns = periods
                ticker_data = prof
            if ticker_data is not None:
                if isinstance(ticker_data, pd.Series):
                    # One row: treat as ratio name -> value (last period)
                    v = _safe_float(ticker_data.dropna().iloc[-1] if len(ticker_data.dropna()) > 0 else None)
                    if v is not None:
                        for ft_name, our_key in ratio_map:
                            if ft_name in str(ticker_data.name):
                                out[our_key] = v
                                break
                elif isinstance(ticker_data, pd.DataFrame):
                    # Rows = ratio names, columns = periods; take last valid period per ratio
                    for ft_name, our_key in ratio_map:
                        if ft_name in ticker_data.index:
                            ser = ticker_data.loc[ft_name]
                            if isinstance(ser, pd.Series):
                                last = ser.dropna()
                                if len(last) > 0:
                                    v = _safe_float(last.iloc[-1])
                                    if v is not None:
                                        out[our_key] = v

        # Revenue TTM from income statement if available
        try:
            income = toolkit.get_income_statement()
            if income is not None and hasattr(income, 'empty') and not income.empty:
                # Structure varies; look for Revenue column and sum last 4 quarters
                if hasattr(income, 'columns'):
                    for col in income.columns:
                        if 'revenue' in str(col).lower() and 'growth' not in str(col).lower():
                            if ticker in income.index:
                                rev_series = income.loc[ticker, col] if hasattr(income.loc[ticker], '__getitem__') else income[col]
                            else:
                                rev_series = income[col] if col in income.columns else None
                            if rev_series is not None and hasattr(rev_series, 'dropna'):
                                rev_series = rev_series.dropna()
                                if len(rev_series) >= 1:
                                    total = rev_series.astype(float).tail(4).sum()
                                    if total and total > 0:
                                        out["revenue_ttm"] = int(round(total))
                            break
        except Exception:
            pass

        # Consider success if we have at least one ratio (revenue optional)
        if any(out.get(k) is not None for k in ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]):
            return out
        return None

    except Exception as e:
        print(f"[FinanceToolkit] fetch_fundamentals_financetoolkit error for {ticker}: {e}")
        return None


def fetch_current_valuation_ratios_financetoolkit(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current P/E and PEG from FinanceToolkit for the valuation path.

    Returns dict with current_pe and/or peg_ratio (scalars), or None on failure.
    """
    if not USE_FINANCETOOLKIT:
        return None
    ticker = ticker.upper().strip()
    try:
        toolkit = _get_toolkit(ticker)
        if toolkit is None:
            return None

        import pandas as pd
        result = {}

        def _last_scalar(data, t: str):
            if data is None or (hasattr(data, 'empty') and data.empty):
                return None
            if isinstance(data, pd.DataFrame) and t in data.index:
                val = data.loc[t].dropna()
                return _safe_float(val.iloc[-1], round_to=2) if len(val) > 0 else None
            if isinstance(data, pd.Series):
                val = data.dropna()
                return _safe_float(val.iloc[-1], round_to=2) if len(val) > 0 else None
            return None

        try:
            pe = toolkit.ratios.get_price_to_earnings_ratio()
            v = _last_scalar(pe, ticker)
            if v is not None:
                result["current_pe"] = v
        except Exception:
            pass

        try:
            peg = toolkit.ratios.get_price_to_earnings_growth_ratio()
            v = _last_scalar(peg, ticker)
            if v is not None:
                result["peg_ratio"] = v
        except Exception:
            pass

        return result if result else None

    except Exception as e:
        print(f"[FinanceToolkit] fetch_current_valuation_ratios_financetoolkit error for {ticker}: {e}")
        return None
