"""
Serializable Pydantic views of dashboard data-layer outputs for docs, export,
and future HTTP APIs. Plain dicts via model_dump(mode="json").
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from factor_exposure import FactorExposureResult
from macro_context import MacroContextResult
from options_iv_term import IVTermStructureResult
from portfolio_insights import PortfolioInsights
from tca_estimate import TCAEstimateResult


class MacroMoverSchema(BaseModel):
    symbol: str
    label: str
    last: Optional[float] = None
    previous_close: Optional[float] = None
    pct_change: Optional[float] = None
    error: Optional[str] = None
    realized_vol_20d: Optional[float] = None
    change_over_sigma: Optional[float] = None


class FredRateSchema(BaseModel):
    series_id: str
    title: str
    value: Optional[float] = None
    as_of: Optional[str] = None
    error: Optional[str] = None


class MacroContextSchema(BaseModel):
    movers: List[MacroMoverSchema]
    rates: List[FredRateSchema]
    fred_configured: bool
    errors: List[str]


class PortfolioInsightsSchema(BaseModel):
    sector_weights: Dict[str, float]
    industry_weights: Dict[str, float]
    top1_pct: float
    top5_pct: float
    herfindahl: float
    portfolio_beta: Optional[float] = None
    per_ticker_beta: Dict[str, float]
    beta_weights_used: Dict[str, float]
    data_warnings: List[str]


class FactorExposureSchema(BaseModel):
    portfolio_factor_betas: Dict[str, float]
    per_ticker_betas: Dict[str, Dict[str, float]]
    per_ticker_r2: Dict[str, float]
    per_ticker_n_obs: Dict[str, int]
    factor_names: List[str]
    as_of: Optional[str] = None
    regression_end: Optional[str] = None
    regression_start: Optional[str] = None
    data_warnings: List[str] = Field(default_factory=list)
    factors_available: bool = False


class IVTermPointSchema(BaseModel):
    expiry: str
    dte: int
    iv_atm: Optional[float] = None
    strike: Optional[float] = None
    source: str = "avg"


class IVTermStructureSchema(BaseModel):
    ticker: str
    spot_used: Optional[float] = None
    points: List[IVTermPointSchema] = Field(default_factory=list)
    data_warnings: List[str] = Field(default_factory=list)


class TCASchema(BaseModel):
    ticker: str
    side: str
    shares: float
    notional_usd: float
    adv_shares: float
    adv_dollar: float
    participation_rate: float
    daily_volatility: float
    annualized_volatility: float
    estimated_impact_frac: float
    estimated_impact_bps: float
    estimated_impact_usd: float
    price_ref: float
    data_warnings: List[str] = Field(default_factory=list)


def macro_context_to_schema(result: MacroContextResult) -> MacroContextSchema:
    return MacroContextSchema(
        movers=[
            MacroMoverSchema(
                symbol=m.symbol,
                label=m.label,
                last=m.last,
                previous_close=m.previous_close,
                pct_change=m.pct_change,
                error=m.error,
                realized_vol_20d=m.realized_vol_20d,
                change_over_sigma=m.change_over_sigma,
            )
            for m in result.movers
        ],
        rates=[
            FredRateSchema(
                series_id=r.series_id,
                title=r.title,
                value=r.value,
                as_of=r.as_of,
                error=r.error,
            )
            for r in result.rates
        ],
        fred_configured=result.fred_configured,
        errors=list(result.errors),
    )


def portfolio_insights_to_schema(ins: PortfolioInsights) -> PortfolioInsightsSchema:
    return PortfolioInsightsSchema(
        sector_weights=dict(ins.sector_weights),
        industry_weights=dict(ins.industry_weights),
        top1_pct=ins.top1_pct,
        top5_pct=ins.top5_pct,
        herfindahl=ins.herfindahl,
        portfolio_beta=ins.portfolio_beta,
        per_ticker_beta=dict(ins.per_ticker_beta),
        beta_weights_used=dict(ins.beta_weights_used),
        data_warnings=list(ins.data_warnings),
    )


def factor_exposure_to_schema(f: FactorExposureResult) -> FactorExposureSchema:
    return FactorExposureSchema(
        portfolio_factor_betas=dict(f.portfolio_factor_betas),
        per_ticker_betas={k: dict(v) for k, v in f.per_ticker_betas.items()},
        per_ticker_r2=dict(f.per_ticker_r2),
        per_ticker_n_obs=dict(f.per_ticker_n_obs),
        factor_names=list(f.factor_names),
        as_of=f.as_of,
        regression_end=f.regression_end,
        regression_start=f.regression_start,
        data_warnings=list(f.data_warnings),
        factors_available=f.factors_available,
    )


def iv_term_structure_to_schema(r: IVTermStructureResult) -> IVTermStructureSchema:
    return IVTermStructureSchema(
        ticker=r.ticker,
        spot_used=r.spot_used,
        points=[
            IVTermPointSchema(
                expiry=p.expiry,
                dte=p.dte,
                iv_atm=p.iv_atm,
                strike=p.strike,
                source=p.source,
            )
            for p in r.points
        ],
        data_warnings=list(r.data_warnings),
    )


def tca_to_schema(t: TCAEstimateResult) -> TCASchema:
    return TCASchema(
        ticker=t.ticker,
        side=t.side,
        shares=t.shares,
        notional_usd=t.notional_usd,
        adv_shares=t.adv_shares,
        adv_dollar=t.adv_dollar,
        participation_rate=t.participation_rate,
        daily_volatility=t.daily_volatility,
        annualized_volatility=t.annualized_volatility,
        estimated_impact_frac=t.estimated_impact_frac,
        estimated_impact_bps=t.estimated_impact_bps,
        estimated_impact_usd=t.estimated_impact_usd,
        price_ref=t.price_ref,
        data_warnings=list(t.data_warnings),
    )


def dump_json_ready(obj: Any) -> Any:
    """If obj is a known dataclass result, convert; else pass through dict."""
    if isinstance(obj, MacroContextResult):
        return macro_context_to_schema(obj).model_dump(mode="json")
    if isinstance(obj, PortfolioInsights):
        return portfolio_insights_to_schema(obj).model_dump(mode="json")
    if isinstance(obj, FactorExposureResult):
        return factor_exposure_to_schema(obj).model_dump(mode="json")
    if isinstance(obj, TCAEstimateResult):
        return tca_to_schema(obj).model_dump(mode="json")
    if isinstance(obj, IVTermStructureResult):
        return iv_term_structure_to_schema(obj).model_dump(mode="json")
    return obj


def build_dashboard_export_payload(
    macro: MacroContextResult,
    insights: Optional[PortfolioInsights] = None,
    factors: Optional[FactorExposureResult] = None,
    tca: Optional[TCAEstimateResult] = None,
) -> Dict[str, Any]:
    """Single JSON object for macro + optional portfolio analytics (TCA optional)."""
    out: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "macro": macro_context_to_schema(macro).model_dump(mode="json"),
    }
    if insights is not None:
        out["portfolio_insights"] = portfolio_insights_to_schema(insights).model_dump(mode="json")
    if factors is not None:
        out["factor_exposure"] = factor_exposure_to_schema(factors).model_dump(mode="json")
    if tca is not None:
        out["tca_estimate"] = tca_to_schema(tca).model_dump(mode="json")
    return out
