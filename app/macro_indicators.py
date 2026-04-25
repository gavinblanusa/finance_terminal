"""
Macro metric registry, band rules, and health evaluation for the Macro Dashboard.

R/Y/G labels are heuristics for quick scanning, not investment advice. See RULE_SET_VERSION.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

# Bump when default bands change; shown in UI methodology.
RULE_SET_VERSION = "1"

# DESIGN.md semantic: Positive, Negative, warn (amber) approximated as hex
PILL_STYLES = {
    "Healthy": {"bg": "#14532D", "fg": "#4ADE80", "border": "#4ADE80", "label": "OK"},
    "Teetering": {"bg": "#422006", "fg": "#E8A838", "border": "#E8A838", "label": "Watch"},
    "Unhealthy": {"bg": "#450A0A", "fg": "#F87171", "border": "#F87171", "label": "Stress"},
    "No Data": {"bg": "#262730", "fg": "#94A3B8", "border": "#94A3B8", "label": "N/A"},
}


@dataclass(frozen=True)
class MetricSpec:
    id: str
    title: str
    primer: str
    line_color: str
    fred_id: str | None  # None = derived in fetch layer (e.g. sahm)
    yaxis_hint: str
    y_log: bool = False  # log y when series has rare huge spikes (COVID) vs current level
    empty_detail: str | None = None  # extra help when the series is missing


# Order = display order
MACRO_DASHBOARD_METRICS: List[MetricSpec] = [
    MetricSpec(
        "gdp",
        "Nominal GDP (Bns)",
        "Gross Domestic Product. Two down quarters = technical recession context.",
        "#1f77b4",
        "GDP",
        "Index",
    ),
    MetricSpec("cpi", "CPI (headline)", "Fed targets ~2% YoY. Heuristic bands, not a forecast.", "#ff7f0e", "CPIAUCSL", "Index"),
    MetricSpec("unemployment", "Unemployment (%)", "BLS unemployment rate, monthly.", "#d62728", "UNRATE", "Percent"),
    MetricSpec("sahm", "Sahm rule (recession signal)", "3M avg unemployment minus 12M min of unemployment. ≥0.5 pp is the classic recession trigger; educational only.", "#F472B6", None, "pp"),
    MetricSpec("treasury_10y", "10Y Treasury (%)", "DGS10 daily. No intraday; see Dashboard for live curve context.", "#2ca02c", "DGS10", "Percent"),
    MetricSpec("nfci", "NFCI (fin. conditions)", "Chicago Fed National Financial Conditions. >0 = tighter than average; educational bands.", "#06B6D4", "NFCI", "Index"),
    MetricSpec("t10yie", "10Y breakeven inflation (%)", "T10YIE, market-implied long-run inflation. Not a policy forecast.", "#A78BFA", "T10YIE", "Percent"),
    MetricSpec("dfii10", "10Y TIPS (real yield %)", "DFII10. Real 10Y.", "#22D3EE", "DFII10", "Percent"),
    MetricSpec("init_claims", "Initial claims (4wk avg, persons)", "IC4WSA: 4-week moving average (persons). Lower is looser labor market stress.", "#FB923C", "IC4WSA", "Persons", y_log=True),
    MetricSpec("m2", "M2 (Bns)", "M2 money stock (seasonally adj.).", "#9467bd", "WM2NS", "Bns $", y_log=True),
    MetricSpec(
        "pmi",
        "ISM Mfg PMI",
        "ISM read; >50 expansion, <50 contraction.",
        "#8c564b",
        "M0204AM356PCEN",
        "Index",
        empty_detail=(
            "FRED `M0204AM356PCEN` is often **missing or moved**; confirm the series in "
            "the [FRED data tools](https://fred.stlouisfed.org/) or wire a different ISM / PMI source in the data layer if you need this read."
        ),
    ),
    MetricSpec("retail_sales", "Retail sales (Mns)", "Total retail, monthly.", "#e377c2", "RSAFS", "Mns $"),
    MetricSpec("consumer_sentiment", "U. Michigan sentiment", "Consumer survey index.", "#7f7f7f", "UMCSENT", "Index"),
]


def macro_metric_by_id(metric_id: str) -> Optional[MetricSpec]:
    for m in MACRO_DASHBOARD_METRICS:
        if m.id == metric_id:
            return m
    return None


def compute_sahm_series(unrate_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Sahm = 3M moving average of UNRATE minus 12M minimum of UNRATE (C. Sahm recession indicator, simplified).
    """
    if unrate_df is None or unrate_df.empty or "value" not in unrate_df.columns:
        return None
    s = unrate_df["value"].astype(float)
    n = len(s)
    if n < 3:
        return None
    out_val: List[float] = []
    idx = s.index
    for i in range(n):
        if i < 2:
            out_val.append(float("nan"))
            continue
        ma3 = float(s.iloc[i - 2 : i + 1].mean())
        j0 = max(0, i - 11)
        m12 = float(s.iloc[j0 : i + 1].min())
        out_val.append(ma3 - m12)
    ser = pd.Series(out_val, index=idx, name="value")
    return pd.DataFrame({"value": ser})


def _yoy_last(df: pd.DataFrame) -> Optional[float]:
    if df is None or len(df) < 5 or "value" not in df.columns:
        return None
    latest_val = float(df["value"].iloc[-1])
    one_yr_ago = df.index[-1] - pd.DateOffset(years=1)
    closest_idx = int(df.index.get_indexer([one_yr_ago], method="nearest")[0])
    old_val = float(df["value"].iloc[closest_idx])
    if old_val == 0:
        return None
    return (latest_val - old_val) / old_val * 100.0


def evaluate_macro_metric(df: pd.DataFrame | None, metric_id: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Returns: status, value_str, pill_text_label (short), as_of (YYYY-MM-DD or None)
    status in Healthy, Teetering, Unhealthy, No Data
    """
    if df is None or df.empty or "value" not in df.columns:
        return "No Data", "N/A", "N/A", None
    as_of: Optional[str] = None
    try:
        as_of = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    except Exception:
        as_of = None
    try:
        latest = float(df["value"].iloc[-1])
    except Exception:
        return "No Data", "N/A", "N/A", as_of

    mid = metric_id
    yoy = _yoy_last(df)

    if mid == "gdp" and yoy is not None:
        vs = f"{yoy:+.1f}% YoY"
        if yoy >= 3.0:
            return "Healthy", vs, "OK", as_of
        if yoy >= 1.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "cpi" and yoy is not None:
        vs = f"{yoy:+.1f}% YoY"
        if yoy <= 2.5:
            return "Healthy", vs, "OK", as_of
        if yoy <= 3.5:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "unemployment":
        vs = f"{latest:.1f}%"
        if latest <= 4.0:
            return "Healthy", vs, "OK", as_of
        if latest <= 4.5:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "sahm":
        vs = f"{latest:.2f} pp"
        if latest < 0.3:
            return "Healthy", vs, "OK", as_of
        if latest < 0.5:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "treasury_10y":
        vs = f"{latest:.2f}%"
        if 2.0 <= latest <= 4.0:
            return "Healthy", vs, "OK", as_of
        if (4.1 <= latest <= 5.0) or (1.5 <= latest < 2.0):
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "nfci":
        vs = f"{latest:+.3f}"
        if latest <= 0.0:
            return "Healthy", vs, "OK", as_of
        if latest <= 0.5:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "t10yie":
        vs = f"{latest:.2f}%"
        if 1.4 <= latest <= 2.6:
            return "Healthy", vs, "OK", as_of
        if 1.0 <= latest < 1.4 or 2.6 < latest <= 3.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "dfii10":
        vs = f"{latest:+.2f}%"
        if -0.5 <= latest <= 1.5:
            return "Healthy", vs, "OK", as_of
        if (-1.0 <= latest < -0.5) or (1.5 < latest <= 2.5):
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "init_claims":
        # FRED IC4WSA is 4-wk moving average; values are **persons** (not 000s in the file).
        k = latest / 1000.0
        vs = f"{k:,.0f}k"  # read as thousands of persons
        if latest < 280_000.0:
            return "Healthy", vs, "OK", as_of
        if latest < 350_000.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "m2" and yoy is not None:
        vs = f"{yoy:+.1f}% YoY"
        if yoy >= 2.0:
            return "Healthy", vs, "OK", as_of
        if yoy >= 0.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "pmi":
        vs = f"{latest:.1f}"
        if latest >= 50.0:
            return "Healthy", vs, "OK", as_of
        if latest >= 45.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "retail_sales" and yoy is not None:
        vs = f"{yoy:+.1f}% YoY"
        if yoy >= 3.0:
            return "Healthy", vs, "OK", as_of
        if yoy >= 1.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    if mid == "consumer_sentiment":
        vs = f"{latest:.1f}"
        if latest >= 70.0:
            return "Healthy", vs, "OK", as_of
        if latest >= 60.0:
            return "Teetering", vs, "Watch", as_of
        return "Unhealthy", vs, "Stress", as_of

    return "No Data", "N/A", "N/A", as_of


def status_pill_html(status: str, short_label: str) -> str:
    stv = status if status in PILL_STYLES else "No Data"
    ps = PILL_STYLES[stv]
    lab = short_label if short_label and short_label != "N/A" else ps["label"]
    return (
        f'<span class="gft-macro-pill" style="display:inline-block;padding:2px 10px;border-radius:6px;'
        f'font-family:IBM Plex Sans, sans-serif;font-size:0.85rem;font-weight:600;'
        f'background:{ps["bg"]};color:{ps["fg"]};border:1px solid {ps["border"]};">{lab}</span>'
    )
