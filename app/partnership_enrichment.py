"""
Batch market-cap lookup and partnership signal enrichment for cached 8-K events.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import partnerships_config as cfg
from partnership_signal import enrich_event_dict


def fetch_market_caps_yf(tickers: List[str], delay_s: float = 0.05) -> Dict[str, Optional[float]]:
    """
    Best-effort market cap in USD via yfinance. Returns ticker -> cap or None.
    """
    import yfinance as yf

    out: Dict[str, Optional[float]] = {}
    seen: set[str] = set()
    for raw in tickers:
        t = str(raw or "").upper().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        cap: Optional[float] = None
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if hasattr(fi, "get"):
                    cap = fi.get("market_cap") or fi.get("marketCap")
                else:
                    cap = getattr(fi, "market_cap", None) or getattr(fi, "marketCap", None)
            if cap is None:
                info = tk.info or {}
                cap = info.get("marketCap") or info.get("enterpriseValue")
            if cap is not None:
                cap = float(cap)
                if cap <= 0:
                    cap = None
        except Exception:
            cap = None
        out[t] = cap
        time.sleep(delay_s)
    return out


def enrich_partnership_events(events: List[dict]) -> List[dict]:
    """Attach filer market cap, score, reasons, excerpt, interest flags to each event."""
    if not events:
        return []
    tickers = sorted({(e.get("filer_ticker") or "").upper() for e in events if e.get("filer_ticker")})
    caps = fetch_market_caps_yf(tickers)
    return [
        enrich_event_dict(
            e,
            caps.get((e.get("filer_ticker") or "").upper()),
            cfg,
        )
        for e in events
    ]
