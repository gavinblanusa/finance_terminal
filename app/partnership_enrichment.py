"""
Batch market-cap lookup and partnership signal enrichment for cached 8-K events.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import partnerships_config as cfg
from partnership_signal import enrich_event_dict

_ROOT = Path(__file__).resolve().parent.parent
# Overridable in tests via monkeypatch.
_CAPS_CACHE_PATH: Path = _ROOT / ".edgar_cache" / "partnership_filer_market_caps.json"
CAP_CACHE_SCHEMA_VERSION = 1
# Filer market cap is slow-moving; disk cache avoids repeat Yahoo traffic on each page load.
FILER_CAP_CACHE_TTL_HOURS = 24.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_cached_at(entry: dict) -> Optional[datetime]:
    s = entry.get("cached_at")
    if not s or not isinstance(s, str):
        return None
    try:
        raw = s.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _entry_cap_fresh(entry: dict, ttl_sec: float, now: datetime) -> Optional[float]:
    """Return cap if entry is within TTL and has a stored positive cap; else None."""
    ts = _parse_cached_at(entry)
    if ts is None:
        return None
    if (now - ts).total_seconds() > ttl_sec:
        return None
    cap = entry.get("market_cap_usd")
    if cap is None:
        return None
    try:
        v = float(cap)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _prune_entries(entries: Any, ttl_sec: float, now: datetime) -> Dict[str, dict]:
    """Drop stale rows so the cache file does not grow forever."""
    if not isinstance(entries, dict):
        return {}
    out: Dict[str, dict] = {}
    for k, v in entries.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        t = k.upper().strip()
        if not t:
            continue
        if _entry_cap_fresh(v, ttl_sec, now) is not None:
            out[t] = v
    return out


def _ensure_edgar_cache_dir() -> None:
    _CAPS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_caps_disk() -> Dict[str, dict]:
    path = _CAPS_CACHE_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    ent = data.get("entries")
    return ent if isinstance(ent, dict) else {}


def _save_caps_disk(entries: Dict[str, dict]) -> None:
    _ensure_edgar_cache_dir()
    path = _CAPS_CACHE_PATH
    payload = {
        "schema_version": CAP_CACHE_SCHEMA_VERSION,
        "updated": _utc_now().isoformat(),
        "entries": entries,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0)
    os.replace(tmp, path)


def _market_cap_for_ticker(ticker: str) -> Optional[float]:
    """
    Best-effort market cap in USD for one symbol via yfinance.
    Prefer fast_info; fall back to .info only when cap is missing (info is slower).
    """
    import yfinance as yf

    t = str(ticker or "").upper().strip()
    if not t:
        return None
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
            info = getattr(tk, "info", None) or {}
            if isinstance(info, dict):
                cap = info.get("marketCap") or info.get("enterpriseValue")
        if cap is not None:
            cap = float(cap)
            if cap <= 0:
                cap = None
    except Exception:
        return None
    return cap


def _fetch_caps_network(tickers: List[str], max_workers: int) -> Dict[str, Optional[float]]:
    if not tickers:
        return {}
    if len(tickers) == 1 or max_workers < 2:
        return {t: _market_cap_for_ticker(t) for t in tickers}
    out: Dict[str, Optional[float]] = {}
    workers = min(max_workers, len(tickers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_ticker = {pool.submit(_market_cap_for_ticker, t): t for t in tickers}
        for fut in as_completed(future_to_ticker):
            t = future_to_ticker[fut]
            try:
                out[t] = fut.result()
            except Exception:
                out[t] = None
    return out


def fetch_market_caps_yf(tickers: List[str], max_workers: int = 4) -> Dict[str, Optional[float]]:
    """
    Best-effort market cap in USD via yfinance. Returns ticker -> cap or None.
    Uses a disk cache (FILER_CAP_CACHE_TTL_HOURS) for successful lookups; does not cache misses.
    """
    seen: set[str] = set()
    unique: List[str] = []
    for raw in tickers:
        t = str(raw or "").upper().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        unique.append(t)
    if not unique:
        return {}

    ttl_sec = float(FILER_CAP_CACHE_TTL_HOURS) * 3600.0
    now = _utc_now()
    raw_entries = _load_caps_disk()
    pruned = _prune_entries(raw_entries, ttl_sec, now)

    hits: Dict[str, Optional[float]] = {}
    need: List[str] = []
    for t in unique:
        prev = pruned.get(t)
        if prev is not None:
            cap = _entry_cap_fresh(prev, ttl_sec, now)
            if cap is not None:
                hits[t] = cap
                continue
        need.append(t)

    fetched: Dict[str, Optional[float]] = {}
    if need:
        fetched = _fetch_caps_network(need, max_workers)

    merged = dict(pruned)
    raw_keys_norm = {str(k).upper().strip() for k in raw_entries if isinstance(k, str) and str(k).strip()}
    dirty = set(pruned.keys()) != raw_keys_norm

    for t, cap in fetched.items():
        if cap is not None and cap > 0:
            merged[t] = {
                "market_cap_usd": cap,
                "cached_at": now.isoformat(),
            }
            dirty = True

    if dirty:
        _save_caps_disk(merged)

    out: Dict[str, Optional[float]] = {}
    for t in unique:
        if t in hits:
            out[t] = hits[t]
        elif t in fetched:
            out[t] = fetched[t]
        else:
            out[t] = None
    return out


def enrich_partnership_signals_only(events: List[dict]) -> List[dict]:
    """Signal fields + excerpts with no Yahoo call (filer cap unknown)."""
    if not events:
        return []
    return [enrich_event_dict(e, None, cfg) for e in events]


def enrich_partnership_with_caps(events: List[dict]) -> List[dict]:
    """Yahoo market-cap batch + full signal row (used after signals-only or SEC refresh)."""
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


def enrich_partnership_events(events: List[dict]) -> List[dict]:
    """Full enrich in one step (SEC refresh path and non-deferred readers)."""
    return enrich_partnership_with_caps(events)
