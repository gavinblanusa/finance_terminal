"""
TOP-lite: merge company news across portfolio and watchlist tickers, score for relevance,
dedupe, and sort.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

NEWS_KEYWORDS = (
    "earnings",
    "guidance",
    "sec",
    "lawsuit",
    "fda",
    "recall",
    "bankruptcy",
    "merger",
    "acquisition",
    "upgrade",
    "downgrade",
    "investigation",
    "restructuring",
)

MAX_SYMBOLS = 25
LIMIT_LARGE_UNIVERSE = 3
LIMIT_SMALL_UNIVERSE = 6
LARGE_UNIVERSE_THRESHOLD = 12


@dataclass
class RankedNewsItem:
    score: int
    datetime: Optional[datetime]
    headline: str
    source: str
    url: str
    tickers_matched: List[str]
    portfolio_hits: List[str]
    watchlist_hits: List[str]


def _parse_dt(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if hasattr(raw, "to_pydatetime"):
        try:
            return raw.to_pydatetime()
        except Exception:
            pass
    if isinstance(raw, (int, float)):
        try:
            ts = int(raw)
            if ts > 1e12:
                ts //= 1000
            return datetime.utcfromtimestamp(ts)
        except (ValueError, OSError):
            return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def _keyword_bonus(text: str) -> int:
    if not text:
        return 0
    low = text.lower()
    for kw in NEWS_KEYWORDS:
        if len(kw) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", low):
                return 1
        elif kw in low:
            return 1
    return 0


def _ticker_tokens(tickers: Set[str]) -> Dict[str, str]:
    return {t.upper(): t for t in tickers if t}


def _matches_tickers(text: str, token_map: Dict[str, str]) -> List[str]:
    if not text or not token_map:
        return []
    upper = text.upper()
    found: List[str] = []
    for tok, display in token_map.items():
        if len(tok) <= 1:
            continue
        if re.search(rf"(?<![A-Z0-9]){re.escape(tok)}(?![A-Z0-9])", upper):
            found.append(display)
    return sorted(set(found))


def build_relevant_news(
    portfolio_tickers: List[str],
    watchlist_tickers: List[str],
    fetch_company_news: Callable[[str, int], List[Dict]],
) -> List[RankedNewsItem]:
    """
    fetch_company_news(ticker, limit) -> list of dicts with headline, url, source, datetime.
    """
    port_set = {t.upper().strip() for t in portfolio_tickers if t and str(t).strip()}
    watch_set = {t.upper().strip() for t in watchlist_tickers if t and str(t).strip()}
    watch_only = watch_set - port_set

    universe: List[str] = []
    seen: Set[str] = set()
    for t in list(port_set) + list(watch_set):
        if t not in seen:
            seen.add(t)
            universe.append(t)
    universe = universe[:MAX_SYMBOLS]

    if not universe:
        return []

    n = len(universe)
    per_limit = LIMIT_LARGE_UNIVERSE if n >= LARGE_UNIVERSE_THRESHOLD else LIMIT_SMALL_UNIVERSE

    token_map = _ticker_tokens(port_set | watch_set)

    raw_items: List[tuple[Dict, str]] = []
    for sym in universe:
        try:
            items = fetch_company_news(sym, per_limit) or []
            for it in items:
                raw_items.append((it, sym))
        except Exception:
            continue

    dedupe_keys: Set[str] = set()
    ranked: List[RankedNewsItem] = []

    for it, queried_sym in raw_items:
        headline = (it.get("headline") or it.get("title") or "").strip()
        url = (it.get("url") or it.get("link") or "").strip()
        source = (it.get("source") or it.get("publisher") or "").strip()
        summary = (it.get("summary") or it.get("text") or "").strip()
        dt = _parse_dt(it.get("datetime") or it.get("published") or it.get("date"))

        text_blob = f"{headline} {summary}"
        matched = _matches_tickers(text_blob, {k: v for k, v in token_map.items() if k in port_set | watch_set})
        if queried_sym.upper() not in [m.upper() for m in matched]:
            matched = sorted(set(matched + [queried_sym]))

        port_hits = [m for m in matched if m.upper() in port_set]
        watch_hits = [m for m in matched if m.upper() in watch_only]

        score = 0
        if port_hits:
            score += 2
        elif watch_hits:
            score += 1
        score += _keyword_bonus(text_blob)

        dedupe = url if url else hashlib.sha256(headline.encode("utf-8", errors="ignore")).hexdigest()[:16]
        dedupe_full = f"{source}|{dedupe}"
        if dedupe_full in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_full)

        ranked.append(
            RankedNewsItem(
                score=score,
                datetime=dt,
                headline=headline or "(no headline)",
                source=source,
                url=url,
                tickers_matched=matched,
                portfolio_hits=port_hits,
                watchlist_hits=watch_hits,
            )
        )

    ranked.sort(
        key=lambda x: (
            -x.score,
            -(x.datetime.timestamp() if x.datetime else 0.0),
        )
    )
    return ranked
