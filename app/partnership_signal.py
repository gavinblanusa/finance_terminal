"""
Pure scoring and copy helpers for Partnerships (8-K) signal view.
Testable without SEC or Streamlit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bump when scoring / enrichment outputs change (invalidates stale cached JSON rows).
SIGNAL_VERSION = 2


def format_display_excerpt(snippet: Optional[str], max_len: int = 220) -> str:
    if not snippet or not str(snippet).strip():
        return ""
    s = " ".join(str(snippet).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def filer_in_cap_band(market_cap: Optional[float], cap_min: float, cap_max: float) -> Optional[bool]:
    """None if cap unknown; True/False if comparable."""
    if market_cap is None or market_cap <= 0:
        return None
    return cap_min <= float(market_cap) <= cap_max


def _name_matches_fragment(corp_name: str, fragment: str) -> bool:
    """Conservative match: substring for longer fragments; tighter for very short tickers."""
    cn = (corp_name or "").lower().strip()
    frag = (fragment or "").lower().strip()
    if not cn or not frag:
        return False
    if len(frag) < 4:
        if cn == frag:
            return True
        for sep in (" ", ".", ",", "-", "("):
            if cn.startswith(frag + sep) or cn.endswith(sep + frag):
                return True
        return f" {frag} " in f" {cn} "
    return frag in cn


def resolve_counterparty_hits(
    counterparties: Sequence[Dict[str, Any]],
    interest_names: Sequence[str],
    aliases: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """
    Match counterparty display names against interest list + alias map.
    Returns (any_hit, sorted unique canonical labels).
    """
    matched: set[str] = set()
    for cp in counterparties:
        name = (cp.get("name") or "").strip()
        if not name:
            continue
        for canonical, syns in aliases.items():
            for syn in syns:
                if _name_matches_fragment(name, syn):
                    matched.add(canonical)
                    break
        for interest in interest_names:
            if _name_matches_fragment(name, interest):
                matched.add(interest)
    return (len(matched) > 0, sorted(matched))


def build_signal_reasons(
    relevance_type: str,
    interest_labels: Sequence[str],
    in_cap_band: Optional[bool],
    num_counterparties: int,
) -> List[str]:
    out: List[str] = []
    if relevance_type == "partnership":
        out.append("Strategic / partnership language in filing")
    elif relevance_type == "other":
        out.append("Item 1.01 (ambiguous vs financing)")
    if interest_labels:
        out.append("Interest counterparty: " + ", ".join(interest_labels[:3]))
        if len(interest_labels) > 3:
            out[-1] += "…"
    if in_cap_band is True:
        out.append("Filer market cap in configured band")
    elif in_cap_band is False:
        out.append("Filer market cap outside band")
    elif in_cap_band is None:
        out.append("Filer market cap unknown")
    if num_counterparties > 0:
        out.append(f"{num_counterparties} counterparty name(s) extracted")
    return out


def score_partnership_event(
    relevance_type: str,
    has_interest_hit: bool,
    in_cap_band: Optional[bool],
    num_counterparties: int,
) -> int:
    """
    0–100 sort score: interest hit and partnership type dominate; cap band nudges.
    """
    score = 0
    if relevance_type == "partnership":
        score += 38
    elif relevance_type == "other":
        score += 12
    if has_interest_hit:
        score += 42
    if in_cap_band is True:
        score += 18
    elif in_cap_band is False:
        score -= 8
    if num_counterparties > 0:
        score += min(4 + 2 * num_counterparties, 14)
    return max(0, min(100, score))


def enrich_event_dict(event: Dict[str, Any], filer_market_cap: Optional[float], cfg: Any) -> Dict[str, Any]:
    """
    Copy event and add signal_* fields. cfg is partnerships_config module.
    Counterparty dicts are updated in place (same objects as in event).
    """
    out = dict(event)
    cps = list(out.get("counterparties") or [])
    out["counterparties"] = cps
    interest_names = getattr(cfg, "COUNTERPARTY_INTEREST_NAMES", ())
    aliases = getattr(cfg, "COUNTERPARTY_ALIASES", {})
    for cp in cps:
        one_hit, _ = resolve_counterparty_hits([cp], interest_names, aliases)
        cp["is_interest"] = one_hit
    interest_hit, labels = resolve_counterparty_hits(cps, interest_names, aliases)
    cap_min = float(getattr(cfg, "FILER_CAP_USD_MIN", 0))
    cap_max = float(getattr(cfg, "FILER_CAP_USD_MAX", 0))
    in_band = filer_in_cap_band(filer_market_cap, cap_min, cap_max)
    rel = (out.get("relevance_type") or "other").lower()
    reasons = build_signal_reasons(rel, labels, in_band, len(cps))
    score = score_partnership_event(rel, interest_hit, in_band, len(cps))
    excerpt = format_display_excerpt(out.get("snippet"))
    out["filer_market_cap"] = filer_market_cap
    out["filer_in_cap_band"] = in_band
    out["interest_hit"] = interest_hit
    out["interest_labels"] = labels
    out["signal_score"] = score
    out["signal_reasons"] = reasons
    out["display_excerpt"] = excerpt
    out["signal_version"] = SIGNAL_VERSION
    return out
