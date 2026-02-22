"""
SEC EDGAR service for 8-K partnership tracking.

Watches configured public companies for Form 8-K Item 1.01 (Entry into a Material
Definitive Agreement), fetches filing content, and extracts counterparty names.
Highlights counterparties that match the configured interest list (e.g. private companies).
"""

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

from partnerships_config import WATCH_TICKERS, COUNTERPARTY_INTEREST_NAMES

# SEC requires a descriptive User-Agent with contact info
USER_AGENT = "GavinFinancialTerminal/1.0 (gavinblanusa@comcast.net)"
SEC_RATE_LIMIT_DELAY = 0.11  # ~9 req/sec to stay under 10
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Project root (caches live at Invest/ root)
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".edgar_cache"
SUBMISSIONS_CACHE_HOURS = 1
COMPANY_TICKERS_CACHE_HOURS = 24
EVENTS_CACHE_FILE = "partnership_events.json"

# Noise: exclude these when extracting counterparties (law firms, agents, subsidiaries, etc.)
COUNTERPARTY_NOISE = frozenset({
    "shareholder representative services",
    "shareholder representative services llc",
    "srs",
    "sec",
    "u.s. securities and exchange commission",
    "the company",
    "the registrant",
    "we",
    "our",
    "the issuer",
    "the borrower",
    "administrative agent",
    "collateral agent",
    "as administrative agent",
    "as collateral agent",
    "as agent",
    "emerging growth company",
    "indicate by check mark",
    "entry into a material definitive agreement",
    "material definitive agreement",
    "credit agreement",
    "credit facility",
    "revolving credit",
    "term loan",
    "indenture",
    "subsidiary guarantors",
    "the guarantors",
    "the lenders",
    "the parties",
    "bank of america",
    "jpmorgan chase",
    "deutsche bank",
    "mizuho bank",
    "credit suisse",
    "goldman sachs",
    "morgan stanley",
    "wells fargo",
    "citibank",
    "as lender",
    "receivables purchase",
    "tax receivable",
    "indemnification agreement",
    "the company,",
    "the registrant,",
})

# Phrases that indicate a routine financing filing (credit/indenture/notes) - we filter these out
FINANCING_SIGNALS = frozenset({
    "credit agreement",
    "credit facility",
    "revolving credit",
    "term loan",
    "term loan agreement",
    "indenture",
    "indenture supplement",
    "supplement to the indenture",
    "notes due",
    "senior notes",
    "convertible notes",
    "indemnification agreement",
    "indemnification ag",
    "tax receivable agreement",
    "tax receivable ag",
    "receivables purchase agreement",
    "receivables purchase ag",
    "credit ag",
    "as administrative agent",
    "as collateral agent",
    "administrative agent and collateral agent",
    "subsidiary guarantors",
    "guarantors party thereto",
    "as issuer",
    "as borrower",
})

# Phrases that indicate a partnership / strategic / stock-moving deal - we keep these
PARTNERSHIP_SIGNALS = frozenset({
    "strategic partnership",
    "strategic alliance",
    "partnership agreement",
    "partnership with",
    "collaboration agreement",
    "collaboration with",
    "joint venture",
    "joint venture agreement",
    "memorandum of understanding",
    "mou ",
    " teaming agreement",
    "teaming agreement",
    "license agreement",
    "licensing agreement",
    "supply agreement",
    "long-term supply",
    "definitive agreement",
    "definitive agreement to acquire",
    "definitive agreement to merge",
    "merger agreement",
    "acquisition agreement",
    "asset purchase agreement",
    "strategic agreement",
    "commercial agreement",
    "distribution agreement",
    "reseller agreement",
    "partnership to",
    "alliance with",
    "collaboration to",
})


def _headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _headers_html() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}


def _throttle():
    time.sleep(SEC_RATE_LIMIT_DELAY)


def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


def _cached_json(path: Path, max_age_hours: float) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if (time.time() - mtime) / 3600 > max_age_hours:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json(path: Path, data: dict):
    _ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities for plain-text parsing."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_ticker_to_cik() -> Dict[str, Tuple[str, str]]:
    """
    Return map ticker -> (cik_10digit, company_title).
    CIK is zero-padded to 10 digits. Caches company_tickers.json under .edgar_cache.
    """
    cache_path = CACHE_DIR / "company_tickers.json"
    cached = _cached_json(cache_path, COMPANY_TICKERS_CACHE_HOURS)
    if cached is not None:
        return _build_ticker_map(cached)

    _throttle()
    try:
        r = requests.get(COMPANY_TICKERS_URL, headers=_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        _ensure_cache_dir()
        _save_json(cache_path, data)
        return _build_ticker_map(data)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[EDGAR] Failed to fetch company tickers: {e}")
        return {}


def _build_ticker_map(data: dict) -> Dict[str, Tuple[str, str]]:
    out = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        cik = entry.get("cik_str")
        ticker = entry.get("ticker")
        title = entry.get("title") or ""
        if cik is not None and ticker:
            cik_str = str(cik).zfill(10)
            out[str(ticker).upper()] = (cik_str, title)
    return out


def _get_submissions_for_cik(cik: str, force_refresh: bool = False) -> Optional[dict]:
    cache_path = CACHE_DIR / f"submissions_{cik}.json"
    if not force_refresh:
        cached = _cached_json(cache_path, SUBMISSIONS_CACHE_HOURS)
        if cached is not None:
            return cached

    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    _throttle()
    try:
        r = requests.get(url, headers=_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        _save_json(cache_path, data)
        return data
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[EDGAR] Submissions failed for CIK {cik}: {e}")
        return None


def _parse_recent_8ks(submissions: dict) -> List[dict]:
    """From submissions JSON, return list of 8-K filings from recent (columnar) array."""
    filings = submissions.get("filings") or {}
    recent = filings.get("recent")
    if not recent or not isinstance(recent, dict):
        return []

    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_docs = recent.get("primaryDocument") or []

    n = len(forms)
    result = []
    for i in range(n):
        if (i < len(forms) and (forms[i] or "").strip().upper() == "8-K" and
                i < len(accession_numbers) and i < len(filing_dates)):
            acc = (accession_numbers[i] or "").strip()
            if not acc:
                continue
            result.append({
                "accessionNumber": acc,
                "filingDate": filing_dates[i] if i < len(filing_dates) else "",
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            })
    return result


def _accession_to_path(accession_number: str) -> str:
    """Convert accession number to URL path part (strip hyphens)."""
    return (accession_number or "").replace("-", "")


def _fetch_8k_index(cik: str, accession_number: str) -> Optional[str]:
    """Fetch the index HTML for a filing; return raw HTML or None."""
    path_part = _accession_to_path(accession_number)
    # Index filename is typically {accession}-index.htm
    index_name = f"{accession_number}-index.htm"
    url = f"{ARCHIVES_BASE}/{cik}/{path_part}/{index_name}"
    _throttle()
    try:
        r = requests.get(url, headers=_headers_html(), timeout=15)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"[EDGAR] Failed to fetch index {accession_number}: {e}")
        return None


def _filename_only(href: str) -> str:
    """Return only the filename (last path segment); strip query string and path."""
    s = (href or "").strip()
    # ix?doc=/Archives/edgar/data/CIK/ACC/filename.htm -> use path after doc=
    if "doc=" in s:
        idx = s.index("doc=") + 4
        s = s[idx:].strip()
    if "?" in s:
        s = s.split("?")[0].strip()
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.rstrip("/").split("/")[-1]
    return s


def _find_primary_doc_from_index(html: str, accession_number: str, primary_doc_hint: str) -> Optional[str]:
    """
    Parse index HTML to find the primary 8-K document filename.
    Handles: href="file.htm", href="/Archives/edgar/.../file.htm", and ix?doc=/Archives/.../file.htm.
    Returns only the filename (e.g. file.htm) for building the fetch URL.
    """
    candidates = []

    # 1) Direct .htm/.html links: href="something.htm" or href="/Archives/edgar/.../file.htm"
    pattern = re.compile(r'href="([^"]+\.(?:htm|html))(?:"|\?|$)', re.I)
    for m in pattern.findall(html):
        fn = _filename_only(m)
        if fn and fn.lower().endswith((".htm", ".html")) and "index" not in fn.lower():
            candidates.append((m, fn))

    # 2) IX (inline) links: href="ix?doc=/Archives/edgar/data/CIK/ACC/filename.htm"
    ix_pattern = re.compile(r'href="(ix\?doc=[^"]+\.(?:htm|html))"', re.I)
    for m in ix_pattern.findall(html):
        fn = _filename_only(m)
        if fn and fn.lower().endswith((".htm", ".html")) and "index" not in fn.lower():
            candidates.append((m, fn))

    if primary_doc_hint and primary_doc_hint.strip():
        hint = primary_doc_hint.strip().lower()
        for _, fn in candidates:
            if hint in fn.lower():
                return fn

    for _, fn in candidates:
        if "ex" not in fn.lower() or "8k" in fn.lower() or "8-k" in fn.lower():
            return fn
    return candidates[0][1] if candidates else None


def _fetch_8k_document(cik: str, accession_number: str, doc_filename: str) -> Optional[str]:
    """Fetch 8-K document; doc_filename must be filename only (e.g. file.htm)."""
    filename_only = _filename_only(doc_filename)
    path_part = _accession_to_path(accession_number)

    # If we got an absolute path (starts with /), use sec.gov + path
    if doc_filename.strip().startswith("/"):
        url = f"https://www.sec.gov{doc_filename.split('?')[0].strip()}"
    else:
        url = f"{ARCHIVES_BASE}/{cik}/{path_part}/{filename_only}"

    _throttle()
    try:
        r = requests.get(url, headers=_headers_html(), timeout=15)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"[EDGAR] Failed to fetch document {filename_only}: {e}")
        return None


def _is_item_101_filing(text: str) -> bool:
    """Return True if the 8-K text indicates Item 1.01 (Material Definitive Agreement)."""
    lower = text.lower()
    if "item 1.01" not in lower and "item 1.1" not in lower:
        return False
    if "entry into a material definitive agreement" in lower:
        return True
    if "material definitive agreement" in lower:
        return True
    return False


def _relevance_type(text: str) -> str:
    """
    Classify 8-K Item 1.01 as 'partnership' (strategic/stock-moving), 'financing' (routine), or 'other'.
    Only 'partnership' and sometimes 'other' should surface in the Partnerships tab; 'financing' is filtered out.
    """
    lower = text.lower()
    # Normalize: collapse whitespace so multi-word phrases match
    normalized = " ".join(lower.split())

    partnership_score = sum(1 for s in PARTNERSHIP_SIGNALS if s in normalized)
    financing_score = sum(1 for s in FINANCING_SIGNALS if s in normalized)

    # Strong financing signals (credit/indenture/notes/agent) -> treat as noise
    if financing_score >= 2:
        return "financing"
    if any(
        phrase in normalized
        for phrase in (
            "credit agreement",
            "credit facility",
            "revolving credit",
            "term loan",
            "indenture",
            "indenture supplement",
            "administrative agent",
            "collateral agent",
            "subsidiary guarantors",
            "tax receivable agreement",
            "receivables purchase agreement",
            "indemnification agreement",
        )
    ):
        return "financing"

    # Strong partnership/strategic signals -> keep
    if partnership_score >= 1:
        return "partnership"
    if any(
        phrase in normalized
        for phrase in (
            "strategic partnership",
            "partnership agreement",
            "collaboration agreement",
            "joint venture",
            "memorandum of understanding",
            "license agreement",
            "supply agreement",
            "merger agreement",
            "acquisition agreement",
            "definitive agreement to",
        )
    ):
        return "partnership"

    # Ambiguous: "definitive agreement" alone could be M&A or credit. If no clear financing, allow as 'other'
    if "definitive agreement" in normalized and financing_score == 0:
        return "other"
    # Default: likely routine (e.g. amendment to credit) -> filter out
    return "financing"


def _extract_counterparties(text: str, relevance_type: str = "partnership") -> List[str]:
    """
    Extract counterparty company names from 8-K text. Prefers partnership-like context;
    returns short, clean names and filters boilerplate.
    """
    lower = text.lower()
    seen = set()
    names = []

    def _clean(name: str) -> str:
        name = re.sub(r"\s+", " ", name).strip()
        # Truncate at first comma if it's clearly trailing boilerplate
        if "," in name and len(name) > 50:
            first = name.split(",")[0].strip()
            if any(s in first for s in ("Inc", "Corp", "LLC", "Ltd", "Co.", "L.P.")):
                return first
        return name

    def _accept(name: str) -> bool:
        if len(name) < 4 or len(name) > 80:
            return False
        key = name.lower()
        if key in seen or key in COUNTERPARTY_NOISE:
            return False
        # Reject phrase-like captures (e.g. "announced a strategic partnership with Acme Corp")
        if " with " in key or " partnership " in key or " agreement " in key:
            return False
        if key.startswith(("announced ", "entered into ", "the company ", "pursuant to ")):
            return False
        # Reject if it's mostly boilerplate
        if any(
            skip in key
            for skip in (
                "indicate by check",
                "emerging growth",
                "material definitive",
                "entry into",
                "the company",
                "the registrant",
                "administrative agent",
                "collateral agent",
                "subsidiary guarantor",
                "credit agreement",
                "indenture",
            )
        ):
            return False
        return True

    # Prefer: "entered into a [Partnership/Collaboration] agreement with X" or "partnership with X"
    for pattern in [
        r"(?:partnership|collaboration|alliance|agreement)\s+with\s+([A-Z][A-Za-z0-9\s,\.&\'-]{2,60}?)(?:\s*,\s*|\s+and\s+|\s+dated\s+|\s+pursuant|\s+\(|\.|$)",
        r"(?:between|among)\s+([A-Z][A-Za-z0-9\s,\.&\'-]{2,60}?)\s+and\s+",
        r"([A-Z][A-Za-z0-9\s,&\'-]{2,50}?(?:\s+(?:Inc\.|Corp\.|Corporation|Ltd\.|LLC|L\.P\.|LP|Co\.|Company|N\.V\.|S\.A\.|AG))[\s\.])",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(1).strip()
            name = _clean(raw)
            if _accept(name):
                seen.add(name.lower())
                names.append(name)

    # Exhibit-style: short all-caps company name (single line)
    for line in text.splitlines():
        line = line.strip()
        if 4 <= len(line) <= 60 and line.isupper():
            if any(s in line for s in ("INC", "CORP", "LLC", "LTD", "CO.", "LP")):
                if _accept(line):
                    seen.add(line.lower())
                    names.append(line)

    return names[:8]


def _check_interest(name: str) -> bool:
    """True if name matches any COUNTERPARTY_INTEREST_NAMES (case-insensitive, substring)."""
    n = (name or "").lower()
    for interest in COUNTERPARTY_INTEREST_NAMES:
        if (interest or "").lower() in n or n in (interest or "").lower():
            return True
    return False


def _process_8k(cik: str, filer_ticker: str, filer_name: str, accession_number: str,
                filing_date: str, primary_doc_hint: str) -> Optional[dict]:
    """
    Fetch 8-K content, detect Item 1.01, extract counterparties. Return event dict or None.
    """
    cache_path = CACHE_DIR / f"8k_{_accession_to_path(accession_number)}.json"
    cached = _cached_json(cache_path, SUBMISSIONS_CACHE_HOURS * 24)
    if cached is not None:
        return cached.get("event")

    index_html = _fetch_8k_index(cik, accession_number)
    if not index_html:
        return None

    doc_filename = _find_primary_doc_from_index(index_html, accession_number, primary_doc_hint)
    if not doc_filename:
        return None

    doc_html = _fetch_8k_document(cik, accession_number, doc_filename)
    if not doc_html:
        return None

    text = _strip_html(doc_html)
    if not _is_item_101_filing(text):
        return None

    relevance = _relevance_type(text)
    # Filter out routine financing (credit/indenture/notes) so only partnership/strategic events surface
    if relevance == "financing":
        return None

    counterparty_raw = _extract_counterparties(text, relevance)
    counterparties = []
    for name in counterparty_raw:
        counterparties.append({"name": name, "is_interest": _check_interest(name)})

    path_part = _accession_to_path(accession_number)
    sec_url = f"{ARCHIVES_BASE}/{cik}/{path_part}/{doc_filename}"

    snippet = ""
    for phrase in ("strategic partnership", "partnership agreement", "collaboration", "joint venture", "agreement with"):
        if phrase in text.lower():
            for line in text.split("."):
                if phrase in line.lower() and len(line.strip()) > 30:
                    snippet = line.strip()[:350]
                    break
            if snippet:
                break
    if not snippet:
        for line in text.split("."):
            if "agreement" in line.lower() and "material" in line.lower():
                snippet = line.strip()[:300]
                break

    event = {
        "filer_ticker": filer_ticker,
        "filer_name": filer_name,
        "filing_date": filing_date,
        "accession_number": accession_number,
        "sec_url": sec_url,
        "counterparties": counterparties,
        "snippet": snippet or None,
        "relevance_type": relevance,
    }
    _save_json(cache_path, {"event": event, "cached_at": datetime.now().isoformat()})
    return event


def get_partnership_events(limit: int = 50, force_refresh: bool = False) -> List[dict]:
    """
    Return list of partnership events (8-K Item 1.01) for watched tickers.
    Each event: filer_ticker, filer_name, filing_date, accession_number, sec_url,
    counterparties (list of {name, is_interest}), snippet (optional).
    Uses file cache for events; when cache is empty returns [] (user should click Refresh).
    """
    events_path = CACHE_DIR / EVENTS_CACHE_FILE
    if not force_refresh and events_path.exists():
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = data.get("events") or []
            return events[:limit]
        except (json.JSONDecodeError, OSError):
            pass

    if force_refresh:
        return refresh_edgar_data(limit=limit)
    return []


def refresh_edgar_data(limit: int = 50) -> List[dict]:
    """
    Force refresh: fetch submissions for all watch tickers, process 8-Ks with Item 1.01,
    extract counterparties, cache results, and return events (newest first, up to limit).
    """
    _ensure_cache_dir()
    ticker_to_cik = get_ticker_to_cik()
    if not ticker_to_cik:
        return []

    all_events = []
    seen_accessions = set()

    for ticker in WATCH_TICKERS:
        t = str(ticker).upper()
        if t not in ticker_to_cik:
            print(f"[EDGAR] Skipping unknown ticker: {ticker}")
            continue
        cik, company_name = ticker_to_cik[t]
        submissions = _get_submissions_for_cik(cik, force_refresh=True)
        if not submissions:
            continue
        name = submissions.get("name") or company_name or t
        eight_ks = _parse_recent_8ks(submissions)
        for filing in eight_ks:
            acc = filing.get("accessionNumber") or ""
            if acc in seen_accessions:
                continue
            seen_accessions.add(acc)
            event = _process_8k(
                cik=cik,
                filer_ticker=t,
                filer_name=name,
                accession_number=acc,
                filing_date=filing.get("filingDate") or "",
                primary_doc_hint=filing.get("primaryDocument") or "",
            )
            if event:
                all_events.append(event)
            if len(all_events) >= limit * 2:
                break
        if len(all_events) >= limit * 2:
            break

    # Sort by filing date descending, then take limit
    all_events.sort(key=lambda e: (e.get("filing_date") or ""), reverse=True)
    events = all_events[:limit]
    _save_json(CACHE_DIR / EVENTS_CACHE_FILE, {"events": events, "updated": datetime.now().isoformat()})
    return events


def clear_edgar_cache():
    """Remove all cached EDGAR data (submissions, 8-K content, events)."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        print("[EDGAR] Cache cleared.")
