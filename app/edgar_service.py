"""
SEC EDGAR service for 8-K partnership tracking.

Watches configured public companies for Form 8-K Item 1.01 (Entry into a Material
Definitive Agreement), fetches filing content, and extracts counterparty names.
Highlights counterparties that match the configured interest list (e.g. private companies).
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

import partnerships_config as partnerships_cfg
from partnerships_config import WATCH_TICKERS

from partnership_signal import events_need_signal_refresh, resolve_counterparty_hits

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
EVENTS_CACHE_SCHEMA_VERSION = 4

# Max 8-K rows per watchlist ticker after merge (recent + optional bulk JSON), before global date sort.
_MAX_8K_ROWS_PER_TICKER = 48
_SEC_GET_MAX_RETRIES = 5
# SEC bulk submission files (CIK…-submissions-001.json) change rarely; cache longer than main submissions.
BULK_SUBMISSIONS_CACHE_HOURS = 24

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


def _retry_after_sleep_seconds(retry_after_header: Optional[str], attempt: int) -> float:
    """Parse Retry-After: delta-seconds, HTTP-date, or fallback exponential cap."""
    fallback = float(min(60, 2**attempt))
    ra = retry_after_header
    if not ra or not str(ra).strip():
        return fallback
    s = str(ra).strip()
    if s.isdigit():
        return float(min(120, int(s)))
    try:
        dt = parsedate_to_datetime(s)
        if dt is None:
            return fallback
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(1.0, min(120.0, delta))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return float(min(60, float(s)))
    except ValueError:
        return fallback


def _sec_get(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: float = 15,
    max_retries: int = _SEC_GET_MAX_RETRIES,
) -> Optional[requests.Response]:
    """
    GET with SEC rate limit spacing, 429 Retry-After, and backoff on 5xx / network errors.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        _throttle()
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait_s = _retry_after_sleep_seconds(r.headers.get("Retry-After"), attempt)
                print(f"[EDGAR] SEC 429, sleeping {wait_s:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
            if r.status_code in (500, 502, 503, 504):
                wait_s = min(30.0, 0.5 * (2**attempt))
                print(
                    f"[EDGAR] SEC {r.status_code} for {url[:72]}… "
                    f"sleep {wait_s:.1f}s ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            wait_s = min(20.0, 0.5 * (2**attempt))
            time.sleep(wait_s)
    if last_err:
        print(f"[EDGAR] SEC GET gave up after {max_retries} tries: {url[:96]} — {last_err}")
    return None


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
    """Write JSON atomically (temp + replace) to avoid torn files on crash."""
    _ensure_cache_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)
    os.replace(tmp, path)


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

    r = _sec_get(COMPANY_TICKERS_URL, _headers())
    if r is None:
        return {}
    try:
        data = r.json()
        _save_json(cache_path, data)
        return _build_ticker_map(data)
    except json.JSONDecodeError as e:
        print(f"[EDGAR] Failed to parse company tickers: {e}")
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
    r = _sec_get(url, _headers())
    if r is None:
        print(f"[EDGAR] Submissions failed for CIK {cik}")
        return None
    try:
        data = r.json()
        _save_json(cache_path, data)
        return data
    except json.JSONDecodeError as e:
        print(f"[EDGAR] Submissions JSON invalid for CIK {cik}: {e}")
        return None


def _parse_columnar_8ks(block: Optional[dict]) -> List[dict]:
    """
    Parse 8-K rows from a SEC columnar block: either filings.recent or a bulk CIK-submissions-NNN.json root.
    """
    if not block or not isinstance(block, dict):
        return []

    forms = block.get("form") or []
    accession_numbers = block.get("accessionNumber") or []
    filing_dates = block.get("filingDate") or []
    primary_docs = block.get("primaryDocument") or []

    n = len(forms)
    result: List[dict] = []
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


def _parse_recent_8ks(submissions: dict) -> List[dict]:
    """From company submissions JSON, 8-K rows in filings.recent only."""
    recent = (submissions.get("filings") or {}).get("recent")
    return _parse_columnar_8ks(recent if isinstance(recent, dict) else None)


def _get_submissions_bulk_file(filename: str) -> Optional[dict]:
    """Fetch CIK-submissions-NNN.json (older columnar history). Cached separately from main submissions."""
    safe_name = filename.replace("/", "_").replace("\\", "_")
    cache_path = CACHE_DIR / f"bulk_submissions_{safe_name}"
    cached = _cached_json(cache_path, BULK_SUBMISSIONS_CACHE_HOURS)
    if cached is not None:
        return cached

    url = f"https://data.sec.gov/submissions/{filename}"
    r = _sec_get(url, _headers())
    if r is None:
        print(f"[EDGAR] Bulk submissions failed: {filename}")
        return None
    try:
        data = r.json()
        _save_json(cache_path, data)
        return data
    except json.JSONDecodeError as e:
        print(f"[EDGAR] Bulk submissions JSON invalid {filename}: {e}")
        return None


def _8k_filings_for_ticker(submissions: dict, max_rows: int) -> List[dict]:
    """
    Merge 8-K rows from filings.recent and optionally SEC bulk submission JSON files
    (partnerships_config.EDGAR_EXTRA_SUBMISSION_JSON_FILES_PER_CIK), dedupe by accession,
    sort by filing date descending, return up to max_rows.
    """
    filings = submissions.get("filings") or {}
    recent_block = filings.get("recent")
    combined: List[dict] = []
    combined.extend(_parse_columnar_8ks(recent_block if isinstance(recent_block, dict) else None))

    extra_n = int(getattr(partnerships_cfg, "EDGAR_EXTRA_SUBMISSION_JSON_FILES_PER_CIK", 0) or 0)
    for fi in (filings.get("files") or [])[: max(0, extra_n)]:
        if not isinstance(fi, dict):
            continue
        name = fi.get("name")
        if not name or not isinstance(name, str):
            continue
        bulk = _get_submissions_bulk_file(name)
        if bulk:
            combined.extend(_parse_columnar_8ks(bulk))

    combined.sort(key=lambda r: (r.get("filingDate") or ""), reverse=True)
    seen: set[str] = set()
    out: List[dict] = []
    for r in combined:
        acc = (r.get("accessionNumber") or "").strip()
        if not acc or acc in seen:
            continue
        seen.add(acc)
        out.append(r)
    return out[:max_rows]


def _normalize_primary_doc_hint(hint: str) -> str:
    return (hint or "").strip().lower()


def _is_8k_disk_cache_valid(cached: dict, primary_doc_hint: str, filing_date: str) -> bool:
    """
    Invalidate per-accession JSON if SEC metadata we keyed on changed (e.g. primary document path).
    Legacy caches without source_* keys remain valid until TTL.
    """
    if "source_primary_document" not in cached:
        return True
    if _normalize_primary_doc_hint(cached.get("source_primary_document") or "") != _normalize_primary_doc_hint(
        primary_doc_hint
    ):
        return False
    sf = (cached.get("source_filing_date") or "").strip()
    cf = (filing_date or "").strip()
    if sf and cf and sf != cf:
        return False
    return True


def _is_8k_skip_cached(cached: dict, primary_doc_hint: str, filing_date: str) -> bool:
    """
    True if we already fetched this accession and dropped it (no Item 1.01 or financing filter).
    Same metadata invalidation as positive 8-K cache.
    """
    if not cached or not isinstance(cached, dict):
        return False
    if not _is_8k_disk_cache_valid(cached, primary_doc_hint, filing_date):
        return False
    reason = (cached.get("skip_reason") or "").strip()
    if not reason:
        return False
    ev = cached.get("event")
    if isinstance(ev, dict) and ev:
        return False
    return True


def _save_8k_skip_cache(
    cache_path: Path,
    primary_doc_hint: str,
    filing_date: str,
    skip_reason: str,
) -> None:
    _save_json(
        cache_path,
        {
            "event": None,
            "skip_reason": skip_reason,
            "cached_at": datetime.now().isoformat(),
            "source_primary_document": _normalize_primary_doc_hint(primary_doc_hint),
            "source_filing_date": (filing_date or "").strip(),
        },
    )


def _accession_to_path(accession_number: str) -> str:
    """Convert accession number to URL path part (strip hyphens)."""
    return (accession_number or "").replace("-", "")


def _fetch_8k_index(cik: str, accession_number: str) -> Optional[str]:
    """Fetch the index HTML for a filing; return raw HTML or None."""
    path_part = _accession_to_path(accession_number)
    # Index filename is typically {accession}-index.htm
    index_name = f"{accession_number}-index.htm"
    url = f"{ARCHIVES_BASE}/{cik}/{path_part}/{index_name}"
    r = _sec_get(url, _headers_html())
    if r is None:
        print(f"[EDGAR] Failed to fetch index {accession_number}")
        return None
    return r.text


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


def _index_htm_candidates(html: str) -> List[str]:
    """Ordered unique .htm/.html filenames from a filing index page (excludes *index*)."""
    seen: set[str] = set()
    out: List[str] = []

    def _add_raw_href(raw: str) -> None:
        fn = _filename_only(raw)
        if not fn or not fn.lower().endswith((".htm", ".html")):
            return
        if "index" in fn.lower():
            return
        key = fn.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(fn)

    for m in re.finditer(r'href="([^"]+\.(?:htm|html))(?:"|\?|$)', html, re.I):
        _add_raw_href(m.group(1))
    for m in re.finditer(r'href="(ix\?doc=[^"]+\.(?:htm|html))"', html, re.I):
        _add_raw_href(m.group(1))
    return out


def _is_exhibit_991_filename(filename: str) -> bool:
    """True if filename looks like Exhibit 99.1 style (press release / agreement body)."""
    b = _filename_only(filename).lower()
    if not b.endswith((".htm", ".html")):
        return False
    return bool(
        re.search(
            r"ex99[-_.]?1(?:[^0-9]|\.htm|\.html)|exhibit[-_.]?99[-_.]?1|ex[-_.]99[-_.]?1(?:\.htm|\.html|$)",
            b,
        )
    )


def _pick_exhibit_991_filename(candidates: List[str], primary_filename: str) -> Optional[str]:
    """First exhibit 99.1 candidate that is not the primary document."""
    primary_lower = _filename_only(primary_filename).lower()
    for fn in candidates:
        if _filename_only(fn).lower() == primary_lower:
            continue
        if _is_exhibit_991_filename(fn):
            return fn
    return None


def _find_primary_doc_from_index(html: str, accession_number: str, primary_doc_hint: str) -> Optional[str]:
    """
    Parse index HTML to find the primary 8-K document filename.
    Handles: href="file.htm", href="/Archives/edgar/.../file.htm", and ix?doc=/Archives/.../file.htm.
    Returns only the filename (e.g. file.htm) for building the fetch URL.
    """
    candidates = _index_htm_candidates(html)
    if not candidates:
        return None

    if primary_doc_hint and primary_doc_hint.strip():
        hint = primary_doc_hint.strip().lower()
        for fn in candidates:
            if hint in fn.lower():
                return fn

    for fn in candidates:
        lower = fn.lower()
        if "ex" not in lower or "8k" in lower or "8-k" in lower:
            return fn
    return candidates[0]


def _fetch_8k_document(cik: str, accession_number: str, doc_filename: str) -> Optional[str]:
    """Fetch 8-K document; doc_filename must be filename only (e.g. file.htm)."""
    filename_only = _filename_only(doc_filename)
    path_part = _accession_to_path(accession_number)

    # If we got an absolute path (starts with /), use sec.gov + path
    if doc_filename.strip().startswith("/"):
        url = f"https://www.sec.gov{doc_filename.split('?')[0].strip()}"
    else:
        url = f"{ARCHIVES_BASE}/{cik}/{path_part}/{filename_only}"

    r = _sec_get(url, _headers_html())
    if r is None:
        print(f"[EDGAR] Failed to fetch document {filename_only}")
        return None
    return r.text


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


def _counterparty_interest_hit(name: str) -> bool:
    """Match interest list + aliases using the same rules as partnership_signal."""
    hit, _ = resolve_counterparty_hits(
        [{"name": name}],
        partnerships_cfg.COUNTERPARTY_INTEREST_NAMES,
        partnerships_cfg.COUNTERPARTY_ALIASES,
    )
    return hit


def _process_8k(cik: str, filer_ticker: str, filer_name: str, accession_number: str,
                filing_date: str, primary_doc_hint: str) -> Optional[dict]:
    """
    Fetch 8-K content, detect Item 1.01, extract counterparties. Return event dict or None.
    """
    cache_path = CACHE_DIR / f"8k_{_accession_to_path(accession_number)}.json"
    cached = _cached_json(cache_path, SUBMISSIONS_CACHE_HOURS * 24)
    if cached is not None:
        if _is_8k_skip_cached(cached, primary_doc_hint, filing_date):
            return None
        if _is_8k_disk_cache_valid(cached, primary_doc_hint, filing_date):
            ev = cached.get("event")
            if isinstance(ev, dict) and ev:
                return ev

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
    index_candidates = _index_htm_candidates(index_html)
    exhibit_fn = _pick_exhibit_991_filename(index_candidates, doc_filename)
    if exhibit_fn:
        ex_html = _fetch_8k_document(cik, accession_number, exhibit_fn)
        if ex_html:
            ex_text = _strip_html(ex_html)
            if len(ex_text) > 80:
                text = f"{text}\n\n--- Exhibit 99.1 ---\n\n{ex_text}"

    if not _is_item_101_filing(text):
        _save_8k_skip_cache(cache_path, primary_doc_hint, filing_date, "no_item_101")
        return None

    relevance = _relevance_type(text)
    # Filter out routine financing (credit/indenture/notes) so only partnership/strategic events surface
    if relevance == "financing":
        _save_8k_skip_cache(cache_path, primary_doc_hint, filing_date, "financing")
        return None

    counterparty_raw = _extract_counterparties(text, relevance)
    counterparties = []
    for name in counterparty_raw:
        counterparties.append({"name": name, "is_interest": _counterparty_interest_hit(name)})

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
    _save_json(
        cache_path,
        {
            "event": event,
            "cached_at": datetime.now().isoformat(),
            "source_primary_document": _normalize_primary_doc_hint(primary_doc_hint),
            "source_filing_date": (filing_date or "").strip(),
        },
    )
    return event


def partnership_events_caps_deferred() -> bool:
    """True if partnership_events.json exists and was saved with caps_enriched=false (Yahoo not run yet)."""
    events_path = CACHE_DIR / EVENTS_CACHE_FILE
    if not events_path.exists():
        return False
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("caps_enriched") is False
    except (json.JSONDecodeError, OSError):
        return False


def hydrate_partnership_market_caps(events: List[dict]) -> List[dict]:
    """
    If the on-disk events file is still cap-deferred, run Yahoo cap enrichment and persist caps_enriched=true.
    No-op when caps are already enriched or the file is missing.
    """
    events_path = CACHE_DIR / EVENTS_CACHE_FILE
    if not events_path.exists():
        return events
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return events
    if file_data.get("caps_enriched") is not False:
        return events
    from partnership_enrichment import enrich_partnership_with_caps

    try:
        out = enrich_partnership_with_caps(events)
        _save_json(
            events_path,
            {
                "events": out,
                "updated": datetime.now().isoformat(),
                "cache_schema_version": EVENTS_CACHE_SCHEMA_VERSION,
                "caps_enriched": True,
            },
        )
        return out
    except Exception as ex:
        print(f"[EDGAR] partnership cap hydrate failed: {ex}")
        return events


def get_partnership_events(
    limit: int = 50,
    force_refresh: bool = False,
    defer_yfinance: bool = False,
) -> List[dict]:
    """
    Return list of partnership events (8-K Item 1.01) for watched tickers.
    Each event: filer_ticker, filer_name, filing_date, accession_number, sec_url,
    counterparties (list of {name, is_interest}), snippet (optional).
    Uses file cache for events; when cache is empty returns [] (user should click Refresh).

    When defer_yfinance is True (Streamlit default via cache wrapper), stale signal rows are refreshed
    and saved with caps_enriched=false first; Yahoo caps are skipped until hydrate_partnership_market_caps
    or a non-deferred read. Legacy cache files without caps_enriched are treated as fully enriched.
    """
    from partnership_enrichment import enrich_partnership_signals_only, enrich_partnership_with_caps

    events_path = CACHE_DIR / EVENTS_CACHE_FILE
    if force_refresh:
        events, _warnings = refresh_edgar_data(limit=limit, force_submissions_refresh=True)
        return events

    if not events_path.exists():
        return []

    try:
        with open(events_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    events = list(file_data.get("events") or [])
    if not events:
        return []

    if events_need_signal_refresh(events):
        try:
            events = enrich_partnership_signals_only(events)
            _save_json(
                events_path,
                {
                    "events": events,
                    "updated": datetime.now().isoformat(),
                    "cache_schema_version": EVENTS_CACHE_SCHEMA_VERSION,
                    "caps_enriched": False,
                },
            )
            if defer_yfinance:
                return events[:limit]
            events = enrich_partnership_with_caps(events)
            _save_json(
                events_path,
                {
                    "events": events,
                    "updated": datetime.now().isoformat(),
                    "cache_schema_version": EVENTS_CACHE_SCHEMA_VERSION,
                    "caps_enriched": True,
                },
            )
            return events[:limit]
        except Exception as ex:
            print(f"[EDGAR] partnership enrich on read failed: {ex}")
            return events[:limit]

    if file_data.get("caps_enriched") is False:
        if defer_yfinance:
            return events[:limit]
        try:
            events = enrich_partnership_with_caps(events)
            _save_json(
                events_path,
                {
                    "events": events,
                    "updated": datetime.now().isoformat(),
                    "cache_schema_version": EVENTS_CACHE_SCHEMA_VERSION,
                    "caps_enriched": True,
                },
            )
        except Exception as ex:
            print(f"[EDGAR] partnership cap enrich on read failed: {ex}")
        return events[:limit]

    return events[:limit]


def refresh_edgar_data(
    limit: int = 50,
    force_submissions_refresh: bool = False,
) -> Tuple[List[dict], List[str]]:
    """
    Force refresh: fetch submissions for all watch tickers, process 8-Ks with Item 1.01,
    extract counterparties, cache results, and return events (newest first, up to limit).

    When force_submissions_refresh is False, uses cached submissions JSON per CIK when still
    within SUBMISSIONS_CACHE_HOURS (faster Refresh). Set True to re-download every CIK feed.

    Returns (events, warnings). Warnings are safe to show in the UI (skipped tickers, etc.).
    """
    from partnership_enrichment import enrich_partnership_events

    _ensure_cache_dir()
    ticker_to_cik = get_ticker_to_cik()
    warnings: List[str] = []
    if not ticker_to_cik:
        warnings.append("SEC company ticker list unavailable; cannot map symbols to CIK.")
        return [], warnings

    all_events: List[dict] = []
    seen_accessions: set[str] = set()
    # Merge 8-K rows across all watch tickers by filing date so early symbols do not starve the rest.
    candidates: List[dict] = []

    for ticker in WATCH_TICKERS:
        t = str(ticker).upper()
        if t not in ticker_to_cik:
            warnings.append(f"No CIK mapping for watchlist ticker {t} (skipped).")
            print(f"[EDGAR] Skipping unknown ticker: {ticker}")
            continue
        cik, company_name = ticker_to_cik[t]
        submissions = _get_submissions_for_cik(cik, force_refresh=force_submissions_refresh)
        if not submissions:
            continue
        name = submissions.get("name") or company_name or t
        eight_ks = _8k_filings_for_ticker(submissions, _MAX_8K_ROWS_PER_TICKER)
        for filing in eight_ks:
            candidates.append(
                {
                    "cik": cik,
                    "filer_ticker": t,
                    "filer_name": name,
                    "accessionNumber": filing.get("accessionNumber") or "",
                    "filingDate": filing.get("filingDate") or "",
                    "primaryDocument": filing.get("primaryDocument") or "",
                }
            )

    candidates.sort(key=lambda row: (row.get("filingDate") or ""), reverse=True)

    max_raw = max(limit * 2, 1)
    for row in candidates:
        acc = (row.get("accessionNumber") or "").strip()
        if not acc or acc in seen_accessions:
            continue
        seen_accessions.add(acc)
        event = _process_8k(
            cik=row["cik"],
            filer_ticker=row["filer_ticker"],
            filer_name=row["filer_name"],
            accession_number=acc,
            filing_date=row.get("filingDate") or "",
            primary_doc_hint=row.get("primaryDocument") or "",
        )
        if event:
            all_events.append(event)
        if len(all_events) >= max_raw:
            break

    # Sort by filing date descending, then take limit
    all_events.sort(key=lambda e: (e.get("filing_date") or ""), reverse=True)
    events = all_events[:limit]
    try:
        events = enrich_partnership_events(events)
    except Exception as ex:
        print(f"[EDGAR] partnership enrich after refresh failed: {ex}")
    _save_json(
        CACHE_DIR / EVENTS_CACHE_FILE,
        {
            "events": events,
            "updated": datetime.now().isoformat(),
            "cache_schema_version": EVENTS_CACHE_SCHEMA_VERSION,
            "caps_enriched": True,
        },
    )
    return events, warnings


def clear_edgar_cache():
    """Remove all cached EDGAR data (submissions, 8-K content, events)."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        print("[EDGAR] Cache cleared.")
