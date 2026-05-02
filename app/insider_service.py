"""
SEC Form 4 insider transaction service.

Uses SEC EDGAR as the canonical free source for insider activity and returns
the same normalized shape consumed by Market Analysis.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import os
import time
import xml.etree.ElementTree as ET

from edgar_service import (
    ARCHIVES_BASE,
    CACHE_DIR,
    _accession_to_path,
    _get_submissions_for_cik,
    _headers_html,
    _sec_get,
    get_ticker_to_cik,
)


INSIDER_CACHE_HOURS = 6
MAX_FORM4_FILINGS = 30
MAX_TRANSACTIONS = 50
FORM4_TYPES = frozenset({"4", "4/A"})
OPEN_MARKET_CODES = frozenset({"P", "S"})


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    for child in list(element):
        if _local_name(child.tag) == name:
            yield child


def _first_child(element: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if element is None:
        return None
    return next(_children(element, name), None)


def _first_descendant(element: ET.Element, name: str) -> Optional[ET.Element]:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _text_at(element: Optional[ET.Element], *path: str) -> str:
    cur = element
    for part in path:
        cur = _first_child(cur, part)
        if cur is None:
            return ""
    return (cur.text or "").strip()


def _desc_text(element: ET.Element, name: str) -> str:
    found = _first_descendant(element, name)
    return (found.text or "").strip() if found is not None else ""


def _parse_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value or "").strip()
    if not s:
        return None
    if "T" in s:
        s = s.split("T", 1)[0]
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_float(value: object) -> Optional[float]:
    s = str(value or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "y", "yes"}


def _relationship_label(owner: Optional[ET.Element]) -> tuple[str, bool, bool]:
    relationship = _first_child(owner, "reportingOwnerRelationship") if owner is not None else None
    is_director = _parse_bool(_text_at(relationship, "isDirector"))
    is_officer = _parse_bool(_text_at(relationship, "isOfficer"))
    is_ten_percent = _parse_bool(_text_at(relationship, "isTenPercentOwner"))
    is_other = _parse_bool(_text_at(relationship, "isOther"))
    officer_title = _text_at(relationship, "officerTitle")
    other_text = _text_at(relationship, "otherText")

    if officer_title:
        label = officer_title
    elif is_director:
        label = "Director"
    elif is_officer:
        label = "Officer"
    elif is_ten_percent:
        label = "10% Owner"
    elif is_other and other_text:
        label = other_text
    else:
        label = ""
    return label, is_officer, is_director


def _normalize_transaction(code: str, acquired_disposed: str) -> str:
    code_u = (code or "").strip().upper()
    ad_u = (acquired_disposed or "").strip().upper()
    if code_u == "P":
        return "Buy"
    if code_u == "S":
        return "Sale"
    if ad_u == "A":
        return "Buy"
    if ad_u == "D":
        return "Sale"
    return "Other"


def parse_form4_xml(xml_text: str, *, sec_link: str, filing_date: object = None) -> List[Dict]:
    """
    Parse SEC Form 4 ownership XML into normalized insider transaction rows.

    Only non-derivative transactions are included. Derivative rows are excluded
    because the app's table/chart expect direct share transactions.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    owner = _first_descendant(root, "reportingOwner")
    owner_name = _text_at(_first_child(owner, "reportingOwnerId"), "rptOwnerName")
    relationship, is_officer, is_director = _relationship_label(owner)
    filing_dt = _parse_date(filing_date)

    non_derivative_table = _first_descendant(root, "nonDerivativeTable")
    if non_derivative_table is None:
        return []

    rows: List[Dict] = []
    for transaction in _children(non_derivative_table, "nonDerivativeTransaction"):
        transaction_dt = _parse_date(_text_at(transaction, "transactionDate", "value"))
        shares = _parse_float(_text_at(transaction, "transactionAmounts", "transactionShares", "value"))
        if transaction_dt is None or shares is None:
            continue

        price = _parse_float(
            _text_at(transaction, "transactionAmounts", "transactionPricePerShare", "value")
        )
        code = _text_at(transaction, "transactionCoding", "transactionCode").upper()
        acquired_disposed = _text_at(
            transaction,
            "transactionAmounts",
            "transactionAcquiredDisposedCode",
            "value",
        ).upper()
        value = int(round(abs(shares * price))) if price is not None else 0

        rows.append(
            {
                "date": transaction_dt,
                "filing_date": filing_dt,
                "transaction": _normalize_transaction(code, acquired_disposed),
                "transaction_raw": code,
                "shares": int(shares) if float(shares).is_integer() else shares,
                "price": price,
                "value": value,
                "name": owner_name,
                "relationship": relationship,
                "sec_link": sec_link,
                "source": "sec",
                "open_market": code in OPEN_MARKET_CODES,
                "is_officer": is_officer,
                "is_director": is_director,
            }
        )
    return rows


def parse_recent_form4_filings(submissions: Optional[dict], limit: int = MAX_FORM4_FILINGS) -> List[Dict]:
    """Extract recent Form 4/4-A filing rows from SEC submissions JSON."""
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    primary_documents = recent.get("primaryDocument") or []

    rows: List[Dict] = []
    for idx, form in enumerate(forms):
        form_s = str(form or "").strip().upper()
        if form_s not in FORM4_TYPES:
            continue
        accession = _safe_list_get(accessions, idx)
        primary_document = _safe_list_get(primary_documents, idx)
        if not accession or not primary_document:
            continue
        rows.append(
            {
                "form": form_s,
                "accession_number": accession,
                "filing_date": _safe_list_get(filing_dates, idx),
                "report_date": _safe_list_get(report_dates, idx),
                "primary_document": primary_document,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _safe_list_get(values: list, idx: int) -> str:
    if idx >= len(values):
        return ""
    return str(values[idx] or "").strip()


def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"insider_{ticker.upper()}.json"


def _load_cached_rows(ticker: str) -> Optional[List[Dict]]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        if (time.time() - path.stat().st_mtime) / 3600 > INSIDER_CACHE_HOURS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    raw_rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list):
        return None
    return [_deserialize_row(row) for row in raw_rows if isinstance(row, dict)]


def _save_cached_rows(ticker: str, rows: List[Dict]) -> None:
    try:
        payload = {
            "cached_at": datetime.now().isoformat(),
            "source": "sec",
            "rows": [_serialize_row(row) for row in rows],
        }
        path = _cache_path(ticker)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=0)
        os.replace(tmp, path)
    except OSError:
        pass


def _serialize_row(row: Dict) -> Dict:
    out = dict(row)
    for key in ("date", "filing_date"):
        value = out.get(key)
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
    return out


def _deserialize_row(row: Dict) -> Dict:
    out = dict(row)
    out["date"] = _parse_date(out.get("date"))
    out["filing_date"] = _parse_date(out.get("filing_date"))
    return out


def _filter_by_date(rows: List[Dict], from_date: Optional[date], to_date: Optional[date]) -> List[Dict]:
    filtered = []
    for row in rows:
        row_date = _parse_date(row.get("date"))
        if row_date is None:
            continue
        if from_date and row_date < from_date:
            continue
        if to_date and row_date > to_date:
            continue
        filtered.append({**row, "date": row_date})
    return filtered


def _filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_path = cik.lstrip("0") or cik
    accession_path = _accession_to_path(accession_number)
    doc_path = (primary_document or "").strip().lstrip("/")
    if doc_path.startswith("http://") or doc_path.startswith("https://"):
        return doc_path
    if doc_path.startswith("Archives/"):
        return f"https://www.sec.gov/{doc_path}"
    parts = doc_path.split("/")
    if len(parts) > 1 and parts[0].lower().startswith("xslf345"):
        doc_path = parts[-1]
    return f"{ARCHIVES_BASE}/{cik_path}/{accession_path}/{doc_path}"


def fetch_insider_transactions_sec(
    ticker: str,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    *,
    max_filings: int = MAX_FORM4_FILINGS,
) -> List[Dict]:
    """Fetch normalized insider transactions from recent SEC Form 4 filings."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []

    cached = _load_cached_rows(ticker)
    if cached is not None:
        return _filter_by_date(cached, from_date, to_date)

    ticker_map = get_ticker_to_cik()
    mapping = ticker_map.get(ticker)
    if not mapping:
        return []
    cik, _company_title = mapping

    submissions = _get_submissions_for_cik(cik)
    filings = parse_recent_form4_filings(submissions, limit=max_filings)
    if not filings:
        return []

    rows: List[Dict] = []
    for filing in filings:
        url = _filing_document_url(cik, filing["accession_number"], filing["primary_document"])
        response = _sec_get(url, _headers_html())
        if response is None:
            continue
        rows.extend(
            parse_form4_xml(
                response.text,
                sec_link=url,
                filing_date=filing.get("filing_date"),
            )
        )

    rows.sort(
        key=lambda row: (
            row.get("date") or date.min,
            row.get("filing_date") or date.min,
        ),
        reverse=True,
    )
    rows = rows[:MAX_TRANSACTIONS]
    if rows:
        _save_cached_rows(ticker, rows)
    return _filter_by_date(rows, from_date, to_date)
