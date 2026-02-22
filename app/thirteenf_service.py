"""
13F Institutional Holdings service.

Fetches SEC 13F-HR filings for configured institutions, parses holdings,
and provides single-filing view, QoQ compare, by-CUSIP holders, and overlap.
"""

import json
import re
import time
import calendar
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

import requests

from edgar_service import (
    USER_AGENT,
    SEC_RATE_LIMIT_DELAY,
    SUBMISSIONS_URL_TEMPLATE,
    ARCHIVES_BASE,
)

# Cache under edgar_cache/13f
# Project root (caches live at Invest/ root)
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".edgar_cache" / "13f"
SUBMISSIONS_CACHE_HOURS = 1
HOLDINGS_CACHE_HOURS = 24

# 13F-HR form types we accept
FORM_13F_HR = "13F-HR"
FORM_13F_HR_AMENDMENT = "13F-HR/A"


def _headers_json() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _headers_xml() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/xml,text/xml,*/*"}


def _headers_html() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}


def _throttle():
    time.sleep(SEC_RATE_LIMIT_DELAY)


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def _accession_to_path(accession_number: str) -> str:
    return (accession_number or "").replace("-", "")


def _cik_pad(cik: str) -> str:
    return str(cik).strip().zfill(10)


def _get_submissions_for_cik(cik: str, force_refresh: bool = False) -> Optional[dict]:
    cik = _cik_pad(cik)
    cache_path = CACHE_DIR / f"submissions_{cik}.json"
    if not force_refresh:
        cached = _cached_json(cache_path, SUBMISSIONS_CACHE_HOURS)
        if cached is not None:
            return cached
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    _throttle()
    try:
        r = requests.get(url, headers=_headers_json(), timeout=15)
        r.raise_for_status()
        data = r.json()
        _save_json(cache_path, data)
        return data
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[13F] Submissions failed for CIK {cik}: {e}")
        return None


def _parse_recent_13f_hr(submissions: dict) -> List[dict]:
    """From submissions JSON, return list of 13F-HR filings (columnar recent array)."""
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
        form = (forms[i] or "").strip().upper() if i < len(forms) else ""
        if form not in (FORM_13F_HR, FORM_13F_HR_AMENDMENT):
            continue
        if i >= len(accession_numbers) or i >= len(filing_dates):
            continue
        acc = (accession_numbers[i] or "").strip()
        if not acc:
            continue
        result.append({
            "accessionNumber": acc,
            "filingDate": filing_dates[i] if i < len(filing_dates) else "",
            "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            "form_type": form,
        })
    return result


def _filing_date_to_quarter(filing_date: str) -> Optional[tuple]:
    """Map filing date (YYYY-MM-DD) to (year, quarter). 13F is filed ~45 days after quarter end."""
    if not filing_date or len(filing_date) < 7:
        return None
    try:
        parts = filing_date.split("-")
        y, m = int(parts[0]), int(parts[1])
        # Filing in Feb -> Q4 prior year; May -> Q1; Aug -> Q2; Nov -> Q3
        if m in (1, 2):
            return (y - 1, 4)
        if m <= 5:
            return (y, 1)
        if m <= 8:
            return (y, 2)
        if m <= 11:
            return (y, 3)
        return (y, 4)
    except (ValueError, IndexError):
        return None


# SEC index pages may use full URLs or paths; we normalize to relative (filename or subpath only)
_SEC_ARCHIVES_PREFIX = "https://www.sec.gov/Archives/edgar/data/"
_SEC_ARCHIVES_PREFIX_ALT = "/Archives/edgar/data/"


def _find_13f_xml_paths_from_index(html: str) -> List[str]:
    """
    Find all 13F XML document paths from index page.
    Index hrefs may be full URLs (https://www.sec.gov/...) or absolute paths (/Archives/...).
    We return either the full URL (for _fetch_13f_xml to use as-is) or a relative path.
    Holdings data is usually in a separate INFORMATION TABLE file (e.g. 50240.xml), not primary_doc.
    We try non-primary .xml files first, then primary_doc.xml.
    """
    pattern = re.compile(r'href="([^"]+\.xml)"', re.I)
    seen = set()
    primary_paths = []
    other_paths = []
    for href in pattern.findall(html):
        path = href.split("?")[0].replace("\\", "/").strip()
        if not path or path in seen:
            continue
        # Normalize to relative path so we don't double-append: keep only the part after .../data/CIK/ACC/
        if path.startswith("http") or path.startswith("/"):
            # Extract relative part: .../1067983/000095012325005701/primary_doc.xml -> primary_doc.xml or xslForm13F_X02/primary_doc.xml
            for prefix in (_SEC_ARCHIVES_PREFIX, _SEC_ARCHIVES_PREFIX_ALT, "https://www.sec.gov"):
                if path.startswith(prefix):
                    rest = path[len(prefix):].lstrip("/")
                    # rest is like "1067983/000095012325005701/primary_doc.xml" - drop first two segments (cik, accession)
                    parts = rest.split("/")
                    if len(parts) >= 3:
                        path = "/".join(parts[2:])  # primary_doc.xml or xslForm13F_X02/primary_doc.xml
                    elif len(parts) == 1:
                        path = parts[0]
                    else:
                        path = rest
                    break
        seen.add(path)
        fn = path.rstrip("/").split("/")[-1].lower()
        if "primary" in fn and "doc" in fn:
            primary_paths.append(path)
        else:
            other_paths.append(path)
    return other_paths + primary_paths


def _cik_archives_path(cik: str) -> str:
    """SEC Archives URLs use numeric CIK (no leading zeros), not 10-digit padded."""
    return str(int(_cik_pad(cik)))


def _fetch_13f_index(cik: str, accession_number: str) -> Optional[str]:
    path_part = _accession_to_path(accession_number)
    cik_num = _cik_archives_path(cik)
    for suffix in ["-index.htm", "-index.html"]:
        url = f"{ARCHIVES_BASE}/{cik_num}/{path_part}/{accession_number}{suffix}"
        _throttle()
        try:
            r = requests.get(url, headers=_headers_html(), timeout=15)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            continue
    return None


def _fetch_13f_xml(cik: str, accession_number: str, xml_filename: str) -> Optional[str]:
    path_part = _accession_to_path(accession_number)
    cik_num = _cik_archives_path(cik)
    if xml_filename.startswith("http"):
        url = xml_filename.split("?")[0]
    else:
        url = f"{ARCHIVES_BASE}/{cik_num}/{path_part}/{xml_filename}"
    _throttle()
    try:
        r = requests.get(url, headers=_headers_xml(), timeout=20)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"[13F] Failed to fetch XML {xml_filename}: {e}")
        return None


def _strip_ns(tag: str) -> str:
    if tag and "}" in tag:
        return tag.split("}", 1)[1]
    return tag or ""


def _parse_13f_xml(xml_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse 13F primary doc XML. Extract cover info and informationTable rows.
    Returns {filer_name, period_end, value_thousands, holdings: [{issuer_name, cusip, value_thousands, shares, principal_type, option_type}]}.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"[13F] XML parse error: {e}")
        return None

    def find_text(parent: ET.Element, local_name: str, default: str = "") -> str:
        for c in parent.iter():
            if _strip_ns(c.tag) == local_name:
                return (c.text or "").strip()
        return default

    def find_child(parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        for c in parent:
            if _strip_ns(c.tag) == local_name:
                return c
        return None

    # Cover / header: reporting period, filer name (often in header or first section)
    filer_name = ""
    period_end = ""
    total_value = 0

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "nameOfIssuer" and not filer_name:
            # Sometimes filer is in a different structure; we'll use first name only if no cover
            pass
        if tag == "periodOfReport" or tag == "reportCalendarOrQuarter":
            period_end = (elem.text or "").strip()
        if tag == "reportType" and (elem.text or "").strip() == "13F-HR":
            # Parent might have company name
            p = elem
            for _ in range(5):
                if p is None:
                    break
                p = list(p)
                for c in p:
                    if _strip_ns(c.tag) == "name":
                        filer_name = (c.text or "").strip()
                        break
                p = elem if p == list(elem) else None
                break

    # Try common cover locations
    if not period_end:
        for e in root.iter():
            if _strip_ns(e.tag) == "periodOfReport":
                period_end = (e.text or "").strip()
                break
    if not filer_name:
        for e in root.iter():
            if _strip_ns(e.tag) == "name" and "company" in (e.get("context") or "").lower():
                filer_name = (e.text or "").strip()
                break
        if not filer_name and hasattr(root, "findall"):
            # Fallback: first name element at root level (often filer)
            for e in root.findall(".//*"):
                if _strip_ns(e.tag) == "name":
                    t = (e.text or "").strip()
                    if len(t) > 2 and len(t) < 200:
                        filer_name = t
                        break

    holdings = []
    # informationTable / infoTable
    for info_table in root.iter():
        if _strip_ns(info_table.tag) != "infoTable":
            continue
        issuer_name = ""
        cusip = ""
        value_thousands = 0
        shares = 0
        principal_type = "SH"
        option_type = ""

        for child in info_table:
            tag = _strip_ns(child.tag)
            if tag == "nameOfIssuer":
                issuer_name = (child.text or "").strip()
            elif tag == "cusip":
                cusip = (child.text or "").strip()
            elif tag == "value":
                try:
                    value_thousands = int(float((child.text or "0").replace(",", "")))
                except ValueError:
                    value_thousands = 0
            elif tag == "shrsOrPrnAmt":
                for sub in child:
                    if _strip_ns(sub.tag) == "sshPrnamt":
                        try:
                            shares = int(float((sub.text or "0").replace(",", "")))
                        except ValueError:
                            shares = 0
                    elif _strip_ns(sub.tag) == "sshPrnamtType":
                        principal_type = (sub.text or "SH").strip().upper() or "SH"
                if shares == 0 and child.text:
                    try:
                        shares = int(float((child.text or "0").replace(",", "")))
                    except ValueError:
                        pass
            elif tag == "putCall":
                option_type = (child.text or "").strip()

        if issuer_name or cusip:
            holdings.append({
                "issuer_name": issuer_name,
                "cusip": cusip,
                "value_thousands": value_thousands,
                "shares": shares,
                "principal_type": principal_type,
                "option_type": option_type or "",
            })

    if not holdings and not filer_name:
        # Maybe different XML structure: try ns
        ns = {"edgar": "http://www.sec.gov/edgar/document/thirteenf"}
        for it in root.findall(".//edgar:infoTable", ns):
            name_el = it.find("edgar:nameOfIssuer", ns)
            cusip_el = it.find("edgar:cusip", ns)
            val_el = it.find("edgar:value", ns)
            sh_el = it.find("edgar:shrsOrPrnAmt", ns)
            if name_el is not None or cusip_el is not None:
                issuer_name = (name_el.text or "").strip() if name_el is not None else ""
                cusip = (cusip_el.text or "").strip() if cusip_el is not None else ""
                try:
                    value_thousands = int(float((val_el.text or "0").replace(",", ""))) if val_el is not None else 0
                except ValueError:
                    value_thousands = 0
                shares = 0
                principal_type = "SH"
                if sh_el is not None:
                    for sub in sh_el:
                        if _strip_ns(sub.tag) == "sshPrnamt":
                            try:
                                shares = int(float((sub.text or "0").replace(",", "")))
                            except ValueError:
                                pass
                        elif _strip_ns(sub.tag) == "sshPrnamtType":
                            principal_type = (sub.text or "SH").strip().upper() or "SH"
                holdings.append({
                    "issuer_name": issuer_name,
                    "cusip": cusip,
                    "value_thousands": value_thousands,
                    "shares": shares,
                    "principal_type": principal_type,
                    "option_type": "",
                })

    total_value = sum(h["value_thousands"] for h in holdings)
    for h in holdings:
        h["pct"] = round((h["value_thousands"] / total_value * 100), 2) if total_value else 0

    if not filer_name:
        filer_name = "Unknown"

    return {
        "filer_name": filer_name,
        "period_end": period_end,
        "value_thousands": total_value,
        "num_holdings": len(holdings),
        "holdings": holdings,
    }


def _get_holdings_cached(cik: str, accession_number: str) -> Optional[Dict]:
    cache_path = CACHE_DIR / f"holdings_{_cik_pad(cik)}_{accession_number.replace('-', '_')}.json"
    cached = _cached_json(cache_path, HOLDINGS_CACHE_HOURS)
    if cached is not None and (cached.get("num_holdings") or 0) > 0:
        return cached

    submissions = _get_submissions_for_cik(cik)
    if not submissions:
        return None
    name = (submissions.get("name") or "").strip()
    filings_list = _parse_recent_13f_hr(submissions)
    filing = next((f for f in filings_list if f["accessionNumber"] == accession_number), None)
    if not filing:
        return None

    html = _fetch_13f_index(cik, accession_number)
    if not html:
        return None
    xml_paths = _find_13f_xml_paths_from_index(html)
    if not xml_paths:
        # Fallback: try info table file first (common SEC naming), then primary
        xml_paths = ["50240.xml", "primary_doc.xml", "infotable.xml"]
    parsed = None
    for xml_path in xml_paths:
        xml_str = _fetch_13f_xml(cik, accession_number, xml_path)
        if not xml_str:
            continue
        p = _parse_13f_xml(xml_str)
        if p and len(p.get("holdings") or []) > 0:
            parsed = p
            break
    if not parsed or not parsed.get("holdings"):
        return None
    # Fill metadata from submissions if info-table-only file had no cover
    if not parsed.get("filer_name"):
        parsed["filer_name"] = name
    if not parsed.get("period_end") and filing.get("filingDate"):
        q = _filing_date_to_quarter(filing["filingDate"])
        if q:
            y, qtr = q
            end_month = (3, 6, 9, 12)[qtr - 1]
            last_day = calendar.monthrange(y, end_month)[1]
            parsed["period_end"] = date_type(y, end_month, last_day).strftime("%Y-%m-%d")

    cik_padded = _cik_pad(cik)
    path_part = _accession_to_path(accession_number)
    sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{path_part}/{accession_number}-index.htm"

    out = {
        "filer_name": parsed["filer_name"] or name,
        "period_end": parsed.get("period_end") or "",
        "value_thousands": parsed["value_thousands"],
        "num_holdings": parsed["num_holdings"],
        "filing_date": filing["filingDate"],
        "form_type": filing["form_type"],
        "sec_url": sec_url,
        "holdings": parsed["holdings"],
        "accession_number": accession_number,
    }
    # Only cache when we got real holdings so a bad parse doesn't stick
    if out["num_holdings"] > 0:
        _save_json(cache_path, out)
    return out


# ---- Public API ----


def get_13f_filings_for_institution(cik: str) -> List[Dict]:
    """Return list of 13F-HR filings for the institution: accession_number, filing_date, year, quarter, form_type, sec_url."""
    submissions = _get_submissions_for_cik(cik)
    if not submissions:
        return []
    name = (submissions.get("name") or "").strip()
    filings_list = _parse_recent_13f_hr(submissions)
    cik_padded = _cik_pad(cik)
    result = []
    for f in filings_list:
        q = _filing_date_to_quarter(f["filingDate"])
        if q:
            y, qtr = q
            path_part = _accession_to_path(f["accessionNumber"])
            sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{path_part}/{f['accessionNumber']}-index.htm"
            result.append({
                "accession_number": f["accessionNumber"],
                "filing_date": f["filingDate"],
                "year": y,
                "quarter": qtr,
                "form_type": f["form_type"],
                "sec_url": sec_url,
            })
    return result


def get_13f_holdings(cik: str, accession_number: str) -> Optional[Dict]:
    """Return full holdings for one filing. Cache by (cik, accession)."""
    data = _get_holdings_cached(cik, accession_number)
    if not data:
        return None
    # Add sym placeholder (no CUSIP mapping in MVP)
    for h in data["holdings"]:
        h.setdefault("sym", "—")
    return data


def get_13f_holdings_by_quarter(cik: str, year: int, quarter: int) -> Optional[Dict]:
    """Resolve filing for (year, quarter) and return same structure as get_13f_holdings."""
    filings = get_13f_filings_for_institution(cik)
    match = next((f for f in filings if f["year"] == year and f["quarter"] == quarter), None)
    if not match:
        return None
    return get_13f_holdings(cik, match["accession_number"])


def get_13f_compare(cik: str, accession_a: str, accession_b: str) -> Optional[Dict]:
    """Compare two filings; return rows with shares/value for each period and diff/chg%."""
    data_a = _get_holdings_cached(cik, accession_a)
    data_b = _get_holdings_cached(cik, accession_b)
    if not data_a or not data_b:
        return None
    by_cusip_a = {h["cusip"]: h for h in data_a["holdings"]}
    by_cusip_b = {h["cusip"]: h for h in data_b["holdings"]}
    all_cusips = set(by_cusip_a) | set(by_cusip_b)
    rows = []
    for cusip in sorted(all_cusips):
        ha = by_cusip_a.get(cusip)
        hb = by_cusip_b.get(cusip)
        name_a = (ha or {}).get("issuer_name", "")
        name_b = (hb or {}).get("issuer_name", "")
        issuer_name = name_b or name_a
        sh_a = (ha or {}).get("shares", 0) or 0
        sh_b = (hb or {}).get("shares", 0) or 0
        val_a = (ha or {}).get("value_thousands", 0) or 0
        val_b = (hb or {}).get("value_thousands", 0) or 0
        opt = (hb or ha or {}).get("option_type", "") or ""
        diff_sh = sh_b - sh_a
        diff_val = val_b - val_a
        chg_pct_sh = round((diff_sh / sh_a * 100), 2) if sh_a else (100 if sh_b else 0)
        chg_pct_val = round((diff_val / val_a * 100), 2) if val_a else (100 if val_b else 0)
        rows.append({
            "cusip": cusip,
            "issuer_name": issuer_name,
            "option_type": opt,
            "shares_a": sh_a,
            "shares_b": sh_b,
            "diff_shares": diff_sh,
            "chg_pct_shares": chg_pct_sh,
            "value_a": val_a,
            "value_b": val_b,
            "diff_value": diff_val,
            "chg_pct_value": chg_pct_val,
        })
    return {
        "filer_name": data_b.get("filer_name", ""),
        "accession_a": accession_a,
        "accession_b": accession_b,
        "filing_date_a": data_a.get("filing_date", ""),
        "filing_date_b": data_b.get("filing_date", ""),
        "period_end_a": data_a.get("period_end", ""),
        "period_end_b": data_b.get("period_end", ""),
        "value_a": data_a.get("value_thousands", 0),
        "value_b": data_b.get("value_thousands", 0),
        "rows": rows,
    }


def get_holders_by_cusip(cusip: str, institution_ciks: List[str], year: int, quarter: int) -> List[Dict]:
    """Return list of {cik, filer_name, shares, value_thousands, pct} for each institution that holds this CUSIP."""
    cusip = (cusip or "").strip().upper()
    if not cusip:
        return []
    result = []
    for cik in institution_ciks:
        data = get_13f_holdings_by_quarter(cik, year, quarter)
        if not data:
            continue
        for h in data.get("holdings", []):
            if (h.get("cusip") or "").strip().upper() == cusip:
                result.append({
                    "cik": _cik_pad(cik),
                    "filer_name": data.get("filer_name", ""),
                    "shares": h.get("shares", 0),
                    "value_thousands": h.get("value_thousands", 0),
                    "pct": h.get("pct", 0),
                })
                break
    return result


def get_overlap_holdings(cik_list: List[str], year: int, quarter: int) -> List[Dict]:
    """Return holdings (CUSIP, issuer_name) that appear in all selected institutions for the quarter."""
    if len(cik_list) < 2:
        return []
    sets_by_cusip = {}
    issuer_by_cusip = {}
    for cik in cik_list:
        data = get_13f_holdings_by_quarter(cik, year, quarter)
        if not data:
            continue
        for h in data.get("holdings", []):
            cusip = (h.get("cusip") or "").strip()
            if not cusip:
                continue
            if cusip not in sets_by_cusip:
                sets_by_cusip[cusip] = set()
                issuer_by_cusip[cusip] = h.get("issuer_name", "")
            sets_by_cusip[cusip].add(_cik_pad(cik))
    ciks_padded = {_cik_pad(c) for c in cik_list}
    overlap = []
    for cusip, holders in sets_by_cusip.items():
        if ciks_padded.issubset(holders):
            overlap.append({
                "cusip": cusip,
                "issuer_name": issuer_by_cusip.get(cusip, ""),
            })
    return overlap
