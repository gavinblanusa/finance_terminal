"""Unit tests for EDGAR partnership pipeline helpers (no network)."""

from edgar_service import (
    _counterparty_interest_hit,
    _find_primary_doc_from_index,
    _index_htm_candidates,
    _is_8k_disk_cache_valid,
    _is_8k_skip_cached,
    _is_exhibit_991_filename,
    _is_item_101_filing,
    _parse_columnar_8ks,
    _parse_recent_8ks,
    _pick_exhibit_991_filename,
    _retry_after_sleep_seconds,
)


def test_parse_recent_8ks_extracts_8k_rows():
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "8-K"],
                "accessionNumber": ["000-1", "000-2", "000-3"],
                "filingDate": ["2025-01-01", "2025-02-01", "2025-03-01"],
                "primaryDocument": ["q.htm", "eightk.htm", "eightk2.htm"],
            }
        }
    }
    rows = _parse_recent_8ks(submissions)
    assert len(rows) == 2
    assert rows[0]["accessionNumber"] == "000-2"
    assert rows[0]["primaryDocument"] == "eightk.htm"


def test_index_htm_candidates_dedupes():
    html = """
    <a href="cover.htm">x</a>
    <a href="/Archives/edgar/data/1/acc/cover.htm">y</a>
    <a href="ix?doc=/Archives/edgar/data/1/acc/ex99-1.htm">z</a>
    """
    c = _index_htm_candidates(html)
    assert "cover.htm" in c
    assert "ex99-1.htm" in c
    assert len([x for x in c if x == "cover.htm"]) == 1


def test_find_primary_doc_prefers_hint():
    html = """
    <a href="ex99-1.htm"></a>
    <a href="nvda-20250101.htm"></a>
    """
    got = _find_primary_doc_from_index(html, "000-000-000", "nvda-20250101.htm")
    assert got == "nvda-20250101.htm"


def test_is_exhibit_991_filename():
    assert _is_exhibit_991_filename("ex99-1.htm")
    assert _is_exhibit_991_filename("exhibit99-1_draft.htm")
    assert _is_exhibit_991_filename("ex-99-1.htm")
    assert not _is_exhibit_991_filename("ex99-2.htm")
    assert not _is_exhibit_991_filename("index.htm")


def test_pick_exhibit_991_skips_primary():
    cands = ["cover.htm", "ex99-1.htm", "ex99-2.htm"]
    assert _pick_exhibit_991_filename(cands, "cover.htm") == "ex99-1.htm"
    assert _pick_exhibit_991_filename(["cover.htm", "ex99-1.htm"], "ex99-1.htm") is None


def test_item_101_detection_with_exhibit_body():
    primary_thin = "8-K summary without item markers"
    exhibit = (
        "Item 1.01 Entry into a Material Definitive Agreement. "
        "The Company entered into a strategic partnership with ExampleCo Inc."
    )
    combined = f"{primary_thin}\n\n--- Exhibit 99.1 ---\n\n{exhibit}"
    assert not _is_item_101_filing(primary_thin)
    assert _is_item_101_filing(combined)


def test_counterparty_interest_hit_matches_aliases():
    assert _counterparty_interest_hit("OpenAI OpCo LLC")
    assert not _counterparty_interest_hit("Totally Unrelated Corp")


def test_parse_columnar_8ks_bulk_file_shape():
    """Bulk CIK-submissions-001.json has columnar arrays at root (not under filings.recent)."""
    bulk = {
        "form": ["8-K", "10-Q"],
        "accessionNumber": ["000-1", "000-2"],
        "filingDate": ["2015-01-02", "2015-03-01"],
        "primaryDocument": ["eightk.htm", "q.htm"],
    }
    rows = _parse_columnar_8ks(bulk)
    assert len(rows) == 1
    assert rows[0]["accessionNumber"] == "000-1"
    assert rows[0]["primaryDocument"] == "eightk.htm"


def test_is_8k_disk_cache_valid():
    legacy = {"event": {"x": 1}}
    assert _is_8k_disk_cache_valid(legacy, "any.htm", "2020-01-01") is True

    row = {"event": {"x": 1}, "source_primary_document": "a.htm", "source_filing_date": "2020-01-01"}
    assert _is_8k_disk_cache_valid(row, "a.htm", "2020-01-01") is True
    assert _is_8k_disk_cache_valid(row, "b.htm", "2020-01-01") is False
    assert _is_8k_disk_cache_valid(row, "a.htm", "2020-01-02") is False


def test_is_8k_skip_cached_negative_hit():
    row = {
        "event": None,
        "skip_reason": "no_item_101",
        "source_primary_document": "a.htm",
        "source_filing_date": "2020-01-01",
    }
    assert _is_8k_skip_cached(row, "a.htm", "2020-01-01") is True
    assert _is_8k_skip_cached(row, "b.htm", "2020-01-01") is False
    assert _is_8k_skip_cached(row, "a.htm", "2020-01-02") is False


def test_is_8k_skip_cached_ignored_when_event_present():
    row = {
        "event": {"filer_ticker": "NVDA"},
        "skip_reason": "financing",
        "source_primary_document": "a.htm",
        "source_filing_date": "2020-01-01",
    }
    assert _is_8k_skip_cached(row, "a.htm", "2020-01-01") is False


def test_is_8k_skip_cached_requires_reason():
    row = {"event": None, "source_primary_document": "a.htm", "source_filing_date": "2020-01-01"}
    assert _is_8k_skip_cached(row, "a.htm", "2020-01-01") is False


def test_retry_after_sleep_seconds():
    assert _retry_after_sleep_seconds("15", 3) == 15.0
    assert _retry_after_sleep_seconds(None, 2) == 4.0
    assert 1.0 <= _retry_after_sleep_seconds("Wed, 11 Jun 3025 12:00:00 GMT", 0) <= 120.0
