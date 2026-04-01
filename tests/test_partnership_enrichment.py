"""Unit tests for partnership_enrichment (mocked yfinance; isolated disk cache)."""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import partnership_enrichment
from partnership_enrichment import enrich_partnership_signals_only, fetch_market_caps_yf


@pytest.fixture(autouse=True)
def isolated_caps_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        partnership_enrichment,
        "_CAPS_CACHE_PATH",
        tmp_path / "partnership_filer_market_caps.json",
    )


@patch("partnership_enrichment._market_cap_for_ticker")
def test_fetch_market_caps_dedupes(mock_cap):
    mock_cap.side_effect = lambda t: {"A": 1e9, "B": 2e9}.get(t)
    got = fetch_market_caps_yf(["A", "a", "B", "A"], max_workers=4)
    assert got == {"A": 1e9, "B": 2e9}
    assert mock_cap.call_count == 2


@patch("partnership_enrichment._market_cap_for_ticker")
def test_fetch_market_caps_sequential_when_max_workers_one(mock_cap):
    mock_cap.side_effect = lambda t: hash(t) % 1000
    got = fetch_market_caps_yf(["X", "Y", "Z"], max_workers=1)
    assert set(got.keys()) == {"X", "Y", "Z"}
    assert mock_cap.call_count == 3


@patch("partnership_enrichment._market_cap_for_ticker")
def test_fetch_market_caps_parallel_path(mock_cap):
    mock_cap.side_effect = lambda t: 1.0 if t == "P" else 2.0
    got = fetch_market_caps_yf(["P", "Q"], max_workers=4)
    assert got["P"] == 1.0 and got["Q"] == 2.0
    assert mock_cap.call_count == 2


@patch("partnership_enrichment._market_cap_for_ticker")
def test_fetch_market_caps_parallel_future_exception_becomes_none(mock_cap):
    def side(t):
        if t == "BAD":
            raise ValueError("simulated failure")
        return 3.0

    mock_cap.side_effect = side
    got = fetch_market_caps_yf(["GOOD", "BAD"], max_workers=4)
    assert got["GOOD"] == 3.0
    assert got["BAD"] is None


def test_fetch_market_caps_empty():
    assert fetch_market_caps_yf([]) == {}
    assert fetch_market_caps_yf(["", "  "]) == {}


@patch("partnership_enrichment._market_cap_for_ticker")
def test_disk_cache_hit_skips_network(mock_cap):
    now = datetime.now(timezone.utc)
    path = partnership_enrichment._CAPS_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    "ZZZ": {"market_cap_usd": 1.5e9, "cached_at": now.isoformat()},
                },
            }
        ),
        encoding="utf-8",
    )
    got = fetch_market_caps_yf(["ZZZ"])
    assert got["ZZZ"] == 1.5e9
    mock_cap.assert_not_called()


@patch("partnership_enrichment._market_cap_for_ticker")
def test_disk_cache_stale_refetches(mock_cap):
    old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    path = partnership_enrichment._CAPS_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    "WWW": {"market_cap_usd": 9e9, "cached_at": old.isoformat()},
                },
            }
        ),
        encoding="utf-8",
    )
    mock_cap.return_value = 8.0
    got = fetch_market_caps_yf(["WWW"])
    assert got["WWW"] == 8.0
    mock_cap.assert_called_once_with("WWW")


@patch("partnership_enrichment._market_cap_for_ticker")
def test_disk_cache_prune_drops_stale_keys(mock_cap):
    old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fresh = datetime.now(timezone.utc)
    path = partnership_enrichment._CAPS_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    "OLD": {"market_cap_usd": 1.0, "cached_at": old.isoformat()},
                    "NEW": {"market_cap_usd": 2.0, "cached_at": fresh.isoformat()},
                },
            }
        ),
        encoding="utf-8",
    )
    mock_cap.return_value = None
    fetch_market_caps_yf(["OLD"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "OLD" not in data.get("entries", {})
    assert "NEW" in data.get("entries", {})


def test_entry_cap_fresh_rejects_bad_float():
    from partnership_enrichment import _entry_cap_fresh

    now = datetime.now(timezone.utc)
    assert _entry_cap_fresh({"market_cap_usd": "x", "cached_at": now.isoformat()}, 86400.0, now) is None


def test_enrich_partnership_signals_only_does_not_touch_yahoo(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("fetch_market_caps_yf must not run for signals-only enrich")

    monkeypatch.setattr(partnership_enrichment, "fetch_market_caps_yf", boom)
    ev = {
        "filer_ticker": "AAPL",
        "filer_name": "Apple",
        "filing_date": "2025-01-01",
        "accession_number": "000-000-000",
        "sec_url": "https://example.com",
        "counterparties": [],
        "snippet": "strategic partnership with ExampleCo",
        "relevance_type": "partnership",
    }
    out = enrich_partnership_signals_only([ev])
    assert len(out) == 1
    assert out[0].get("signal_version") is not None
    assert out[0].get("filer_market_cap") is None
