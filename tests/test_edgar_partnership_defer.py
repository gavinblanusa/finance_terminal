"""Partnership cap defer / hydrate (no SEC, no yfinance)."""

import json
from unittest.mock import patch

import pytest

import edgar_service
from edgar_service import hydrate_partnership_market_caps, partnership_events_caps_deferred


@pytest.fixture
def edgar_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar_service, "CACHE_DIR", tmp_path)
    return tmp_path


def test_partnership_events_caps_deferred_true(edgar_cache_dir):
    p = edgar_cache_dir / "partnership_events.json"
    p.write_text(
        json.dumps({"events": [{"filer_ticker": "X"}], "caps_enriched": False}),
        encoding="utf-8",
    )
    assert partnership_events_caps_deferred() is True


def test_partnership_events_caps_deferred_legacy_missing_key(edgar_cache_dir):
    p = edgar_cache_dir / "partnership_events.json"
    p.write_text(json.dumps({"events": [{"filer_ticker": "X"}]}), encoding="utf-8")
    assert partnership_events_caps_deferred() is False


@patch("partnership_enrichment.enrich_partnership_with_caps")
def test_hydrate_persists_caps_enriched(mock_enrich, edgar_cache_dir):
    mock_enrich.return_value = [{"filer_ticker": "X", "filer_market_cap": 1e9}]
    p = edgar_cache_dir / "partnership_events.json"
    p.write_text(
        json.dumps({"events": [{"filer_ticker": "X"}], "caps_enriched": False}),
        encoding="utf-8",
    )
    out = hydrate_partnership_market_caps([{"filer_ticker": "X"}])
    assert out[0].get("filer_market_cap") == 1e9
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("caps_enriched") is True


@patch("partnership_enrichment.enrich_partnership_with_caps")
def test_hydrate_no_op_when_already_enriched(mock_enrich, edgar_cache_dir):
    p = edgar_cache_dir / "partnership_events.json"
    p.write_text(
        json.dumps({"events": [{"filer_ticker": "X"}], "caps_enriched": True}),
        encoding="utf-8",
    )
    hydrate_partnership_market_caps([{"filer_ticker": "X"}])
    mock_enrich.assert_not_called()
