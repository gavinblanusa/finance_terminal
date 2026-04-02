"""Tests for rule-based event tags on ranked news."""

from __future__ import annotations

from relevant_news import build_relevant_news, extract_event_tags


def test_extract_event_tags_order_and_cap() -> None:
    assert extract_event_tags("Company beats EPS expectations", "") == ["earnings"]
    t = extract_event_tags("FDA recall and merger talks", "")
    assert "fda" in t
    assert "m&a" in t
    assert len(t) <= 3


def test_extract_event_tags_empty() -> None:
    assert extract_event_tags("", "") == []


def test_build_relevant_news_tags_and_score_bump() -> None:
    def fake_news(ticker: str, limit: int):
        return [
            {
                "headline": "ACME Corp beats earnings, FDA clears device",
                "url": "https://example.com/a",
                "source": "Test",
                "datetime": None,
            }
        ]

    ranked = build_relevant_news(["ACME"], [], fake_news)
    assert len(ranked) == 1
    item = ranked[0]
    assert "earnings" in item.event_tags or "fda" in item.event_tags
    # base 2 portfolio + keyword bonus possible + high-impact tag bump
    assert item.score >= 2
