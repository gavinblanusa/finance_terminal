"""Unit tests for partnership_signal (no SEC, no Streamlit)."""

import types

from partnership_signal import (
    SIGNAL_VERSION,
    build_signal_reasons,
    enrich_event_dict,
    events_need_signal_refresh,
    filer_in_cap_band,
    format_display_excerpt,
    resolve_counterparty_hits,
    score_partnership_event,
)


def test_format_display_excerpt_truncates():
    long = "x" * 300
    out = format_display_excerpt(long, max_len=50)
    assert len(out) == 50
    assert out.endswith("…")
    assert format_display_excerpt("  hello  world  ") == "hello world"


def test_filer_in_cap_band():
    assert filer_in_cap_band(None, 1, 10) is None
    assert filer_in_cap_band(-1, 1, 10) is None
    assert filer_in_cap_band(5e8, 5e8, 20e9) is True
    assert filer_in_cap_band(1e8, 5e8, 20e9) is False


def test_resolve_counterparty_hits():
    cps = [{"name": "Something OpenAI Holdings LLC"}, {"name": "Random Corp"}]
    aliases = {"OpenAI": ["OpenAI"]}
    interest = ["Stripe"]
    hit, labels = resolve_counterparty_hits(cps, interest, aliases)
    assert hit is True
    assert "OpenAI" in labels


def test_score_partnership_event_ordering():
    s_high = score_partnership_event("partnership", True, True, 2)
    s_low = score_partnership_event("other", False, False, 0)
    assert s_high > s_low


def test_build_signal_reasons_includes_interest():
    reasons = build_signal_reasons("partnership", ["OpenAI"], True, 1)
    assert any("Interest" in r for r in reasons)
    assert any("Strategic" in r for r in reasons)


def test_name_matches_word_boundary_ramp_not_rampage():
    """4+ char aliases use token boundaries so Ramp does not match Rampage."""
    from partnership_signal import _name_matches_fragment

    assert _name_matches_fragment("Ramp Inc", "Ramp") is True
    assert _name_matches_fragment("Rampage Technologies LLC", "Ramp") is False
    assert _name_matches_fragment("OpenAI LP", "OpenAI") is True


def test_events_need_signal_refresh_any_row():
    assert events_need_signal_refresh([{"signal_version": 1}, {"signal_version": SIGNAL_VERSION}]) is True
    assert events_need_signal_refresh([{"signal_version": SIGNAL_VERSION}]) is False
    assert events_need_signal_refresh([]) is False


def test_enrich_event_dict_sets_signal_version():
    cfg = types.SimpleNamespace(
        COUNTERPARTY_INTEREST_NAMES=["AcmeCo"],
        COUNTERPARTY_ALIASES={"AcmeCo": ["AcmeCo"]},
        FILER_CAP_USD_MIN=500_000_000,
        FILER_CAP_USD_MAX=20_000_000_000,
    )
    event = {
        "filer_ticker": "TEST",
        "relevance_type": "partnership",
        "counterparties": [{"name": "AcmeCo Inc", "is_interest": False}],
        "snippet": "entered into a strategic partnership with AcmeCo Inc for purposes of testing.",
    }
    out = enrich_event_dict(event, 1e9, cfg)
    assert out["signal_version"] == SIGNAL_VERSION
    assert out["interest_hit"] is True
    assert out["filer_in_cap_band"] is True
    assert out["signal_score"] >= 40
    assert out["display_excerpt"]
