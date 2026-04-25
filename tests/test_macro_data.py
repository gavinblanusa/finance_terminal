"""Unit tests for macro data cache helpers (no network)."""

import json

import macro_data as md


def test_macro_file_cache_status_empty_dir(tmp_path, monkeypatch) -> None:
    """No .macro_cache yet: all metrics missing, no newest mtime."""
    monkeypatch.setattr(md, "MACRO_CACHE_DIR", tmp_path)
    newest, n_present, n_missing = md.macro_file_cache_status(["gdp", "cpi"])
    assert newest is None
    assert n_present == 0
    assert n_missing == 2


def test_macro_file_cache_status_partial_hits(tmp_path, monkeypatch) -> None:
    """One json file: count and newest mtime set."""
    (tmp_path / "gdp.json").write_text(
        json.dumps(
            [
                {"date": "2020-01-01", "value": 1.0},
            ]
        )
    )
    monkeypatch.setattr(md, "MACRO_CACHE_DIR", tmp_path)
    newest, n_present, n_missing = md.macro_file_cache_status(["gdp", "cpi"])
    assert n_present == 1
    assert n_missing == 1
    assert newest is not None
