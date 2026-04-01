# Project learnings

Cross-session notes from gstack (`/learn`). Source: `~/.gstack/projects/<slug>/learnings.jsonl` (slug from git `origin`, not the local folder name). Regenerate or extend via `/learn export` / `/learn add` in the repo root.

## Project Learnings

### Patterns

- **[partnerships-config-knobs]**: `partnerships_config.py` holds WATCH_TICKERS, counterparty interest names and aliases, and filer cap min/max; scoring and filters depend on keeping this file accurate. (confidence: 9/10)
- **[partnership-signal-unit-tests]**: `tests/test_partnership_signal.py` locks scoring and name-match behavior without hitting SEC or Streamlit. (confidence: 8/10)
- **[partnership-interest-match-at-ingest]**: `edgar_service` uses `partnership_signal.resolve_counterparty_hits` for counterparty `is_interest` at 8-K ingest so it matches the same aliases and token rules as the signal layer. (confidence: 9/10)

### Pitfalls

- **[streamlit-app-import-path]**: Run `streamlit run app/main.py` from repo root; modules under `app/` use imports like `from models import Base`, so `app/` must be on `sys.path`. (confidence: 9/10)

### Preferences

- _(none in export set)_

### Architecture

- **[partnership-8k-relevance-filter]**: Partnerships tab surfaces Item 1.01 rows classified as partnership or optional other; routine financing 8-Ks are filtered out in `edgar_service` so the list stays strategic. (confidence: 9/10)
- **[partnership-edgar-cache-and-warnings]**: Events are cached at `.edgar_cache/partnership_events.json`; `refresh_edgar_data` returns `(events, warnings)` so the UI can show watchlist tickers skipped (e.g. missing CIK). (confidence: 9/10)
- **[partnership-signal-enrichment]**: `partnership_signal` and `partnership_enrichment` add score, reasons, excerpts, interest hits, and filer cap band using batched yfinance; stale `signal_version` triggers enrich on read. (confidence: 9/10)
- **[disk-caches-at-repo-root]**: Warm data lives under `.market_cache`, `.ipo_cache`, and `.edgar_cache` at the project root; deleting them forces a cold refill on next use. (confidence: 9/10)
- **[edgar-sec-get-and-atomic-cache]**: Partnerships SEC fetches use `edgar_service._sec_get`: throttle, 429 Retry-After (seconds or HTTP-date), 5xx backoff; `_save_json` writes temp then `os.replace` for atomic cache files. (confidence: 9/10)
- **[edgar-8k-refresh-merge-exhibit-cache]**: `refresh_edgar_data` merges 8-K rows from `filings.recent` (per ticker, capped), optionally first N bulk `CIK-submissions-NNN.json` via `EDGAR_EXTRA_SUBMISSION_JSON_FILES_PER_CIK`, sorts by filingDate, dedupes accession; `_process_8k` appends Exhibit 99.1 HTML text when index lists it; per-accession JSON cache stores `source_primary_document` and `source_filing_date` to invalidate on SEC metadata drift. (confidence: 9/10)
- **[partnerships-lazy-caps-and-submissions-cache]**: Partnerships refresh reuses on-disk SEC submissions index unless `force_submissions_refresh`; dropped 8-K accessions are negatively cached; Streamlit loads events with `defer_yfinance` so signal enrichment runs before Yahoo caps; filer caps use parallel yfinance and `.edgar_cache/partnership_filer_market_caps.json` TTL; `partnership_events.json` tracks `caps_enriched` and `hydrate_partnership_market_caps` fills caps when the user picks a cap band or Load market caps on All. (confidence: 9/10)

### Tools

- **[gstack-learnings-slug]**: gstack project slug for learnings paths is derived from git `origin` (`user/repo`), not the local clone folder name; run `gstack-learnings-*` from the repo root. (confidence: 10/10)
