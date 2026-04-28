# SEC Form 4 Insider Activity Upgrade

## Summary

- Upgrade Market Analysis > Corporate Activity so insider transactions use direct SEC EDGAR Form 4 data as the canonical free source.
- Keep Finnhub as a fallback when configured, and add an OpenInsider screener link only as a human cross-check.
- Scope is Market Analysis only; no Dashboard, radar, database migration, or OpenInsider scraping.

## Key Changes

- Add a focused insider data service that reuses the app's existing SEC/EDGAR utilities to map ticker to CIK, fetch recent `4` and `4/A` filings, download filing XML, and normalize transactions.
- Preserve `market_data.fetch_insider_transactions(ticker, from_date=None, to_date=None)` so existing UI/chart integrations keep working.
- Return normalized rows with the existing fields plus useful metadata: `source`, `filing_date`, `price`, `open_market`, `is_officer`, `is_director`, and SEC filing link.
- Cache SEC insider results under `.edgar_cache/` with a short TTL, and fall back to the current Finnhub path if SEC fetch/parsing yields no usable rows.
- Update the Corporate Activity insider table with source labeling, OpenInsider cross-check link, and compact filters for open-market buys, minimum value, and officer/director rows.
- Document the source order: SEC Form 4 primary, Finnhub fallback, OpenInsider link for verification only.

## Tests And QA

- Add unit tests for SEC Form 4 XML parsing: purchases, sales, officer/director flags, missing prices, malformed numeric fields, and ignored non-Form-4 filings.
- Add orchestration tests confirming SEC-first behavior and Finnhub fallback without live network calls.
- Run:
  - `pytest tests/test_insider_service.py -q`
  - `pytest tests/`
  - `ruff check app/ tests/`
- Manual QA:
  - Start Streamlit on port `8501`.
  - Check Market Analysis for tickers like `AAPL` and `NVDA`.
  - Confirm the insider table renders without API keys, filters work, SEC links open, and OpenInsider link points to the ticker screener.
  - Confirm empty/no-data states are clear and do not mention Finnhub as required.

## Execution Notes

- Use TDD: write failing parser/orchestration tests first, then implement, then refactor.
- Commit only after verification passes with message: `feat: add SEC insider activity source`.

## Assumptions

- Direct SEC EDGAR is the source of truth because it is free, official, and current enough for this terminal.
- OpenInsider is useful for user confidence, but not reliable enough to scrape as an application dependency.
- No new third-party dependency is required; use Python stdlib XML parsing and existing SEC helper patterns.
