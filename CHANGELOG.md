# Changelog

All notable changes to this project are documented here. Version format: `MAJOR.MINOR.PATCH.MICRO`.

## [0.1.3.0] - 2026-04-01

### Added

- **OpenBB data layer:** shared fetch kernel (`openbb_fetch.py`) with provider chains, timeouts, and logging; `openbb_provider_registry.py` as the single provider order source; adapter routes for profile, fundamentals, news, macro (FRED when `FRED_API_KEY` is set, else pandas-datareader), IPO historical price, and more.
- **CI:** GitHub Actions runs `pytest` with `USE_OPENBB=true` and `USE_OPENBB=false`, plus `verify_openbb_coverage_doc.py` so registry chains stay documented in `docs/OPENBB_COVERAGE.md`.
- **Docs:** **`docs/OPENBB_COVERAGE.md`** (what uses OpenBB and provider try-order), env and architecture updates, and **`TODOS.md`** for backlog tracking.

### Fixed

- **Market Analysis fundamentals:** **Revenue (TTM)** now sums the correct income-statement column (prefers `total_revenue`, ignores `cost_of_revenue` and other non-total lines that matched a naive `"revenue"` substring). Numeric strings from providers are coerced before summing.

## [0.1.2.0] - 2026-03-31

### Changed

- **Partnerships tab** loads faster: **Refresh** reuses the saved SEC filing index unless you check **re-fetch SEC filing index**; **8-K** rows we already skip (not Item 1.01 or not financing) are remembered so we do not re-parse them.
- **Market caps:** Yahoo pulls run in **parallel** with a **disk cache** (TTL); on first load you get signals quickly and **full cap enrich** runs when you pick a **cap band** filter or use **Load market caps** on **All**.

### Added

- **Tests:** `tests/test_edgar_partnership_defer.py`, `tests/test_partnership_enrichment.py`; more cases in `tests/test_edgar_service_partnerships.py`.

## [0.1.1.0] - 2026-03-31

### Added

- **Partnerships EDGAR pipeline:** SEC `GET` retries with **429** handling (**Retry-After** as seconds or HTTP-date) and **5xx** backoff; **atomic** cache writes; **global date merge** of 8-K candidates across the watchlist so late tickers are not starved; optional **Exhibit 99.1** merge for Item 1.01 text; **per-accession cache** keys on primary document and filing date so metadata drift triggers a refetch; optional older 8-K rows via **`EDGAR_EXTRA_SUBMISSION_JSON_FILES_PER_CIK`** in `partnerships_config.py` (default **0**). Counterparty **interest** flags at ingest use the same rules as `partnership_signal`.
- **Tests:** `tests/test_edgar_service_partnerships.py` for columnar parsing, cache validity, retry-after helper.

### Changed

- **Events cache** `cache_schema_version` bumped to **4** with the above behavior.
- **Streamlit** `.streamlit/config.toml`: **`base = dark`**, **`borderColor`**, and **`[theme.sidebar]`** explicit colors to reduce sidebar widget empty-color console warnings.

## [0.1.0.0] - 2026-03-31

### Added

- **Partnerships signal layer:** ranked Item 1.01 rows with `signal_score` / `signal_reasons`, filer market-cap band, short excerpts, counterparty aliases and interest hits, batched yfinance enrichment on refresh and when cache rows need a newer signal (`partnership_signal.py`, `partnership_enrichment.py`, `partnerships_config.py`, `edgar_service.py`).
- **Partnerships UI:** one **Signal** column (score + hits + reasons), cap filters including **All (dim outside band)**, refresh warnings when some tickers skip, **About** expander, **Inspect** for full excerpt and “why”.
- **Design system** `DESIGN.md` and **CLAUDE.md** skill routing; tabular mono styling pass on Dashboard, Market/Macro, Portfolio, IPO, 13F, Partnerships.

### Changed

- Dependencies: setuptools + pandas-datareader for Python 3.12 FRED macro path.
- `.gitignore`: ignore `.gstack/` local tooling output.

### Fixed

- Nav chrome: emoji removed from primary sidebar titles (design review).
