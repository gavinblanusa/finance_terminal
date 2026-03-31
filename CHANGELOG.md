# Changelog

All notable changes to this project are documented here. Version format: `MAJOR.MINOR.PATCH.MICRO`.

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
