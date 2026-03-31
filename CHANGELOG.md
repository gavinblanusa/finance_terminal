# Changelog

All notable changes to this project are documented here. Version format: `MAJOR.MINOR.PATCH.MICRO`.

## [0.1.0.0] - 2026-03-31

### Added

- **Partnerships signal layer:** scoring (`signal_score`, `signal_reasons`), filer market-cap band, display excerpts, counterparty aliases and interest hits, yfinance enrichment batched on refresh and stale-cache re-read (`partnership_signal.py`, `partnership_enrichment.py`, `partnerships_config.py`, `edgar_service.py`).
- **Partnerships UI:** combined Signal column, cap filters including “All (dim outside band)”, partial refresh warnings, About expander, inspect with full “Why”.
- **Design system doc** `DESIGN.md` and **CLAUDE.md** routing; tabular mono styling pass across Dashboard, Market/Macro, Portfolio, IPO, 13F, Partnerships.

### Changed

- Dependencies: setuptools + pandas-datareader for Python 3.12 FRED macro path.
- `.gitignore`: ignore `.gstack/` local tooling output.

### Fixed

- Nav chrome: emoji removed from primary nav titles per design review.
