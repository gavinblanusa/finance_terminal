# Open Source Repositories for Gavin Financial Terminal

This document catalogs open source GitHub repositories that could be used to **perform analysis**, **gather/find/leverage data**, or **display data** in the Gavin Financial Terminal. It is intended as a **main source of context for future AI agents** implementing a new feature or tool from this list.

**Audience:** AI agents and developers adding or refactoring features.  
**Project context:** See root `README.md` and `docs/ARCHITECTURE.md` for entry points, data flow, and where key logic lives.

**Document purpose:** To give implementers everything needed in one place: which repo fits where, exact function/module touchpoints, env vars, cache and DB conventions, validation steps, and a worked example. If you are an AI agent, read the "For implementers" section and the relevant repo subsection before writing code.

---

## Table of Contents

1. [Overview and grouping](#overview-and-grouping)
2. [For implementers: before you start, touchpoints, and validation](#for-implementers-before-you-start-touchpoints-and-validation)
3. [Analysis](#1-analysis)
4. [Data: gather, find, leverage](#2-data-gather-find-leverage)
5. [Display](#3-display)
6. [Quick reference and implementation order](#quick-reference-and-implementation-order)
7. [Example: minimal first integration (pandas-ta)](#example-minimal-first-integration-pandas-ta)
8. [Project file reference](#project-file-reference-for-implementers)

---

## Overview and grouping

Repos are grouped by primary function:

| Group | Purpose |
|-------|--------|
| **Analysis** | Technical indicators, fundamentals/valuation, backtesting, portfolio/risk metrics |
| **Data** | Market data APIs, SEC EDGAR (8-K, 13F), IPO calendars, news/sentiment |
| **Display** | Charts (TradingView/Bloomberg-style), Streamlit components, dashboard patterns |

**Project areas** these can plug into:

- **Dashboard** — `main.py` → `dashboard_page()`; uses `tax_engine`, `db`, `models`
- **Portfolio & Taxes** — `main.py` → `portfolio_taxes_page()`; uses `tax_engine`, `db`, `models`
- **Market Analysis** — `main.py` → `market_analysis_page()`; uses `market_data.py` (OHLCV, valuation, signals, profile, news), `api_clients.py`
- **IPO Vintage Tracker** — `main.py` → `ipo_tracker_page()`; uses `ipo_service.py`
- **Partnerships** — `main.py` → `partnerships_page()`; uses `edgar_service.py`, `partnerships_config.py`
- **13F Institutional Holdings** — `thirteenf_service.py`, `thirteenf_config.py` (if exposed in UI)

---

## For implementers: before you start, touchpoints, and validation

This section gives future AI agents enough context to integrate a repo safely and verify the result.

### Before you start (checklist)

1. **Read** `docs/ARCHITECTURE.md` so you know data flow (e.g. OHLCV → cache → chart; 8-K → `.edgar_cache/` → partnerships page). Changing the wrong layer can break caching or rate-limit handling.
2. **Read** the root `README.md` “For AI and refactors” section for entry points, env vars, and where key logic lives.
3. **Run the app** once (`streamlit run app/main.py` from project root) and use the page you will touch (e.g. Market Analysis, Partnerships). This establishes a baseline: what “working” looks like before your change.
4. **Check `.env`** for existing API keys. Many repos (OpenBB, FinanceToolkit, EdgarTools) can reuse `FMP_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `MASSIVE_API_KEY` (Polygon), etc. Do not duplicate keys; document in this file if a new env var is required.
5. **Pin new dependencies** in `requirements.txt` with a version (e.g. `pandas-ta>=0.3.14`) so upgrades are intentional.

### Concrete code touchpoints

These are the main functions and modules you will call, wrap, or replace when integrating. Use this table to avoid changing unrelated code.

| Area | Module | Key functions / entry points | Notes |
|------|--------|------------------------------|--------|
| **OHLCV & indicators** | `market_data.py` | `fetch_ohlcv(ticker)`, `calculate_sma`, `calculate_rsi`, `calculate_bollinger_bands`, `calculate_stochastic_rsi`, `calculate_tradingview_signals(df)`, `save_tv_signals_to_cache`, `load_tv_signals_from_cache` | OHLCV comes from yfinance + `api_clients`; indicators run on the returned DataFrame. Signal cache: `.market_cache/{TICKER}_tv_signals_{timeframe}.json`. |
| **Valuation / fundamentals** | `market_data.py` | `fetch_valuation_data`, `get_valuation_chart_data`, `load_valuation_from_db`, `save_valuation_to_db`, `get_company_profile`, `get_fundamentals_ratios`, `fetch_company_news` | DB-first for valuation; FMP, Alpha Vantage, Finnhub, yfinance. |
| **Price chart (UI)** | `main.py` | `market_analysis_page()` — where the price chart is rendered | Primary chart: **streamlit-lightweight-charts** via `chart_utils` (OHLCV→candlestick/volume). Receives OHLCV from `fetch_ohlcv()`; Plotly still used for TradingView signals chart. |
| **8-K / Partnerships** | `edgar_service.py` | `get_partnership_events()`, `refresh_edgar_data()`, `_process_8k()`, `_parse_recent_8ks()`, `_is_item_101_filing()`, `_extract_counterparties()` | Cache: `.edgar_cache/partnership_events.json`. Config: `partnerships_config.py` (watched tickers, counterparty names). |
| **IPO calendar** | `ipo_service.py` | `fetch_ipo_calendar()`, `_parse_ipo_data()`, `get_vintage_performance()`, `get_ipo_price_history()` | Cache: `.ipo_cache/ipo_calendar_30d.json`. Finnhub today; finance_calendars would feed same or alternate calendar. |
| **13F** | `thirteenf_service.py` | `get_13f_filings_for_institution()`, `get_13f_holdings()`, `get_13f_holdings_by_quarter()` | Cache: `.edgar_cache/13f/`. Config: `thirteenf_config.py` (CIK list). |
| **DB session** | `main.py`, `db.py` | `get_db_session()` | Pass the same session into TaxEngine and any code that reads/writes `Trades`, `ValuationHistory`, `CompanyProfile`, etc. Do not create a new engine in feature modules. |

### How to validate an integration

- **Market Analysis (indicators / chart):** Run app → Market Analysis → pick a ticker (e.g. AAPL). Confirm price chart and signals load; if you changed indicators, compare a few values (e.g. RSI) with a known source or previous run. Check that cache files are still written under `.market_cache/` and that “Save TV” (if present) still works.
- **Market Analysis (valuation):** Same page, open valuation chart. Confirm P/E and/or revenue load; if you switched to FinanceToolkit or OpenBB, spot-check one ratio against FMP or yfinance.
- **Partnerships:** Run app → Partnerships. Confirm 8-K partnership events list loads and counterparties appear. If you replaced `edgar_service` with EdgarTools, run `refresh_edgar_data()` once and confirm `.edgar_cache/partnership_events.json` is updated and the UI still renders.
- **IPO Tracker:** Run app → IPO Vintage Tracker. Confirm calendar and vintage performance load. If you added finance_calendars, ensure date/field mapping matches what the UI expects.
- **Regression:** After any change, run Dashboard and Portfolio & Taxes and confirm portfolio value and tax summary still compute (no import errors or missing session).

### Env vars used by this project (and by some repos)

| Variable | Used in project | Used by open source repos (this doc) |
|----------|------------------|--------------------------------------|
| `FMP_API_KEY` | `market_data.py` (profile, fundamentals) | FinanceToolkit, OpenBB (FMP provider) |
| `ALPHA_VANTAGE_API_KEY` / `ALPHA_API_KEY` | `market_data.py`, `tax_engine.py` | OpenBB (Alpha Vantage provider) |
| `FINNHUB_API_KEY` | `market_data.py`, `ipo_service.py` | OpenBB |
| `MASSIVE_API_KEY` | `api_clients.py` (Polygon) | OpenBB (Polygon provider) |
| `TWELVEDATA_API_KEY`, `EODHD_API_KEY` | `api_clients.py` | OpenBB (if configured) |
| `USE_FINANCETOOLKIT` | `financetoolkit_adapter.py` (feature flag) | Set to `false` to disable FinanceToolkit and use OpenBB/FMP only |
| EdgarTools identity | — | EdgarTools: `set_identity("Name <email>")` in code; not in `.env` by default |

---

## 1. Analysis

Repositories that add or improve **technical analysis**, **fundamental/valuation metrics**, **backtesting**, or **portfolio/risk analytics**.

---

### 1.1 pandas-ta (Technical indicators)

| Field | Value |
|-------|--------|
| **Repository** | [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta) |
| **License** | MIT |
| **Stars** | ~6.1k |
| **Language** | Python 3 |
| **Install** | `pip install pandas-ta` |

**Description**  
Pandas TA is a Pandas extension (`.ta`) providing **150+ technical analysis indicators** in pure Python using numba and numpy. No TA-Lib required. Results align closely with TA-Lib where comparable.

**Key features**

- 150+ indicators (momentum, overlap, volatility, trend, etc.)
- Optional 60+ candlestick patterns when TA-Lib is installed
- DataFrame integration: `df.ta.sma(20)`, `df.ta.rsi()`, etc.
- Includes RSI, Stochastic RSI (stochrsi), Bollinger Bands, SMA, EMA, and many more
- Custom strategy/indicator directory support

**Where it fits in this project**

- **Primary:** `market_data.py` — replace or extend custom indicator logic (SMA, RSI, Bollinger Bands) and **TradingView-style signals** (e.g. Stochastic RSI, momentum, supply/demand). Signals are currently cached per ticker/timeframe in `.market_cache/` and optionally DB.
- **Secondary:** Any new indicator-based screens or strategy ideas that consume OHLCV from `fetch_ohlcv()`.

**Integration notes for implementers**

- OHLCV is provided by `market_data.fetch_ohlcv()` (and `api_clients` as fallback). Pass the returned DataFrame (or its `Open/High/Low/Close/Volume` columns) into pandas-ta.
- Preserve existing cache keys and DB usage for signals (see `docs/MARKET_ANALYSIS_DATA_REFACTOR.md`) when adding indicators.
- Rate limits still apply to upstream data (Alpha Vantage, Polygon, etc.); caching remains important.

**Gotchas / considerations**

- Some indicators have different default parameters than TA-Lib or TradingView; document or align params if matching external references.
- Optional TA-Lib adds candlestick patterns but adds a native dependency; pure-Python mode is sufficient for most indicators.

---

### 1.2 Freqtrade Technical

| Field | Value |
|-------|--------|
| **Repository** | [freqtrade/technical](https://github.com/freqtrade/technical) |
| **License** | Check repo (GPL-3.0 for freqtrade) |
| **Stars** | ~975 |
| **Language** | Python |
| **Install** | Typically as part of freqtrade or via repo |

**Description**  
Technical indicator and utility library used by the Freqtrade bot project. Wraps TA-Lib, PyTi, and other sources into a single API for strategy development.

**Key features**

- Unified API over multiple indicator libraries
- Geared toward automated trading strategies
- Many indicators available in one place

**Where it fits in this project**

- **Optional alternative** to pandas-ta for Market Analysis indicators if you need strategy-oriented naming or additional wrappers. Less central than pandas-ta for a Streamlit-first app; pandas-ta is a better fit for ad-hoc and cached indicators.

**Integration notes**

- Evaluate license (GPL) if shipping as part of a closed product. For personal/internal use it can still be referenced for logic or patterns.
- Would plug into the same `market_data.py` signal/indicator pipeline as pandas-ta.

---

### 1.3 VectorBT (Backtesting and portfolio analysis)

| Field | Value |
|-------|--------|
| **Repository** | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) |
| **License** | MIT |
| **Stars** | ~6.7k |
| **Language** | Python |
| **Install** | `pip install vectorbt` or `pip install "vectorbt[full]"` for optional deps |

**Description**  
High-performance backtesting and quantitative analysis using **vectorized** operations (pandas/NumPy + Numba). Designed to test many strategy variants quickly and build interactive dashboards (Plotly, Jupyter).

**Key features**

- Vectorized backtesting (no event loop); fast iteration over parameter grids
- Built-in portfolio metrics (returns, Sharpe, drawdown, etc.)
- Plotly-based interactive charts
- Can pull price data (e.g. yfinance) or accept your own DataFrames
- Portfolio optimization and time-series analysis use cases

**Where it fits in this project**

- **Market Analysis:** Add a “Backtest” or “Strategy” view that runs a simple strategy (e.g. SMA crossover, RSI threshold) on cached OHLCV and shows equity curve and metrics.
- **Dashboard / Portfolio:** Optional “strategy performance” or “what-if” analytics using portfolio or watchlist tickers.
- Data source: reuse `market_data.fetch_ohlcv()` or DB/cache so as not to duplicate API calls.

**Integration notes for implementers**

- Use existing session/cache discipline: e.g. load OHLCV from `.market_cache/` or DB, then pass to VectorBT; avoid triggering new API calls from inside VectorBT’s data layer if you already have rate limits.
- A separate Streamlit page or sub-section under Market Analysis is a natural place (see also **marketcalls/vectorbt-streamlit** in Display).
- `requirements.txt`: add `vectorbt` (and optionally `vectorbt[full]` if using all viz features).

**Gotchas**

- Full install can pull in heavier dependencies; start with minimal `vectorbt` and add as needed.
- VectorBT has its own conventions for signals/orders; map your indicator outputs (e.g. from pandas-ta) into its format.

---

### 1.4 FinanceToolkit (Fundamental ratios and valuation)

| Field | Value |
|-------|--------|
| **Repository** | [JerBouma/FinanceToolkit](https://github.com/JerBouma/FinanceToolkit) |
| **License** | MIT |
| **Stars** | ~4.4k+ |
| **Language** | Python |
| **Install** | `pip install financetoolkit -U` |

**Description**  
Toolkit that computes **180+ financial ratios and indicators** with **transparent, documented methodology**. Aims to fix inconsistencies across data providers (e.g. P/E differing between sources) by exposing how each ratio is calculated.

**Key features**

- 50+ ratios in five categories: Efficiency, Liquidity, Profitability, Solvency, Valuation
- Works with FMP (API key) or Yahoo Finance as fallback
- Supports equities, options, ETFs, indices, currencies, crypto, commodities
- Can pair with “Finance Database” for peer/competitor analysis

**Where it fits in this project**

- **Market Analysis:** Valuation chart (P/E, revenue over time) and company fundamentals. Currently `market_data.py` uses FMP, Alpha Vantage, Finnhub for valuation and `get_fundamentals_ratios`; FinanceToolkit can **standardize** ratio definitions and calculations.
- **Company profile/fundamentals:** Align metrics with a single, documented source of truth and optionally persist key ratios in `company_fundamentals` or `valuation_history` (see `models.py`).

**Integration notes for implementers**

- FMP API key is already used in the project; FinanceToolkit can use the same key.
- Keep DB-first behavior: compute ratios then `save_valuation_to_db` / update company fundamentals so the app remains cache/DB-first (see `docs/MARKET_ANALYSIS_DATA_REFACTOR.md`).
- Document which ratios you expose in the UI so future changes stay consistent.

**Gotchas**

- Version upgrades may change default behavior; pin version in `requirements.txt` during integration and test ratio outputs.

**Implemented:** 2025-02-18. FinanceToolkit is used in `market_data.get_fundamentals_ratios()` (fundamentals: profitability ratios, revenue TTM) and optionally for current P/E and PEG in `market_data.fetch_valuation_data()`. Adapter: `app/financetoolkit_adapter.py`. Feature flag: `USE_FINANCETOOLKIT` (default true). Fallback order: FinanceToolkit → OpenBB → FMP. `CompanyFundamentals.data_source` can be `financetoolkit`.

---

### 1.5 skfolio (Portfolio optimization)

| Field | Value |
|-------|--------|
| **Repository** | [skfolio/skfolio](https://github.com/skfolio/skfolio) |
| **License** | BSD-3-Clause |
| **Stars** | ~1.9k |
| **Language** | Python |
| **Install** | `pip install skfolio` |

**Description**  
Portfolio optimization library built on scikit-learn: mean-variance, risk parity, efficient frontier, CVaR, hierarchical clustering, and related estimators.

**Key features**

- Scikit-learn style API (fit/predict, pipelines)
- Multiple optimization objectives and constraints
- Asset allocation and risk decomposition
- Integrates with common pandas/NumPy workflows

**Where it fits in this project**

- **Dashboard or Portfolio & Taxes:** Optional “Optimize allocation” or “Suggested allocation” based on current positions (and optionally historical returns from `market_data` or cached prices). Could read from `Trades`/aggregated positions via `tax_engine.get_portfolio_summary` and run optimization for display only (no automatic trading).

**Integration notes**

- Input: aggregate positions (and optionally historical returns) from existing DB/session. Do not bypass tax_engine or duplicate trade logic.
- Present as an analytical view with clear disclaimer (e.g. “for illustration only”), not as trade advice.

---

### 1.6 Backtesting.py

| Field | Value |
|-------|--------|
| **Repository** | [kernc/backtesting.py](https://github.com/kernc/backtesting.py) |
| **License** | AGPL-3.0 |
| **Stars** | ~7.9k |
| **Language** | Python |
| **Install** | `pip install backtesting` |

**Description**  
Lightweight backtesting framework with a simple API. Produces performance metrics (e.g. Sharpe, Sortino, Calmar, max drawdown) and built-in visualization.

**Key features**

- Minimal setup: define a strategy class, run backtest on OHLCV
- Built-in metrics and HTML/plot output
- Good for teaching and quick strategy checks

**Where it fits in this project**

- **Market Analysis:** Alternative to VectorBT for a simpler “backtest this strategy” feature (e.g. one indicator or one rule). Use cached OHLCV from `market_data` to avoid extra API calls.

**Integration notes**

- AGPL-3.0: if the app is distributed, ensure compliance (e.g. open source or network-use exception). For personal/internal use, less of an issue.
- Same data discipline as VectorBT: feed from cache/DB, not live API.

---

### 1.7 Sharpefolio / Stock Portfolio Analyzer (Reference)

| Field | Value |
|-------|--------|
| **Sharpefolio** | [melvinmt/sharpefolio](https://github.com/melvinmt/sharpefolio) — Sharpe/Sortino-focused portfolio suggestions |
| **Stock Portfolio Analyzer** | [Erik-Kelemen/Stock-Portfolio-Analyzer](https://github.com/Erik-Kelemen/Stock-Portfolio-Analyzer) — Streamlit app: P&L, NAV, volatility, alpha, beta, Sharpe, CSV upload |

**Where it fits**

- **Reference only:** Ideas for portfolio metrics (Sharpe, Sortino) or Streamlit layout (tables, charts) on Dashboard/Portfolio. Not necessarily added as a dependency; useful for design and formula reference.

---

## 2. Data: gather, find, leverage

Repositories that help **unify data sources**, **fetch market/fundamental data**, **parse SEC filings (8-K, 13F)**, **IPO calendars**, or **news/sentiment**.

---

### 2.1 OpenBB (Unified financial data platform)

<!-- Implemented: 2025-02-18 openbb_adapter in market_data.fetch_ohlcv, get_company_profile, get_fundamentals_ratios, fetch_company_news; tax_engine.fetch_single_price; ipo_service.get_historical_price, get_current_price -->

| Field | Value |
|-------|--------|
| **Repository** | [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) |
| **License** | Check repo (permissive) |
| **Stars** | 60k+ |
| **Language** | Python |
| **Install** | `pip install openbb`; providers: `pip install "openbb[all]"` |

**Description**  
Open Data Platform (ODP): single Python API over many financial data providers. “Connect once, consume everywhere.” Supports REST API, Excel, and AI/MCP integrations.

**Key features**

- One interface for equity prices, fundamentals, and more
- **Public (default):** FRED, SEC, Polygon, Federal Reserve, ECB, IMF, OECD, etc.
- **Optional providers:** Alpha Vantage, yfinance, FMP, Intrinio, Tiingo, Tradier (install via `openbb[all]` or specific extras)
- Returns standardized DataFrames; can specify provider per call or use auto-fallback
- Reduces “source down” and rate-limit fragmentation by switching providers at one layer

**Where it fits in this project**

- **Broad refactor:** Replace or wrap `api_clients.py` and parts of `market_data.py` with OpenBB for OHLCV, fundamentals, and company data. Existing env vars (Polygon, Alpha Vantage, FMP, etc.) can often be mapped to OpenBB provider config.
- **Incremental:** Introduce OpenBB for one domain first (e.g. equity price history), keep existing code paths until migration is validated. Preserve cache/DB-first behavior and rate-limit awareness.

**Integration notes for implementers**

- Map existing API keys to OpenBB provider settings (see OpenBB docs). Keep `.env` and avoid committing keys.
- Preserve `market_data` cache keys and DB usage: e.g. call OpenBB, then write to `.market_cache/` and `valuation_history` / `company_fundamentals` as today.
- `docs/ARCHITECTURE.md` and root README list which modules use which APIs; update when switching to OpenBB.

**Gotchas**

- OpenBB’s API and provider set evolve; pin version and check release notes when upgrading.
- Some providers still have rate limits; OpenBB does not remove the need for caching and throttling in a multi-user or scripted context.

---

### 2.2 EdgarTools (SEC EDGAR: 8-K, 10-K, 10-Q, 13F)

| Field | Value |
|-------|--------|
| **Repository** | [dgunning/edgartools](https://github.com/dgunning/edgartools) |
| **License** | MIT |
| **Stars** | ~1.7k |
| **Language** | Python |
| **Install** | `pip install edgartools` |

**Description**  
Production-oriented Python library for SEC EDGAR. Parses 8-K (including **Item 1.01**), 10-K, 10-Q, 13F, Form 4, and others. Optimized for speed (XBRL, PyArrow) and returns structured objects and Pandas DataFrames.

**Key features**

- **8-K:** Access by item, e.g. `EightK(filing)['Item 1.01']` for material agreements (partnerships)
- **13F:** Institutional holdings as DataFrames
- 10–30x faster than some alternatives in benchmarks
- Type hints, 1000+ tests, MIT license

**Where it fits in this project**

- **Partnerships (8-K):** Replace or augment `edgar_service.py` for fetching and parsing 8-K filings and extracting **Item 1.01** partnership events. Current logic uses custom EDGAR fetching and parsing; EdgarTools can reduce custom code and improve robustness.
- **13F:** Optionally replace or augment `thirteenf_service.py` parsing (submissions, XML parsing) with EdgarTools’ 13F support. Keep `thirteenf_config.py` for CIK list and any app-specific filtering.

**Integration notes for implementers**

- Set identity: `from edgar import *; set_identity("Your Name <email@example.com>")` (SEC requires a User-Agent).
- Preserve existing cache layout under `.edgar_cache/` (and partnerships JSON) so the Partnerships page and any 13F UI keep working. EdgarTools can be the parser; your code can still decide what to cache and where.
- `partnerships_config.py`: continue to use it for watched tickers and counterparty highlighting; feed EdgarTools results into that pipeline.

**Gotchas**

- EdgarTools may return slightly different field names or structures than current `edgar_service`; map to existing data structures or update consumers (e.g. `partnerships_page()`) once.

---

### 2.3 sec-edgar (Batch filing download)

| Field | Value |
|-------|--------|
| **Repository** | [sec-edgar/sec-edgar](https://github.com/sec-edgar/sec-edgar) |
| **License** | Apache 2.0 / MIT (check repo) |
| **Stars** | ~1.3k |
| **Install** | `pip install secedgar` |

**Description**  
Focused on **downloading** SEC filings (periodic reports, 8-K, etc.) by CIK, ticker, or filing type. Batch download and date-based filtering. Does not replace full-featured parsing; use with a parser (e.g. EdgarTools).

**Key features**

- Simple API: `filings(cik_lookup="aapl", filing_type=FilingType.FILING_8K, ...).save(path)`
- Multiple companies and date ranges
- Good for bulk refresh of raw filings

**Where it fits in this project**

- **Partnerships / EDGAR:** Use to **bulk-download** 8-K (or other) filings for a list of tickers/CIKs; then parse with EdgarTools or existing `edgar_service` logic. Complements EdgarTools (which focuses on parsing and structure).

**Integration notes**

- Respect SEC rate limits; use existing `.edgar_cache/` and any throttling you have in `edgar_service`.
- User-Agent required (SEC policy); set in secedgar API if needed.

---

### 2.4 py-sec-edgar (Download and parse multiple form types)

| Field | Value |
|-------|--------|
| **Repository** | [ryansmccoy/py-sec-edgar](https://github.com/ryansmccoy/py-sec-edgar) |
| **Stars** | ~116 |
| **Description** | Downloads and parses 10-K, 10-Q, 8-K, 13-D, S-1; extracts structured and unstructured data. |

**Where it fits**

- **Alternative parser** if you want structured extraction without switching to EdgarTools. Less star count and possibly less 8-K item-level polish than EdgarTools; evaluate for Item 1.01 use case before committing.

---

### 2.5 finance_calendars (IPO and earnings calendars — NASDAQ)

| Field | Value |
|-------|--------|
| **Repository** | [s-kerin/finance_calendars](https://github.com/s-kerin/finance_calendars) |
| **Stars** | ~75 |
| **Install** | `pip install finance_calendars` (check PyPI/repo) |

**Description**  
Wrapper around **NASDAQ public API** for IPO and earnings calendars. No API key required for the NASDAQ data it uses.

**Key features**

- **IPO:** `get_priced_ipos_this_month()`, `get_filed_ipos_by_month()`, `get_upcoming_ipos_this_month()`, `get_withdrawn_ipos_this_month()`, and by-month variants
- **Earnings / dividends** calendar methods as well

**Where it fits in this project**

- **IPO Vintage Tracker:** Complement or replace Finnhub-based IPO calendar in `ipo_service.py`. Use for upcoming/priced/filed IPOs; keep existing vintage performance and price history logic (Finnhub or yfinance), and merge NASDAQ calendar with your `IPO_Registry` and exchange selection.

**Integration notes**

- Cache results (e.g. in `.ipo_cache/` or DB) to avoid repeated NASDAQ calls and align with existing IPO cache strategy.
- Compare date ranges and field names with Finnhub so the IPO calendar UI still shows a single coherent list.

---

### 2.6 fical (IPO/earnings iCal feeds)

| Field | Value |
|-------|--------|
| **Repository** | [niusann/fical](https://github.com/niusann/fical) |
| **Description** | Generates iCalendar (ICS) for NASDAQ IPOs and earnings; updated via GitHub Actions; subscribe in Google/Apple/Outlook. |

**Where it fits**

- **UX enhancement:** Offer “Add to calendar” or “Subscribe to IPO calendar” using fical’s public URLs or by generating similar ICS from your own IPO data (e.g. from `ipo_service` or finance_calendars). No need to depend on fical’s repo directly unless you want to reuse its pipeline.

---

### 2.7 Stock news sentiment (Reference)

| Field | Value |
|-------|--------|
| **Example** | [javierdejesusda/Stock-news-sentiment-analysis](https://github.com/javierdejesusda/Stock-news-sentiment-analysis) — Alpha Vantage news + VADER and FinBERT sentiment |

**Where it fits**

- **Market Analysis — Company news:** Add sentiment scores to `fetch_company_news` / company news block. Pipeline: fetch news (FMP/Finnhub/Alpha Vantage as today), run sentiment (VADER or FinBERT), store/display scores. Useful for future AI agents implementing “sentiment overlay” on news.

---

## 3. Display

Repositories that improve **charting** (TradingView/Bloomberg-style) or **Streamlit** dashboard patterns.

---

### 3.1 TradingView Lightweight Charts (JavaScript)

| Field | Value |
|-------|--------|
| **Repository** | [tradingview/lightweight-charts](https://github.com/tradingview/lightweight-charts) |
| **License** | Apache 2.0 |
| **Stars** | ~13.8k |
| **Language** | TypeScript/JavaScript |

**Description**  
Official TradingView library for financial charts: candlestick, line, area, bar; volume; real-time updates; small bundle (~35 KB).

**Where it fits**

- **Front-end basis** for the Market Analysis price chart. Typically used via a **Streamlit component** that embeds it (see below), not directly from Python.

---

### 3.2 streamlit-lightweight-charts (Streamlit component)

**Implemented:** Used in Market Analysis for the main price chart via `chart_utils.ohlcv_to_lightweight_charts_data`, `chart_utils.build_technical_chart_config`, and `renderLightweightCharts()` in `main.py`.

| Field | Value |
|-------|--------|
| **Repository** | [freyastreamlit/streamlit-lightweight-charts](https://github.com/freyastreamlit/streamlit-lightweight-charts) |
| **Install** | `pip install streamlit-lightweight-charts` |
| **PyPI** | [streamlit-lightweight-charts](https://pypi.org/project/streamlit-lightweight-charts/) |

**Description**  
Streamlit component that wraps TradingView Lightweight Charts. Renders candlestick, line, area, bar, baseline, histogram with configurable options.

**Key features**

- `renderLightweightCharts(charts: List[Dict], key: str)` — pass chart config from Python
- Overlays, styling, price scales, markers
- Fits into existing Streamlit layout

**Where it fits in this project**

- **Market Analysis — Price chart:** The primary technical price chart is TradingView-style candlesticks and volume. Data source: `market_data.fetch_ohlcv()` then `chart_utils` then component format. Plotly is still used for the TradingView signals chart and rescaling.

**Integration notes for implementers**

- Convert DataFrame from `fetch_ohlcv()` to the format expected by the component (usually list of dicts with time, open, high, low, close, volume). Keep timeframe and cache keys consistent with existing behavior.
- Optional: keep Plotly as fallback or for a second chart type; document in ARCHITECTURE which component is primary.

---

### 3.3 lightweight-charts-python / bn_lightweight-charts-python

| Field | Value |
|-------|--------|
| **Repositories** | [smalinin/bn_lightweight-charts-python](https://github.com/smalinin/bn_lightweight-charts-python), PyPI `lightweight-charts` |
| **Description** | Python wrapper for Lightweight Charts; supports Streamlit via `StreamlitChart` (e.g. `lightweight_charts.widgets.StreamlitChart`). |

**Where it fits**

- **Alternative** to `streamlit-lightweight-charts` if you prefer a different API (e.g. `chart.set(df); chart.load()`). Same data source: OHLCV from `market_data`.

---

### 3.4 Reference Streamlit apps (UI patterns)

| Repo | Use in this project |
|------|----------------------|
| [tranhlok/stock-dashboard](https://github.com/tranhlok/stock-dashboard) | Candlestick + MA + Bollinger; IEX; layout ideas for Market Analysis. |
| [paduel/streamlit_finance_chart](https://github.com/paduel/streamlit_finance_chart) | Simple Streamlit + yfinance + MA; structure and caching patterns. |
| [marketcalls/vectorbt-streamlit](https://github.com/marketcalls/vectorbt-streamlit) | If you add VectorBT backtests, reference for Streamlit + VectorBT visualization. |

Use as **reference only** (copy patterns or structure), not necessarily as installed dependencies.

---

## Quick reference and implementation order

**By function**

| Function | Repo | Primary use in project |
|----------|------|-------------------------|
| Analysis — indicators | pandas-ta | `market_data.py`: indicators and TradingView-style signals |
| Analysis — fundamentals | FinanceToolkit | `market_data.py`: valuation/fundamentals consistency |
| Analysis — backtest | VectorBT | Market Analysis: strategy backtest view |
| Data — unified API | OpenBB | Refactor `api_clients` / `market_data` data layer |
| Data — SEC 8-K / 13F | EdgarTools | `edgar_service.py`, optionally `thirteenf_service.py` |
| Data — IPO calendar | finance_calendars | `ipo_service.py`: NASDAQ IPO calendar |
| Display — charts | streamlit-lightweight-charts | Market Analysis: TradingView-style price chart |

**Suggested implementation order (for a future agent)**

1. **streamlit-lightweight-charts** — Improve chart UX with minimal change to data pipeline.
2. **pandas-ta** — Expand and standardize indicators/signals in `market_data.py`.
3. **EdgarTools** — Simplify 8-K (and optionally 13F) parsing; reduce custom EDGAR code.
4. **OpenBB** — Optionally unify market/fundamental data behind one API.
5. **FinanceToolkit** — Standardize valuation/fundamental ratios and documentation. *(Implemented: see 1.4.)*
6. **finance_calendars** — Add or backup IPO calendar from NASDAQ.
7. **VectorBT** — Add “backtest this strategy” or similar analysis view.

---

## Example: minimal first integration (pandas-ta)

This is a template for integrating one repo with minimal risk. A future agent can follow the same pattern for other repos.

**Goal:** Use pandas-ta for one additional indicator (e.g. Stochastic RSI) in Market Analysis while keeping existing cache and UI behavior.

1. **Add dependency:** In `requirements.txt`, add `pandas-ta>=0.3.14` (or current version from PyPI). Run `pip install -r requirements.txt`.
2. **Locate call site:** In `market_data.py`, `calculate_tradingview_signals()` (and helpers like `calculate_stochastic_rsi()`) build the signals DataFrame from OHLCV. The OHLCV DataFrame is already available there; it has columns such as `Open`, `High`, `Low`, `Close`, `Volume`.
3. **Use pandas-ta without breaking existing flow:** Import pandas_ta (e.g. `import pandas_ta as ta`). On the same DataFrame, you can call e.g. `df.ta.stochrsi()` to get Stochastic RSI columns. Merge those columns into the existing signals DataFrame that `calculate_tradingview_signals()` returns, so the rest of the pipeline (caching, UI) stays unchanged.
4. **Preserve cache shape:** The TV signals cache key and structure are used by `load_tv_signals_from_cache`, `save_tv_signals_to_cache`, and the Market Analysis page. If you add columns, ensure the UI and any code that reads the cache still handle the shape (e.g. new columns are optional for display).
5. **Validate:** Run the app, open Market Analysis, select a ticker, and confirm signals (including the new indicator) appear and that "Save TV" and reload still work. Check `.market_cache/{TICKER}_tv_signals_1W.json` to see the new fields persisted.

**Why this is a good first integration:** It touches only `market_data.py` and the existing signal pipeline; it does not change DB schema, env vars, or page routing. The same pattern (add lib → call at existing function → preserve cache/API) applies to EdgarTools (replace parsing in `edgar_service.py`) and other repos.

---

## Project file reference (for implementers)

- **App entry and pages:** `app/main.py` — `main()`, `dashboard_page()`, `portfolio_taxes_page()`, `market_analysis_page()`, `ipo_tracker_page()`, `partnerships_page()`, `thirteenf_page()`. Run from project root: `streamlit run app/main.py`.
- **Data and APIs:** `app/market_data.py` (OHLCV, valuation, signals, profile, news), `app/openbb_adapter.py`, `app/api_clients.py` (Polygon, Twelve Data, EODHD), `app/ipo_service.py`, `app/edgar_service.py`, `app/thirteenf_service.py`.
- **Config:** `partnerships_config.py`, `thirteenf_config.py`; `.env` for API keys.
- **Persistence:** `db.py`, `models.py`; caches: `.market_cache/`, `.ipo_cache/`, `.edgar_cache/`.
- **Docs:** `docs/ARCHITECTURE.md`, `docs/MARKET_ANALYSIS_DATA_REFACTOR.md`, root `README.md` (“For AI and refactors” section).

When you complete an integration, add a short “Implemented” note (e.g. `<!-- Implemented: YYYY-MM-DD repo-name in module.function -->`) so the next agent knows what is already in use.
