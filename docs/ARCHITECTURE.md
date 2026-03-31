# Architecture and data flow

For AIs and refactors: this doc explains how data moves through the app and how pages map to modules. Use the root README for entry points and env vars; use this for “where does this data come from?” and “what breaks if I change that?”.

---

**Layout:** App code in `app/`. Run: `streamlit run app/main.py` from project root. Caches and `.env` at root. Optional HTTP: `PYTHONPATH=app uvicorn terminal_api:app` (see **`docs/DATA_LAYER_REFERENCE.md`** HTTP section).

## Page → module map

| Page | app/main.py functions | Modules used | Persistence |
|------|-------------------|--------------|-------------|
| **Dashboard** | `dashboard_page()` | `db`, `models` (Trades, Watchlist), `tax_engine`, `macro_context`, `fi_context`, `portfolio_insights`, `factor_exposure`, `tca_estimate`, `relevant_news`, `market_data` (`get_company_profile`, `fetch_ohlcv`, `fetch_company_news`); `data_schemas` for JSON export + TCA session snapshot | PostgreSQL (trades, watchlist); Streamlit `@st.cache_data` (15m macro/insights/factors/TCA/FI strip, 10m news); `.market_cache/` (OHLCV file cache + `ff5_factors_daily.csv`); `st.session_state["gft_export_tca"]` for optional TCA in export |
| **Portfolio & Taxes** | `portfolio_taxes_page()` | `db`, `models` (Trades), `tax_engine` (CRUD, HIFO, CSV import, prices) | PostgreSQL (trades) |
| **Market Analysis** | `market_analysis_page()` | `market_data` (OHLCV, valuation, signals, profile, fundamentals, news, competitors); `options_iv_term` (ATM IV term via yfinance); `market_data` may call `api_clients` | `.market_cache/` (file); `valuation_history`, `peer_overrides` (DB); Streamlit cache for IV term (600s) |
| **IPO Vintage Tracker** | `ipo_tracker_page()` | `ipo_service`, `db`, `models` (IPO_Registry) | `.ipo_cache/` (file); PostgreSQL (ipo_registry) |
| **Partnerships** | `partnerships_page()` | `edgar_service`, `partnerships_config` | `.edgar_cache/` (file) |
| **13F Institutional Holdings** | `thirteenf_page()` | `thirteenf_service`, `thirteenf_config` | `.edgar_cache/13f/` (file) |

---

## Data flow by feature

### Portfolio and taxes

1. **Trades** are stored only in PostgreSQL (`Trades`). All trade CRUD in the Portfolio page goes through the same DB session (from `get_db_session()`).
2. **TaxEngine** receives that session and builds positions/lots in memory from `Trades`; it does not persist lots—only the underlying trades are stored.
3. **Prices** for unrealized gains: `tax_engine.fetch_single_price` / `fetch_prices_batch` — **OpenBB quote first**, then yfinance, Alpha Vantage, `api_clients`. In-memory cache 15 min.
4. **CSV import**: `tax_engine.import_trades_from_csv` + column mapping in UI → inserts into `Trades` via the same session.

Refactor note: Changing how lots are computed (e.g. LIFO) means changing `TaxEngine` in `tax_engine.py` only; trade storage stays the same.

### Dashboard (macro, PORT-lite, ranked news)

1. **Macro snapshot** (`macro_context.build_macro_context`): Per-symbol **yfinance** daily history (~15d) for indices, VIX, FX, DXY, gold, crude; last vs prior close → **Change %**. Optional **FRED** `series/observations` (latest point) for `DGS10`, `DGS2`, `DGS3MO`, `EFFR`, `T10Y2Y` when `FRED_API_KEY` is set. Cached in `main.py` as `_cached_macro_context` (TTL 900s). **Refresh Prices** clears this cache.
1b. **FI proxy strip** (`fi_context.build_fi_context_strip`): **^TNX**, **HYG**, **LQD**, **TLT**, **IEF** via yfinance (~15d) for last / change % — credit & duration **proxies** (not TRACE). Cached `_cached_fi_context_strip` (900s); cleared with Refresh.
2. **Portfolio risk snapshot** (`portfolio_insights.build_portfolio_insights`): Input is cached portfolio positions from `get_portfolio_data`. **Sector weights**: `market_data.get_company_profile` per ticker (DB-first `CompanyProfile`). **Concentration**: top-1 / top-5 % of value, Herfindahl on weights. **Beta**: `fetch_ohlcv(ticker, 2)` and `SPY`, simple returns, aligned dates, min ~120 overlapping days; per-ticker β = Cov(rᵢ,rₘ)/Var(rₘ); portfolio β value-weighted over tickers with valid β (weights renormalized). Cached `_cached_portfolio_insights(positions_key)` (900s); cleared with Refresh.
3. **Headlines for your book** (`relevant_news.build_relevant_news`): Universe = unique portfolio tickers + `Watchlist` (cap 25). Fetches `fetch_company_news` per symbol (OpenBB → Finnhub). Scores headlines (portfolio mention, watchlist, keyword list), dedupes by source+URL, sorts. Cached `_cached_relevant_news(port_tuple, watch_tuple)` (600s); cleared with Refresh.
3b. **JSON snapshot** (`data_schemas.build_dashboard_export_payload`): Dashboard download bundles macro + `PortfolioInsights` + `FactorExposureResult` when the book is non-empty; merges **`tca_estimate`** dict from `st.session_state["gft_export_tca"]` after the user runs **Execution · TCA** (cleared on Refresh).
4. **Factor exposure** (`factor_exposure.build_factor_exposure`): Ken French **daily 5-factor (2×3)** ZIP downloaded to `.market_cache/ff5_factors_daily.csv` (24h file TTL). Per-ticker OLS of excess return on `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`; portfolio loadings value-weighted. Uses `fetch_ohlcv` (same stack as β). Cached `_cached_factor_exposure(positions_key, ff_cache_mtime)` (900s); **Refresh Prices** clears it. See **`docs/DATA_LAYER_REFERENCE.md`** and Pydantic mirrors in `data_schemas.py`.
5. **TCA estimate** (`tca_estimate.estimate_trade_impact`): Illustrative pre-trade impact from ADV (20d mean volume), participation, realized vol; square-root heuristic (educational only). Cached `_cached_tca_estimate(ticker, shares, side)` (900s); cleared with Refresh.
6. **Dashboard UI helpers** (`main.py`): `_gft_dash_callout` renders macro/risk/news/load failures as styled panels (error / warning / info). `_gft_metrics_container()` wraps metric rows in `st.container(border=True)` when Streamlit ≥ 1.29 (see `requirements.txt`). Empty macro DF and “no positions” paths use the same dashed **`.gft-empty-state`** treatment as headlines. Research/execution subsections use **`gft-dash-section-stack-research`** / **`gft-dash-section-stack-exec`** for accent borders.

### Market Analysis

1. **Ticker selection** (and optional watchlist) is UI state; no persistence of “current ticker” across runs.
2. **Price chart**: The main technical price chart uses **streamlit_lightweight_charts** + `chart_utils` (OHLCV→candlestick/volume config). OHLCV: `market_data.fetch_ohlcv(ticker)` → file cache `.market_cache/{TICKER}_ohlcv.json` (4h TTL) → on miss: **OpenBB** (`openbb_adapter.fetch_ohlcv_openbb`), then yfinance, then backup via `api_clients` (Polygon, Twelve Data, EODHD).
3. **Valuation** (P/E, revenue chart): `market_data.get_valuation_chart_data()` → internally `fetch_valuation_data()` which is **DB-first**: `load_valuation_from_db()`; if DB has enough quarters and recent enough, use it and only fetch current P/E from yfinance. Else fetch from APIs (Alpha Vantage, Finnhub, FMP, yfinance) then **optional** `save_valuation_to_db()` (currently on “Save” button).
4. **TradingView-style signals**: Computed from OHLCV in `market_data.calculate_tradingview_signals()`. Cached in `.market_cache/{TICKER}_tv_signals_{timeframe}.json`. Currently saved only when user clicks “Save TV”.
5. **Company profile / fundamentals / news**: `market_data.get_company_profile`, `get_fundamentals_ratios`, `fetch_company_news` — **Fundamentals:** when `USE_FINANCETOOLKIT` is enabled, **FinanceToolkit first** (`financetoolkit_adapter.fetch_fundamentals_financetoolkit`), then OpenBB, then FMP. Otherwise OpenBB first, then FMP. Profile and news: OpenBB first, then FMP/Finnhub. Profile/fundamentals use DB cache (`CompanyProfile`, `CompanyFundamentals`). **Valuation:** current P/E and PEG can optionally come from FinanceToolkit in `fetch_valuation_data()` when enabled; historical P/E and revenue unchanged.
6. **Options ATM IV term** (`options_iv_term.build_iv_term_structure`): Listed expiries via yfinance; ATM strike vs summary spot (or inferred); IV from call/put average when both exist. Cached `_cached_iv_term_structure(ticker, spot_key)` (600s). **Clear Cache** on Market Analysis clears this Streamlit cache.
7. **Black–Scholes panel** (`options_black_scholes.black_scholes_european`): European call/put theory prices in an expander; risk-free default from `_cached_tnx_last_percent()` (^TNX). Optional alignment to IV table presets.
8. **Competitors**: `market_data.get_competitors(ticker, sort_by, max_peers)` → FMP company screener (industry + market cap, with fallback to sector / sector+wide cap), optional TF-IDF description similarity ranking, user overrides from `peer_overrides` table. Screener result cached in `.market_cache/peers_candidates_{TICKER}.json` (6h). `clear_peers_cache(ticker)` invalidates file cache.

### OpenBB adapter

- **`openbb_adapter.py`**: Thin wrapper around OpenBB (ODP). Used by `market_data`, `tax_engine`, and `ipo_service`. Each function returns `None` on failure or when OpenBB is not installed so callers fall back to existing providers. Pattern: **OpenBB first, then fallback**. Env: `.env` loaded on adapter import; `POLYGON_API_KEY` set from `MASSIVE_API_KEY` if unset; `USE_OPENBB=false` disables OpenBB.
- **`financetoolkit_adapter.py`**: FinanceToolkit wrapper for fundamentals (profitability ratios, revenue TTM) and optional current P/E and PEG. Used by `market_data.get_fundamentals_ratios()` and `fetch_valuation_data()`. Returns `None` on failure so callers fall back to OpenBB/FMP. Env: `USE_FINANCETOOLKIT=false` disables; `FMP_API_KEY` used by FinanceToolkit when set.

Refactor note: See **MARKET_ANALYSIS_DATA_REFACTOR.md** for planned auto-save and tiered freshness (e.g. auto-save valuation and TV signals when data is freshly fetched).

### IPO Vintage Tracker

1. **Calendar**: `ipo_service.fetch_ipo_calendar()` → `.ipo_cache/ipo_calendar_30d.json` (6h) → on miss: Finnhub.
2. **Vintage performance**: `ipo_service.get_vintage_performance(ticker, ipo_date)` uses **OpenBB** for current/historical price when available, then yfinance (or `api_clients` for historical close if needed); no DB for this.
3. **Registry**: Manual IPO list and “following” state in PostgreSQL (`IPO_Registry`). `main.py` ipo_tracker_page reads/writes this via `db` + `models`.

### Partnerships

1. **Event list**: `edgar_service.get_partnership_events()` → reads `.edgar_cache/partnership_events.json` if valid; else `refresh_edgar_data()` which uses SEC EDGAR (ticker→CIK, submissions, 8-K docs), filters Item 1.01, extracts counterparties, then writes back to `.edgar_cache/`.
2. **Config**: `partnerships_config.py` defines watched tickers and “interest” counterparty names; used when rendering and highlighting events.

### 13F Institutional Holdings

1. **Filings and holdings**: `thirteenf_service.get_13f_filings_for_institution(cik)`, `get_13f_holdings_by_quarter()`, `get_13f_compare()`, `get_holders_by_cusip()`, `get_overlap_holdings()` — all read from SEC EDGAR; submissions and holdings cached under `.edgar_cache/13f/` (file only; no DB).
2. **Config**: `thirteenf_config.py` defines institutions (name, CIK) to track; used by the 13F page for dropdowns and queries.

---

## Cross-cutting points

- **Data layer contracts**: Field-level documentation and failure modes for macro, PORT-lite, factors, TCA, options, FI strip, and **HTTP routes** live in **`docs/DATA_LAYER_REFERENCE.md`**. Serializable shapes are defined in **`app/data_schemas.py`**. **`app/terminal_api.py`** exposes read-only FastAPI endpoints; portfolio shape is built by **`app/portfolio_snapshot.py`** (also used by Streamlit `get_portfolio_data`).
- **DB session**: Created once per request/flow in `main.py` (e.g. `get_db_session()`). Passed into `TaxEngine` and any code that touches `Trades`, `Watchlist`, `IPO_Registry`, or cached valuation/profile/fundamentals. Do not open multiple sessions for the same logical operation.
- **Rate limits**: Alpha Vantage (5/min), EODHD (20/day), etc. Caches and DB-first paths are the main mitigation; see `market_data.py` and `api_clients.py` for throttling.
- **Caches on disk**: All under project root: `.market_cache/`, `.ipo_cache/`, `.edgar_cache/`. Safe to clear for a full refresh; app will refill on next use.

---

## Refactor and tech-debt notes

- **Market Analysis data**: Auto-save and “From DB”/“Cached” UX are described in **MARKET_ANALYSIS_DATA_REFACTOR.md**. Check that doc’s checklist against current code before changing valuation or TV signal persistence.
- **OHLCV**: Currently file-only. An optional PostgreSQL table for OHLCV is described in the refactor doc; not implemented as of this writing.
- **VISION.md** (root): High-level product vision and feature list; not an architecture doc.
