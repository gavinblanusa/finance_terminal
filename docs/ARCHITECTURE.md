# Architecture and data flow

For AIs and refactors: this doc explains how data moves through the app and how pages map to modules. Use the root README for entry points and env vars; use this for “where does this data come from?” and “what breaks if I change that?”.

---

**Layout:** App code in `app/`. Run: `streamlit run app/main.py` from project root. Caches and `.env` at root.

## Page → module map

| Page | app/main.py functions | Modules used | Persistence |
|------|-------------------|--------------|-------------|
| **Dashboard** | `dashboard_page()` | `db`, `models` (Trades), `tax_engine` (TaxEngine, portfolio summary, prices) | PostgreSQL (trades) |
| **Portfolio & Taxes** | `portfolio_taxes_page()` | `db`, `models` (Trades), `tax_engine` (CRUD, HIFO, CSV import, prices) | PostgreSQL (trades) |
| **Market Analysis** | `market_analysis_page()` | `market_data` (OHLCV, valuation, signals, profile, fundamentals, news); `market_data` may call `api_clients` | `.market_cache/` (file); `valuation_history` (DB) |
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

### Market Analysis

1. **Ticker selection** (and optional watchlist) is UI state; no persistence of “current ticker” across runs.
2. **Price chart**: The main technical price chart uses **streamlit_lightweight_charts** + `chart_utils` (OHLCV→candlestick/volume config). OHLCV: `market_data.fetch_ohlcv(ticker)` → file cache `.market_cache/{TICKER}_ohlcv.json` (4h TTL) → on miss: **OpenBB** (`openbb_adapter.fetch_ohlcv_openbb`), then yfinance, then backup via `api_clients` (Polygon, Twelve Data, EODHD).
3. **Valuation** (P/E, revenue chart): `market_data.get_valuation_chart_data()` → internally `fetch_valuation_data()` which is **DB-first**: `load_valuation_from_db()`; if DB has enough quarters and recent enough, use it and only fetch current P/E from yfinance. Else fetch from APIs (Alpha Vantage, Finnhub, FMP, yfinance) then **optional** `save_valuation_to_db()` (currently on “Save” button).
4. **TradingView-style signals**: Computed from OHLCV in `market_data.calculate_tradingview_signals()`. Cached in `.market_cache/{TICKER}_tv_signals_{timeframe}.json`. Currently saved only when user clicks “Save TV”.
5. **Company profile / fundamentals / news**: `market_data.get_company_profile`, `get_fundamentals_ratios`, `fetch_company_news` — **Fundamentals:** when `USE_FINANCETOOLKIT` is enabled, **FinanceToolkit first** (`financetoolkit_adapter.fetch_fundamentals_financetoolkit`), then OpenBB, then FMP. Otherwise OpenBB first, then FMP. Profile and news: OpenBB first, then FMP/Finnhub. Profile/fundamentals use DB cache (`CompanyProfile`, `CompanyFundamentals`). **Valuation:** current P/E and PEG can optionally come from FinanceToolkit in `fetch_valuation_data()` when enabled; historical P/E and revenue unchanged.

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

- **DB session**: Created once per request/flow in `main.py` (e.g. `get_db_session()`). Passed into `TaxEngine` and any code that touches `Trades`, `Watchlist`, `IPO_Registry`, or cached valuation/profile/fundamentals. Do not open multiple sessions for the same logical operation.
- **Rate limits**: Alpha Vantage (5/min), EODHD (20/day), etc. Caches and DB-first paths are the main mitigation; see `market_data.py` and `api_clients.py` for throttling.
- **Caches on disk**: All under project root: `.market_cache/`, `.ipo_cache/`, `.edgar_cache/`. Safe to clear for a full refresh; app will refill on next use.

---

## Refactor and tech-debt notes

- **Market Analysis data**: Auto-save and “From DB”/“Cached” UX are described in **MARKET_ANALYSIS_DATA_REFACTOR.md**. Check that doc’s checklist against current code before changing valuation or TV signal persistence.
- **OHLCV**: Currently file-only. An optional PostgreSQL table for OHLCV is described in the refactor doc; not implemented as of this writing.
- **VISION.md** (root): High-level product vision and feature list; not an architecture doc.
