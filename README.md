# Gavin Financial Terminal

A personal financial intelligence platform modeled after Bloomberg/TradingView, built with Streamlit and PostgreSQL. The app has six main areas: **Dashboard** (portfolio + tax summary), **Portfolio & Taxes** (trades, lots, HIFO), **Market Analysis** (charts, valuation, TradingView-style signals), **IPO Vintage Tracker** (calendar + post-IPO performance), **Partnerships** (SEC 8-K partnership events), and **13F Institutional Holdings** (quarterly 13F filings, compare quarters, by-CUSIP, overlap).

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

1. Ensure PostgreSQL is installed and running
2. Create a database named `gavin_financial` (or update the name in your `.env` file)
3. Create a `.env` file in the project root with the following variables (or copy `.env.example` to `.env` and fill in your values). **Do not commit `.env` or paste API keys into the repo; use environment variables only.**

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gavin_financial
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 3. Initialize Database

The database tables will be automatically created on first run. Alternatively, from the project root:

```bash
python -c "from app.db import init_db; init_db()"
```

### 4. Run the Application

From the project root (the `Invest` folder):

```bash
streamlit run app/main.py
```

## Project Structure

| Path | Purpose |
|------|---------|
| `app/` | Application package (all app code) |
| `app/main.py` | Streamlit app: dashboard, portfolio/taxes, market analysis, IPO tracker, partnerships, 13F holdings |
| `app/models.py` | SQLAlchemy models: Trades, Watchlist, IPO_Registry, ValuationHistory, CompanyProfile, CompanyFundamentals |
| `app/db.py` | PostgreSQL connection and session management |
| `app/tax_engine.py` | HIFO tax lot tracking, gain calculations, CSV import |
| `app/market_data.py` | OHLCV cache, valuation (P/E, revenue), TradingView-style signals, company profile/fundamentals/news |
| `app/openbb_adapter.py` | OpenBB data layer (OHLCV, quote, profile, fundamentals, news) |
| `app/financetoolkit_adapter.py` | FinanceToolkit layer (fundamentals, optional current P/E and PEG) |
| `app/api_clients.py` | Price/OHLCV APIs: Polygon, Twelve Data, EODHD (with rate limits) |
| `app/ipo_service.py` | IPO calendar (Finnhub), vintage performance, price history |
| `app/edgar_service.py` | SEC EDGAR 8-K filings, partnership-event extraction |
| `app/thirteenf_service.py` | SEC 13F institutional holdings |
| `app/partnerships_config.py` | Config: watched tickers and counterparties for partnerships page |
| `app/thirteenf_config.py` | Config: 13F institutions to track |
| `app/plotly_chart_rescale.py` | Plotly chart y-axis rescaling helper |
| `app/chart_utils.py` | Lightweight-charts helpers: OHLCV→chart config, technical chart build |
| `docs/` | Documentation (architecture, open-source repos, refactor notes) |
| `requirements.txt` | Python dependencies |
| `VISION.md` | Project vision and feature roadmap |
| `docs/README.md` | Index of docs; when to read each |
| `docs/ARCHITECTURE.md` | Data flow, page→module map, refactor notes |
| `docs/MARKET_ANALYSIS_DATA_REFACTOR.md` | Refactor plan: auto-save, DB/cache-first for market analysis |

## Features

### Dashboard
- Portfolio value and metrics overview
- Unrealized gains/losses with real-time prices (via yfinance)
- Tax liability summary (Short-Term vs Long-Term gains)
- Portfolio allocation pie chart
- Gains by tax status bar chart
- Position details table
- Urgent alerts for lots nearing long-term status

### Portfolio & Taxes
- **Manual Trade Entry**: Add individual buy/sell transactions
- **CSV Import**: Bulk import trade history with column mapping
- **Tax Lot Viewer**: See individual tax lots with:
  - Holding period tracking
  - Long-Term (>365 days) vs Short-Term classification
  - Days until long-term status
  - Cost basis and unrealized gains per lot
  - Alerts for lots within 30 days of becoming long-term

### Tax Engine (HIFO Logic)
The tax engine uses **Highest-In-First-Out (HIFO)** methodology:
- When simulating a sale, highest cost basis shares are "sold" first
- This minimizes taxable gains for tax optimization
- Tracks individual tax lots from each purchase

### Market Analysis
- **Ticker search** with OHLCV cache (file + optional DB), valuation history (PostgreSQL-first)
- **Price chart**: Primary chart is TradingView-style via `streamlit_lightweight_charts` and `chart_utils` (candlestick + volume); Plotly is used for the TradingView signals chart and rescaling.
- **Valuation chart**: P/E and revenue over time (yfinance, Alpha Vantage, Finnhub, FMP)
- **TradingView-style signals**: Stochastic RSI, momentum, supply/demand zones; cached per ticker/timeframe
- **Competitors**: Peer comparison (same industry + similar market cap via FMP screener); optional sort by description similarity; user overrides (add/remove peers) stored in DB; table with ticker, name, sector, industry, market cap, P/E, revenue TTM. Optional env: `PEERS_COUNTRY`, `PEERS_EXCHANGE`.
- Company profile, fundamentals, and news (FMP, Finnhub)

### IPO Vintage Tracker
- **IPO calendar** for upcoming IPOs (Finnhub, cached)
- **Vintage performance**: 1/2/3-year post-IPO returns and price history
- **Anniversary alerts** for IPO vintages
- Manual IPO registry and exchange selection

### Partnerships
- **SEC 8-K partnership events**: Item 1.01 filings, extracted counterparties
- Configurable watchlist of tickers and highlighted counterparties (`partnerships_config.py`)
- Cached EDGAR data to respect rate limits

### 13F Institutional Holdings
- **Quarterly 13F filings** for configurable institutions (via `thirteenf_config.py`)
- Holdings view (13f.info-style), compare two quarters, lookup by CUSIP, overlap across institutions
- Data from SEC EDGAR; cache under `.edgar_cache/13f/`

---

## For AI and refactors

This section gives future AIs and refactors enough context to navigate the codebase and change it safely.

### Entry points and routing

- **App entry**: From project root, `streamlit run app/main.py` → `app/main.py` → `main()` → sidebar `st.sidebar.radio` chooses page.
- **Pages** (all in `app/main.py`): `dashboard_page()`, `portfolio_taxes_page()`, `market_analysis_page()`, `ipo_tracker_page()`, `partnerships_page()`, `thirteenf_page()`. Sidebar order: Dashboard, Portfolio & Taxes, Market Analysis, IPO Vintage Tracker, Partnerships, 13F Holdings.
- **Module usage**: Dashboard/Portfolio use `db`, `models`, `tax_engine`. Market Analysis uses `market_data` (which can use `financetoolkit_adapter`, `openbb_adapter`, and `api_clients` for OHLCV and fundamentals). IPO uses `ipo_service`. Partnerships uses `edgar_service` and `partnerships_config`. 13F uses `thirteenf_service` and `thirteenf_config`. All app code lives under `app/`; caches (`.market_cache/`, `.ipo_cache/`, `.edgar_cache/`) and `.env` stay at project root.

### Where key logic lives

| Concern | Primary location | Notes |
|--------|-------------------|--------|
| Portfolio aggregation, tax summary | `tax_engine.py` (`TaxEngine`, `get_portfolio_summary`) | Reads `Trades` from DB via session passed in |
| Trade CRUD, lots, CSV import | `main.py` (portfolio_taxes_page) + `tax_engine.py` (HIFO, `import_trades_from_csv`) | Trades stored in `models.Trades` |
| OHLCV fetch, cache, indicators | `market_data.py` (`fetch_ohlcv`, …), `openbb_adapter.py` | OpenBB first, then yfinance, then `api_clients`; file cache `.market_cache/` |
| Valuation (P/E, revenue) | `market_data.py` (`fetch_valuation_data`, …) | DB table `valuation_history`; also Alpha Vantage / Finnhub / FMP |
| Company profile, fundamentals, news | `market_data.py` (`get_company_profile`, `get_fundamentals_ratios`, `fetch_company_news`), `financetoolkit_adapter.py`, `openbb_adapter.py` | Fundamentals: FinanceToolkit first when enabled, then OpenBB, then FMP; profile/news: OpenBB then FMP/Finnhub; profile/fundamentals use DB cache |
| Single-ticker price (tax/portfolio) | `tax_engine.py` (`fetch_single_price`, `get_cached_price`), `openbb_adapter.py` | OpenBB quote first, then yfinance + Alpha Vantage + `api_clients`; in-memory cache 15 min |
| IPO calendar and vintage | `ipo_service.py` (`fetch_ipo_calendar`, `get_vintage_performance`, `get_ipo_price_history`), `openbb_adapter.py` | OpenBB for current/historical price when available; Finnhub for calendar; file cache `.ipo_cache/` |
| 8-K partnership events | `edgar_service.py` (`get_partnership_events`, `refresh_edgar_data`) | SEC EDGAR; file cache `.edgar_cache/` |
| 13F institutional holdings | `thirteenf_service.py` (`get_13f_filings_for_institution`, `get_13f_holdings_by_quarter`, `get_13f_compare`, etc.) | SEC EDGAR; file cache `.edgar_cache/13f/` |

### Environment variables

Load from `.env` in project root. Used by:

| Variable | Used in | Purpose |
|----------|---------|---------|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | `db.py` | PostgreSQL connection |
| `ALPHA_VANTAGE_API_KEY` or `ALPHA_API_KEY` | `market_data.py`, `tax_engine.py` | Valuation/earnings; price fallback |
| `FMP_API_KEY` | `market_data.py`, `financetoolkit_adapter.py` | Company profile, fundamentals; FinanceToolkit uses it for ratio data when enabled |
| `FINNHUB_API_KEY` | `market_data.py`, `ipo_service.py` | Earnings, news; IPO calendar |
| `MASSIVE_API_KEY` | `market_data.py` (via `api_clients`), `openbb_adapter.py` | Polygon price/OHLCV (OpenBB uses `POLYGON_API_KEY`; adapter sets it from `MASSIVE_API_KEY` if unset) |
| `POLYGON_API_KEY` | `openbb_adapter.py` | OpenBB Polygon provider (optional; if unset, adapter copies from `MASSIVE_API_KEY`) |
| `TWELVEDATA_API_KEY` | `api_clients.py` | Price/OHLCV fallback |
| `EODHD_API_KEY` | `api_clients.py` | Price/OHLCV fallback (e.g. 20/day) |
| `USE_OPENBB` | `openbb_adapter.py` | Set to `false` to disable OpenBB and use only fallbacks (default: enabled) |
| `USE_FINANCETOOLKIT` | `financetoolkit_adapter.py` | Set to `false` to disable FinanceToolkit for fundamentals and current P/E/PEG; app falls back to OpenBB then FMP (default: enabled) |
| `PEERS_COUNTRY` | `market_data.py` | Optional. Restrict competitor screener to country (e.g. `US`). |
| `PEERS_EXCHANGE` | `market_data.py` | Optional. Restrict competitor screener to exchange (e.g. `NASDAQ,NYSE`). |

**OpenBB:** The app uses OpenBB as the primary data layer when available (OHLCV, quote, profile, fundamentals, news, IPO prices). Ensure `.env` is loaded before the first OpenBB use (the adapter loads it when imported). After installing or removing OpenBB provider extensions, run `openbb-build` to refresh the Python interface.

### Persistence and caches

- **PostgreSQL** (`db.py` + `models.py`): `trades`, `watchlist`, `ipo_registry`, `valuation_history`, `company_profile`, `company_fundamentals`, `peer_overrides`.
- **File caches** (project root): `.market_cache/` (OHLCV, ticker info, TV signals, Alpha Vantage responses, peers candidates), `.ipo_cache/` (IPO calendar JSON), `.edgar_cache/` (EDGAR submissions, 8-K, partnership_events.json).
- **In-memory**: `tax_engine` price cache (15 min TTL).

### Conventions and gotchas

- **DB session**: Pages get a session via `get_db_session()`; pass it into `TaxEngine` and any code that reads/writes `Trades` or other models. Don’t create a new engine in feature modules.
- **Market data rate limits**: Alpha Vantage (5/min), EODHD (20/day), etc. See `market_data.py` and `api_clients.py` for throttling. Prefer cache/DB before external calls.
- **Valuation**: Already DB-first in `fetch_valuation_data()`; TradingView and valuation have “Save” buttons—see `docs/MARKET_ANALYSIS_DATA_REFACTOR.md` for planned auto-save and freshness behavior.

### More detail

- **`docs/README.md`** — Index of all docs and when to use them.
- **`docs/ARCHITECTURE.md`** — Data flow, page→module map, and refactor notes.
- **`docs/MARKET_ANALYSIS_DATA_REFACTOR.md`** — Refactor plan for market analysis data (auto-save, DB/cache-first).

