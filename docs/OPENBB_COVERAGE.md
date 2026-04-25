# OpenBB coverage map

Single inventory for OpenBB-backed fetches. **Provider order** is defined in `app/openbb_provider_registry.py` (`OPENBB_PROVIDER_CHAINS`). **Update this file in the same PR** when you change providers, TTL, or consumers.

CI runs `python scripts/verify_openbb_coverage_doc.py` so every **multi-provider** chain in the registry appears below.

| Domain | Adapter function | Consumer(s) | Providers (order) | Cache layer | TTL | Notes |
|--------|------------------|-------------|-------------------|-------------|-----|--------|
| Equity OHLCV | `fetch_ohlcv_openbb` | `market_data.fetch_ohlcv` | polygon → yfinance → fmp | L2 file `.market_cache/{TICKER}_ohlcv.json` | 4h (see `market_data`) | Requires ≥50 rows after normalize; `run_provider_chain` |
| Equity quote | `fetch_quote_openbb` | `tax_engine.fetch_single_price`, `ipo_service.get_current_price` | yfinance → fmp → polygon | L1 in-memory quotes (`tax_engine`, 15m) | 15m | Kernel |
| Equity profile | `fetch_profile_openbb` | `market_data.get_company_profile` | fmp → yfinance → intrinio | L2 DB `CompanyProfile` | DB-backed | Kernel |
| Equity fundamentals | `fetch_fundamentals_openbb` | `market_data.get_fundamentals_ratios` | ratios: **fmp → intrinio**; income: **fmp → intrinio → polygon → yfinance** | L2 DB `CompanyFundamentals` | DB-backed | Two `run_provider_chain` calls |
| Company news | `fetch_news_openbb` | `market_data.fetch_company_news`, `thirteenf_agent` | fmp → polygon → yfinance → tiingo → intrinio | None in adapter; callers may cache | — | 14d lookback; Benzinga omitted (often key-gated) |
| Equity historical (point) | `fetch_historical_price_openbb` | `ipo_service.get_historical_price` | polygon → yfinance → fmp | Via IPO flows / cache | — | Single-day window |
| Macro series | `fetch_macro_data_openbb` | `macro_data.fetch_macro_indicator` | OpenBB **fred** → pandas_datareader FRED; keys include `nfci`, `t10yie`, `dfii10`, `init_claims` (and legacy GDP/CPI/…); `sahm` derived from `unemployment` in `macro_data` | L2 file `.macro_cache/{metric}.json` | 24h | `economy.fred_series` when `USE_OPENBB` + `FRED_API_KEY` |

## Environment

| Variable | Used by |
|----------|---------|
| `USE_OPENBB` | `openbb_fetch` — disables all `obb` usage when false |
| `OPENBB_REQUEST_TIMEOUT_SEC` | `openbb_fetch.run_provider_chain` (default 30) |
| `POLYGON_API_KEY` / `MASSIVE_API_KEY` | Polygon provider; `MASSIVE` copied to `POLYGON` in `openbb_fetch` |
| `FRED_API_KEY` | **Dashboard** rates via `macro_context`; macro indicators via OpenBB `fred_series` when OpenBB is enabled, else pandas_datareader |

## Log format

Logger: `gft.openbb`. Structured info lines include `dataset_id`, `symbol`, `provider`, `elapsed_ms`, `outcome` (`ok`, `empty`, `timeout`, `fallback`, `skip_disabled`).
