# Market Analysis Data Refactoring Recommendations

**Context for AIs/refactors:** This is a refactor plan left by a prior session. Before implementing, check the codebase for the current state: some checklist items may already be done (e.g. `from_database` pass-through). Use **ARCHITECTURE.md** for where valuation/TV/OHLCV logic lives.

**Status:** Valuation and TradingView auto-save are implemented; `from_database` pass-through is in place.

---

This document outlines how to refactor market analysis data flow so that:
1. **DB/cache is checked first** — valid, up-to-date data is used when possible to avoid rate limits.
2. **Tiered freshness** — price/RSI/TradingView stay fresh; revenue growth can lag.
3. **Minimal frontend effort** — no reliance on remembering to click "Save" for each of the three sections.

---

## Current State Summary

| Data Type | Storage | Freshness | User Action |
|-----------|---------|-----------|-------------|
| **OHLCV** (price, RSI, technical chart) | File cache (`.market_cache/{TICKER}_ohlcv.json`) | 4 hours | None (auto-cached on fetch) |
| **Valuation** (P/E + revenue chart) | PostgreSQL `valuation_history` | DB checked first (140-day staleness for revenue) | Must click "💾 Save" to persist |
| **TradingView signals** | File cache (`{TICKER}_tv_signals_{timeframe}.json`) | 24h (1W) / 4h (1D, 4H) | Must click "💾 Save TV" to persist |

**Valuation** already uses a DB-first path in `fetch_valuation_data()`: if `load_valuation_from_db()` returns enough quarters and the most recent quarter is within 140 days, that data is used and only yfinance is called for current P/E and PEG. So the main gaps are:
- **Auto-save**: valuation and TradingView are only saved when the user clicks Save.
- **OHLCV** is file-only (no PostgreSQL); freshness is uniform (4h) with no “last bar date” check.
- **from_database** is not passed through to the valuation chart response, so the UI never shows “From DB”.

---

## Recommendations

### 1. Tiered Freshness (Keep / Clarify)

- **Must be fresh (price, RSI, TradingView chart)**  
  - Keep OHLCV cache TTL short (e.g. 4 hours) or add a **“last bar date”** rule: use cache only if the latest row is today or yesterday (market hours–aware if needed). That way one API call per ticker per day is enough for the technical view.
- **Can lag (revenue growth)**  
  - Keep current behavior: DB is used if the most recent quarter is within ~140 days. No need to refresh revenue more often than quarterly.

No structural change required; optionally add “last bar date” to OHLCV cache validation to avoid unnecessary refreshes when the market hasn’t produced new data.

### 2. DB-First, Single Check per Ticker

- **Valuation**: Already DB-first in `fetch_valuation_data()`. Keep it.
- **OHLCV**: Continue file-cache-first in `fetch_ohlcv()`. Optionally add a PostgreSQL table (e.g. `ohlcv_cache`: ticker, last_date, payload) later for cross-device persistence; for now file cache is fine.
- **TradingView**: Derived from OHLCV. If OHLCV is cached/valid, TV signals can be recomputed and then auto-saved (see below).

Flow: one ticker search → one DB check (valuation) + one cache check (OHLCV) → only call APIs when cache/DB miss or stale.

### 3. Auto-Save to Remove “Forgot to Save”

- **Valuation**  
  When `fetch_valuation_data()` returns data **from API** (not from DB), **automatically** call `save_valuation_to_db()` after building the chart data (e.g. at the end of `get_valuation_chart_data()` or right after a successful API-based fetch in the page). That way every time you pay the API cost, the result is persisted for next time. Keep the “💾 Save” button as an optional “force refresh and save” or for when DB write fails.
- **TradingView**  
  After computing TradingView signals in the market analysis page, **automatically** call `save_tv_signals_to_cache()`. No need to click “💾 Save TV”; the cache is filled on first view. Keep the button for “force refresh and save” if you add that later.

Result: no need to remember to save any of the three sections; persistence happens when data is freshly fetched or computed.

### 4. Optional: OHLCV in PostgreSQL

If you want a single source of truth and to avoid file cache:

- Add a table, e.g. `ohlcv_cache` (ticker, last_bar_date, data_json or columns, updated_at).
- In `fetch_ohlcv()`: read from this table first; if `last_bar_date >= today - 1` (or your rule), return it; else fetch from API and upsert.

This is optional; file cache plus auto-save already reduces API usage.

### 5. Rate Limit Protection Summary

- One search = at most one API path per data type (OHLCV, valuation) when cache/DB misses.
- Auto-save means each API-driven load is written once, so the next load is a cache/DB hit.
- Show a short status line under each section: “From DB (saved …)” or “Cached (…) so users know they didn’t burn an API call.

### 6. Small Bug Fix

- In `get_valuation_chart_data()`, include `from_database: valuation.get('from_database', False)` in the returned dict so the valuation section correctly shows “📂 From DB” when data came from PostgreSQL.

---

## Implementation Checklist

- [x] **Pass through `from_database`** in `get_valuation_chart_data()`.
- [x] **Auto-save valuation**: after building valuation chart data from API (not from DB), call `save_valuation_to_db()` once (e.g. from `get_valuation_chart_data()` or from the Streamlit page after a successful API-based load).
- [x] **Auto-save TradingView**: after `calculate_tradingview_signals()` and before displaying the chart, call `save_tv_signals_to_cache()` for the current ticker/timeframe.
- [ ] (Optional) Add “last bar date” check to OHLCV cache so cache is invalid only when new trading day data is expected.
- [ ] (Optional) Add PostgreSQL table for OHLCV and use it in `fetch_ohlcv()` for DB-first OHLCV.

These changes keep your two goals aligned: **seamless, valid, up-to-date data** with **minimal frontend effort** and **fewer API calls** to avoid rate limits.
