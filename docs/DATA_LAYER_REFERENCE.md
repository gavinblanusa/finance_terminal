# Data layer reference

This app is a **Streamlit monolith** (no public HTTP API). This document is the contract for **dashboard-facing data builders**: inputs, output shapes, sources, caches, and common failures. For JSON-friendly snapshots, use Pydantic helpers in [`app/data_schemas.py`](../app/data_schemas.py) (`*_to_schema`, `dump_json_ready`, `build_dashboard_export_payload`).

---

## `macro_context.build_macro_context`

**Purpose:** Cross-asset movers (GMM-lite) and optional rates (BTMM-lite).

| Field (conceptual) | Type | Notes |
|-------------------|------|--------|
| `movers` | list of rows | Yahoo symbols; last / prev / change % |
| `rates` | list of rows | FRED latest observation per series when configured |
| `fred_configured` | bool | `True` if `FRED_API_KEY` set and FRED was queried |
| `errors` | list[str] | Aggregated mover/FRED error strings |

**Sources:** yfinance (daily ~15d), FRED API (`api.stlouisfed.org`) when `FRED_API_KEY` is set.

**Caching:** `_cached_macro_context` in `main.py`, TTL 900s. Cleared by **Refresh Prices**.

**Failures:** Network errors, Yahoo empty history, FRED key missing (rates empty; not an error), FRED series errors per row.

**Example (Python):**

```python
from macro_context import build_macro_context, macro_context_to_dataframes

ctx = build_macro_context()
movers_df, rates_df = macro_context_to_dataframes(ctx)
```

**Schema:** `MacroContextSchema` via `data_schemas.macro_context_to_schema(ctx).model_dump(mode="json")`.

---

## `portfolio_insights.build_portfolio_insights`

**Purpose:** PORT-lite — sector/industry weights, concentration, value-weighted β vs SPY.

**Inputs:**

- `positions`: `[{"ticker": str, "current_value": float}, ...]`
- `get_company_profile_fn(ticker)` → dict with `sector`, `industry`
- `fetch_ohlcv_fn(ticker, period_years)` → OHLCV `DataFrame` (`Close` required)

| Output field | Type | Notes |
|-------------|------|--------|
| `sector_weights` | dict[str, float] | % of portfolio (sum ~100) |
| `industry_weights` | dict[str, float] | % of portfolio |
| `top1_pct`, `top5_pct` | float | Concentration, 0–100 |
| `herfindahl` | float | HHI on weights |
| `portfolio_beta` | float or None | vs SPY |
| `per_ticker_beta` | dict[str, float] | OLS β vs SPY |
| `beta_weights_used` | dict[str, float] | Renormalized weights among names used |
| `data_warnings` | list[str] | Missing sector, short overlap, etc. |

**Sources:** `CompanyProfile` (DB/API via `get_company_profile`), OHLCV for each holding + `SPY` (`fetch_ohlcv`), ~120+ overlapping daily returns.

**Caching:** `_cached_portfolio_insights(positions_key)` in `main.py`, TTL 900s.

**Schema:** `PortfolioInsightsSchema` via `portfolio_insights_to_schema`.

---

## `factor_exposure.build_factor_exposure`

**Purpose:** Fama–French **5-factor** loadings (daily): value-weighted portfolio coefficients and per-ticker regressions.

**Inputs:**

- `positions`: same as PORT-lite
- `fetch_ohlcv_fn(ticker, period_years)` — default build uses `period_years=3`
- Ken French **F-F 5 Factors 2×3 daily** ZIP (see below)

| Output field | Type | Notes |
|-------------|------|--------|
| `portfolio_factor_betas` | dict[str, float] | Keys: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA` |
| `per_ticker_betas` | dict[str, dict[str, float]] | Same factor keys per ticker |
| `per_ticker_r2` | dict[str, float] | In-sample R² |
| `per_ticker_n_obs` | dict[str, int] | Trading days in regression |
| `regression_start` / `regression_end` | str or None | ISO dates on factor calendar |
| `as_of` | str or None | UTC timestamp of build |
| `factors_available` | bool | `False` if download/parse failed |
| `data_warnings` | list[str] | Low R², short overlap, OHLCV skips |

**Sources:**

- Factors: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html) — `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`
- Stock returns: same OHLCV pipeline as the rest of the app

**Caching:**

- File: `.market_cache/ff5_factors_daily.csv` (24h TTL before re-download)
- Streamlit: `_cached_factor_exposure(positions_key, ff_cache_mtime)`, TTL 900s; `ff_cache_mtime` ties cache to on-disk factor file updates

**Model:** For each ticker, \(y = r_{\text{stock}} - RF\), regress on `[Mkt-RF, SMB, HML, RMW, CMA]` with intercept (`sklearn.linear_model.LinearRegression`). French file values are converted from **percent to decimal**. Portfolio betas are value-weighted over tickers with a successful fit.

**Failures:** No network, ZIP parse error, insufficient overlapping history (`MIN_OVERLAP_DAYS` = 120), missing `Close` / volume-only issues on OHLCV.

**Schema:** `FactorExposureSchema` via `factor_exposure_to_schema`.

---

## `tca_estimate.estimate_trade_impact`

**Purpose:** **Illustrative** pre-trade impact (TRA-lite): participation vs ADV and a square-root style cost heuristic (not a calibrated Almgren–Chriss broker model).

**Inputs:**

- `ticker`, `shares` (> 0), `side` (`buy` / `sell`)
- `fetch_ohlcv_fn(ticker, period_years)` — default `period_years=2`

| Output field | Type | Notes |
|-------------|------|--------|
| `participation_rate` | float | `shares / adv_shares` (can exceed 1) |
| `adv_shares` | float | Mean `Volume` over last 20 sessions |
| `adv_dollar` | float | `adv_shares * price_ref` |
| `daily_volatility` / `annualized_volatility` | float | From recent close-to-close returns |
| `estimated_impact_frac` | float | Capped scalar cost fraction |
| `estimated_impact_bps` | float | `impact_frac * 10000` |
| `estimated_impact_usd` | float | `notional * impact_frac` |
| `price_ref` | float | Last close |
| `data_warnings` | list[str] | Zero ADV, high participation, etc. |

**Caching:** `_cached_tca_estimate(ticker, shares, side)` in `main.py`, TTL 900s; cleared on **Refresh Prices**.

**Failures:** Invalid ticker/shares returns `None`. Missing OHLCV returns a result object with zeros and warnings.

**Schema:** `TCASchema` via `tca_to_schema`.

---

## `options_iv_term.build_iv_term_structure`

**Purpose:** ATM implied volatility **term structure** (BVOL / OVME-lite): per listed expiry, IV at the strike nearest spot from Yahoo option chains.

**Inputs:** `ticker`, `max_expirations` (default 12), optional `spot_override` (aligns ATM strike to your quote).

| Output field | Type | Notes |
|-------------|------|--------|
| `ticker` | str | Upper symbol |
| `spot_used` | float or None | Spot from override or yfinance |
| `points` | list | `IVTermPoint`: `expiry`, `dte`, `iv_atm` (decimal, e.g. 0.25 = 25% vol), `strike`, `source` (`avg` / `call` / `put`) |
| `data_warnings` | list[str] | Chain failures, missing expiries |

**Sources:** yfinance `Ticker.options` and `option_chain(expiry)`.

**Caching:** `_cached_iv_term_structure(ticker, spot_key)` in `main.py` (Market Analysis), TTL 600s. Cleared when **Clear Cache** runs on that page.

**Failures:** No options on symbol, network errors, NaN IV in chain.

**Schema:** `IVTermStructureSchema` via `iv_term_structure_to_schema`.

---

## `data_schemas.build_dashboard_export_payload`

**Purpose:** One JSON object combining dashboard analytics for download (Dashboard **⬇ JSON snapshot** button).

**Arguments:** `macro` (required), optional `insights`, `factors`, `tca`.

**Output keys:** `generated_at_utc`, `macro`, and any of `portfolio_insights`, `factor_exposure`, `tca_estimate` when provided.

**Note:** The Dashboard download merges **`tca_estimate`** when the user has run **Execution · TCA** in the same Streamlit session (`st.session_state["gft_export_tca"]`). **Refresh Prices** clears that session key.

---

## `fi_context.build_fi_context_strip`

**Purpose:** **SRCH/YAS-lite** context strip—liquid **ETF and ^TNX proxies** for credit and duration tone (not TRACE or bond inventory).

**Returns:** `(rows: list[FIProxyRow], errors: list[str])` — each row has `symbol`, `label`, `last`, `previous_close`, `pct_change`, optional `error`.

**Symbols:** `^TNX`, `HYG`, `LQD`, `TLT`, `IEF` (see `FI_PROXIES` in code).

**Caching:** `_cached_fi_context_strip()` in `main.py`, TTL 900s; cleared on **Refresh Prices**.

---

## `options_black_scholes.black_scholes_european`

**Purpose:** **OVME-lite** closed-form **European** call/put under Black–Scholes (continuous yield `q`, scipy `norm.cdf`).

**Inputs:** `spot`, `strike`, `time_years`, `rate`, `volatility` (annual decimals), optional `dividend_yield`.

**Returns:** `BlackScholesResult` with `call_price`, `put_price`, `d1`, `d2`, and echo of inputs.

**UI:** Market Analysis expander uses cached **`_cached_tnx_last_percent()`** to seed risk-free (%) from **^TNX** last close.

---

## HTTP API (FastAPI)

Optional **read-only** REST surface in [`app/terminal_api.py`](../app/terminal_api.py). Run from project root:

```bash
PYTHONPATH=app uvicorn terminal_api:app --host 127.0.0.1 --port 8800
```

- **OpenAPI / Swagger UI:** `http://127.0.0.1:8800/docs`
- **OpenAPI JSON:** `http://127.0.0.1:8800/openapi.json`

**Auth:** If `GFT_API_KEY` is set in `.env`, all routes except `GET /health` require header:

`Authorization: Bearer <GFT_API_KEY>`

If `GFT_API_KEY` is unset, routes are open (local dev only—do not expose publicly).

**CORS:** Set `GFT_CORS_ORIGINS` to `*` (any origin; credentials disabled) or a comma-separated allowlist. Unset = no CORS middleware.

**Rate limiting:** [slowapi](https://github.com/laurent/slowapi) per client IP. `GFT_RATE_LIMIT` defaults to `60/minute`. Set to `0`, `off`, or `none` for a very high cap (effectively off).

| Method | Path | Response |
|--------|------|----------|
| GET | `/health` | `status`, `auth_configured`, `cors_configured`, `rate_limit` — no auth |
| GET | `/v1/macro` | `MacroContextSchema` JSON |
| GET | `/v1/fi` | List of FI proxy rows (same columns as dashboard strip) |
| GET | `/v1/portfolio` | Full portfolio snapshot dict (same as `fetch_portfolio_snapshot_dict`) or 503 |
| GET | `/v1/analytics/dashboard?include_factors=true` | `build_dashboard_export_payload` shape (no session TCA) |
| GET | `/v1/options/iv-term?ticker=&spot=&max_expirations=12` | `IVTermStructureSchema` JSON |
| POST | `/v1/analytics/tca` | Body `{"ticker","shares","side":"buy"\|"sell"}` → `TCASchema` |
| POST | `/v1/options/black-scholes` | Body: `spot`, `strike`, `time_years`, `rate`, `volatility` (decimals), optional `dividend_yield` |

**Examples:**

```bash
curl -s http://127.0.0.1:8800/health
curl -s -H "Authorization: Bearer YOUR_KEY" http://127.0.0.1:8800/v1/macro
curl -s "http://127.0.0.1:8800/v1/options/iv-term?ticker=AAPL&spot=200"
curl -s -X POST http://127.0.0.1:8800/v1/analytics/tca \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","shares":100,"side":"buy"}'
```

**Implementation notes:** Portfolio and analytics routes use [`app/portfolio_snapshot.py`](../app/portfolio_snapshot.py) and [`app/analytics_export.py`](../app/analytics_export.py). The Streamlit app’s `get_portfolio_data()` calls the same snapshot function (cached in Streamlit only).

---

## Environment variables (dashboard-related)

| Variable | Used by |
|----------|---------|
| `FRED_API_KEY` | `macro_context.fetch_fred_rates` |
| `GFT_API_KEY` | Optional Bearer auth for `terminal_api` |
| `GFT_CORS_ORIGINS` | `*` or comma-separated origins for FastAPI CORS |
| `GFT_RATE_LIMIT` | slowapi limit (default `60/minute`; `0`/`off`/`none` ≈ disable) |
| API keys for OpenBB / yfinance / Polygon / etc. | `market_data.fetch_ohlcv` and quotes (see README) |

---

## Versioning note

Output shapes are stable for **programmatic export** when accessed through `data_schemas` models. If you add fields, extend the Pydantic models and this document together.
