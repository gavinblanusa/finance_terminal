# Project: Gavin Financial Terminal

**Objective:** A personal one-stop financial terminal—your own "fiscal.ai" for portfolio, research, SEC intelligence, and future workflows. All your financial needs in one place, with your data and your stack.

## Inspiration

This project is a personal compilation of capabilities found in platforms like:

- **Fiscal.ai** — AI summaries of filings/earnings, institutional-grade financials, 13F and ownership/insider, dashboards and watchlists, earnings calendar, analyst estimates.
- **Unusual Whales** — Congress/Senate trading, SEC filings (13D/F, insider), options flow and alerts, custom notifications.
- **TIKR** — Global screening, superinvestor/13F tracking, valuation builder, portfolio and watchlist, news and earnings transcripts.

The goal is one terminal that combines these themes: portfolio and tax control, market research and valuation, SEC and regulatory intelligence, IPO and events, and (over time) screening, alerts, and optional AI summarization.

## Current scope (six areas)

1. **Dashboard** — Portfolio value, allocation, tax summary, unrealized gains, alerts for lots nearing long-term.
2. **Portfolio & Taxes** — Trade entry and CSV import, HIFO tax lots, long-term vs short-term tracking, cost basis and gain per lot.
3. **Market Analysis** — Ticker research, OHLCV and TradingView-style charts, valuation (P/E, revenue), company profile, fundamentals, news, technical signals.
4. **IPO Vintage Tracker** — Upcoming IPO calendar, 1/2/3-year post-IPO performance, registry and anniversary alerts.
5. **Partnerships** — SEC 8-K Item 1.01 partnership events, configurable watchlist and counterparty highlighting.
6. **13F Institutional Holdings** — Quarterly 13F filings by institution, holdings view, compare quarters, by-CUSIP and overlap analysis.

## Direction / roadmap

- **Screening and filters** — Stock and fund screeners with configurable criteria.
- **Alerts and notifications** — Custom alerts for price, filings, earnings, or portfolio events.
- **Expanded SEC** — Insider transactions, Congress-style trading data (where available), 13D/F and other filing types.
- **Optional AI summarization** — Summaries of filings, earnings calls, or news (when adding AI tooling).
- **API or export** — Programmatic access or export of your portfolio and saved data.

## Tech stack

- **Frontend:** Streamlit (with streamlit-lightweight-charts for main price chart).
- **Database:** PostgreSQL (SQLAlchemy) — trades, watchlist, IPO registry, valuation history, company profile/fundamentals.
- **Data layer:** OpenBB (primary), FinanceToolkit (fundamentals), Polygon/FMP/Finnhub/Alpha Vantage/EODHD/Twelve Data for prices and fundamentals; SEC EDGAR for 8-K and 13F.
- **Caches:** File caches at project root (`.market_cache/`, `.ipo_cache/`, `.edgar_cache/`) and DB-first where applicable.

For implementation details, entry points, and data flow, **README.md** and **docs/ARCHITECTURE.md** are the source of truth.
