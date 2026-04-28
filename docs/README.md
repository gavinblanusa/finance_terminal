# Docs index

This folder holds implementation notes and architecture docs left for **future AIs and refactors**. Read these when you need more context than the root README.

| Document | When to use it |
|----------|-----------------|
| **../DESIGN.md** (repo root) | You’re changing Streamlit UI: fonts, tokens, `.gft-*` classes, or layout. |
| **ARCHITECTURE.md** | You need data flow, page→module mapping, or a quick “how does X get its data?” before refactoring or adding features. Includes the **Macro Dashboard** row (FRED `.macro_cache/`, `macro_dashboard_page()`). |
| **DATA_LAYER_REFERENCE.md** | You need dashboard data contracts, optional **FastAPI** paths (`terminal_api`), sources, caches, failures, Pydantic export paths, Market Analysis **SEC Form 4** insider rows, or **Partnerships** EDGAR/Yahoo cache behavior. |
| **OPENBB_COVERAGE.md** | You need which datasets use OpenBB, provider try-order, env toggles (`USE_OPENBB`, timeouts), or how CI keeps this file aligned with `openbb_provider_registry.py`. |
| **PROJECT_LEARNINGS.md** | You want the gstack **`/learn`** snapshot: patterns, pitfalls, and architecture notes from cross-session logs (live source under `~/.gstack/projects/<slug>/`; see **AGENTS.md** for `scripts/gstack-learnings.sh`). |
| **MARKET_ANALYSIS_DATA_REFACTOR.md** | You’re working on Market Analysis data: cache/DB-first behavior, auto-save for valuation and TradingView, or rate-limit handling. It’s a refactor plan from a prior session; valuation and TradingView auto-save are implemented—checklist is partially outdated. |
| **OPEN_SOURCE_REPOS.md** | You're adding or evaluating an open source tool for analysis, data (market/SEC/IPO), or display. Lists repos with install, fit in the project, integration notes, and suggested implementation order. |
| **plans/** (`docs/plans/`) | gstack **/autoplan** or **/office-hours** artifacts: approved implementation plans (for example MDW OHLCV adapter). Not required reading unless you are executing that plan. |

Root README **“For AI and refactors”** remains the single best first stop for entry points, env vars, and where key logic lives.
