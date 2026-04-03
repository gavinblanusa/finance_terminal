# TODOS

Backlog from OpenBB data-layer design (`~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-design-20260401-154308.md`).

## OpenBB / data layer

1. **Optional CI matrix for OpenBB**  
   **Done:** `.github/workflows/ci.yml` runs `pytest` with `USE_OPENBB=true` and `USE_OPENBB=false`.

2. **Verify OpenBB HTTP timeout configuration**  
   **Done for v1:** `docs/OPENBB_COVERAGE.md` documents `OPENBB_REQUEST_TIMEOUT_SEC` (executor around each provider try) vs internal FRED client timeouts. Revisit if you upgrade OpenBB major or add more economy routes.

3. **Inventory drift guard (optional)**  
   **Done:** `scripts/verify_openbb_coverage_doc.py` (CI job `openbb-doc-registry`) ensures every multi-provider chain in `openbb_provider_registry.py` appears in `docs/OPENBB_COVERAGE.md`.

## PORT-lite / factor attribution (2026-04-02)

From `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-design-20260402-203520.md` plus `/plan-eng-review` + `/plan-design-review`:

- **Unit tests for factor + attribution:** Ship in the same PR as attribution (`tests/test_factor_exposure.py`, synthetic OHLCV + factor panel). Not deferred.
- **Scenario replay (historical shock / preset dates):** Deferred past v1 attribution. Implement after factor contribution + residual are stable; use SPY, ^TNX, HYG as labeled proxies only; document “illustrative, not desk stress.”

## Morning stack / autoplan deferrals (2026-04-02)

From `docs/plans/morning-stack-approach-b.md` (Approach B implementation + CEO/design review notes):

- **LLM or NLP event tagging** for headlines (v1 stays rule-based; revisit if tag noise stays high).
- **Separate “Global context” page** (Approach C) if Dashboard density hurts scan time after B ships.
- **Morning digest** (email/push) using the same pipeline.
- **Macro calendar** (scheduled events) vs post-hoc headline tags only.
- **LLM kill/continue doc:** one-page decision on when to add model-based classification (cost, latency, offline).

## QA follow-ups (2026-04-01)

- **Medium — console:** Streamlit 1.55 may still log empty `theme.sidebar` widget/skeleton colors in some setups. Tracked in local QA report under `.gstack/` (gitignored). No reliable `config.toml` fix found yet.
- **Fixed in v0.1.3.0:** Revenue (TTM) on Market Analysis used the wrong income column when `cost_of_revenue` appeared before `total_revenue`; OpenBB income merge now picks an explicit revenue column order.
