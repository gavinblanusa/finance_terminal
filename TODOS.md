# TODOS

Backlog from OpenBB data-layer design (`~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-design-20260401-154308.md`).

## OpenBB / data layer

1. **Optional CI matrix for OpenBB**  
   **Done:** `.github/workflows/ci.yml` runs `pytest` with `USE_OPENBB=true` and `USE_OPENBB=false`.

2. **Verify OpenBB HTTP timeout configuration**  
   **Done for v1:** `docs/OPENBB_COVERAGE.md` documents `OPENBB_REQUEST_TIMEOUT_SEC` (executor around each provider try) vs internal FRED client timeouts. Revisit if you upgrade OpenBB major or add more economy routes.

3. **Inventory drift guard (optional)**  
   **Done:** `scripts/verify_openbb_coverage_doc.py` (CI job `openbb-doc-registry`) ensures every multi-provider chain in `openbb_provider_registry.py` appears in `docs/OPENBB_COVERAGE.md`.

## QA follow-ups (2026-04-01)

- **Medium — console:** Streamlit 1.55 logs empty `theme.sidebar` widget/skeleton colors despite `[theme.sidebar]` in `.streamlit/config.toml`. See ISSUE-001 in `.gstack/qa-reports/qa-report-localhost-2026-04-01.md`.
- **Medium — data:** Market Analysis / AAPL: Revenue (TTM) shows N/A while other fundamentals load (ISSUE-002).
