# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Gavin Financial Terminal is a single-service Python/Streamlit app backed by a PG database. All app code lives in `app/`. See `README.md` for entry points, env vars, project structure, and the "For AI and refactors" section. See `docs/ARCHITECTURE.md` for data flow and page-to-module mapping. Dashboard factor loadings download Ken French data into `.market_cache/ff5_factors_daily.csv` (see `docs/DATA_LAYER_REFERENCE.md`). FI proxy strip and ^TNX for BS/rates use **yfinance** only. JSON export can include **TCA** from `st.session_state` after running TCA on the Dashboard.

### Starting services

1. **PG database** must be running before the app starts:
   ```
   sudo pg_ctlcluster 16 main start
   ```
2. **Streamlit app**:
   ```
   streamlit run app/main.py --server.port 8501 --server.headless true
   ```
   The app auto-creates DB tables on first connection.
3. **Optional FastAPI** (read-only JSON): `PYTHONPATH=app uvicorn terminal_api:app --host 127.0.0.1 --port 8800` — see `docs/DATA_LAYER_REFERENCE.md`. If `GFT_API_KEY` is set, send `Authorization: Bearer …` on `/v1/*`. Env: `GFT_CORS_ORIGINS`, `GFT_RATE_LIMIT` (slowapi).

### Non-obvious gotchas

- **pg_hba.conf**: The VM's PG is configured with `trust` auth for local TCP connections (127.0.0.1 and ::1). The `.env` DB credentials match the local PG superuser. No password verification actually occurs due to trust auth — this is intentional for the dev environment.
- **`~/.local/bin` on PATH**: pip installs scripts (including `streamlit`, `openbb-build`, `ruff`) to `~/.local/bin`. This is added to PATH via `~/.bashrc`. If a new shell doesn't find `streamlit`, run `export PATH="$HOME/.local/bin:$PATH"`.
- **OpenBB build step**: After installing or changing OpenBB provider extensions, run `openbb-build` to regenerate the Python interface. This is included in the update script.
- **Tests**: FastAPI contract tests live in `tests/` (`pytest`); `pytest.ini` sets `pythonpath = app`. Run `pytest tests/` from the repo root. CI (`.github/workflows/ci.yml`) runs the suite with **`USE_OPENBB=true`** and **`USE_OPENBB=false`**, and a **`openbb-doc-registry`** job runs `scripts/verify_openbb_coverage_doc.py`. Targeted runs: `pytest tests/test_terminal_api.py` (mocks, no network/DB), `pytest tests/test_openbb_kernel.py` (OpenBB kernel/adapter mocks), `pytest tests/test_partnership_signal.py`, `pytest tests/test_factor_exposure.py` (synthetic factors / attribution). Linting: `ruff check app/`.
- **Module imports use relative-style**: `app/db.py` imports `from models import Base` (not `from app.models`). The app expects to be launched from the project root with `streamlit run app/main.py`, which adds `app/` to `sys.path`. When running ad-hoc scripts, prepend `app/` to `sys.path` (e.g. `sys.path.insert(0, 'app')`).
- **API keys are optional**: The app works with zero API keys; it falls back to yfinance and SEC EDGAR (both free, no key). More keys improve data coverage but are not required for development.

### Gstack project learnings (`/learn`)

Cross-session notes for this repo live in **`~/.gstack/projects/<slug>/learnings.jsonl`** (append-only JSONL). The **slug comes from `git remote`**, not the local folder name (this repo: `gavinblanusa-finance_terminal` because origin is `gavinblanusa/finance_terminal`).

- **Append** (from repo root, so slug resolves):  
  `~/.claude/skills/gstack/bin/gstack-learnings-log '{"skill":"learn","type":"architecture","key":"my-key","insight":"One sentence.","confidence":8,"source":"documented"}'`
- **Search**: same directory, then  
  `~/.claude/skills/gstack/bin/gstack-learnings-search --query "keyword" --limit 20`
- **Wrapper** (cd’s to repo root for you): `scripts/gstack-learnings.sh recent`, `scripts/gstack-learnings.sh search "keyword" 20`, `scripts/gstack-learnings.sh log '<json>'`
- **Requires [bun](https://bun.sh)** on PATH (`gstack-learnings-log` validates JSON with bun). First log creates `~/.gstack/projects/<slug>/` automatically.
