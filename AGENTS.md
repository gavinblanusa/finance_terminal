# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Gavin Financial Terminal is a single-service Python/Streamlit app backed by a PG database. All app code lives in `app/`. See `README.md` for entry points, env vars, project structure, and the "For AI and refactors" section. See `docs/ARCHITECTURE.md` for data flow and page-to-module mapping.

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

### Non-obvious gotchas

- **pg_hba.conf**: The VM's PG is configured with `trust` auth for local TCP connections (127.0.0.1 and ::1). The `.env` DB credentials match the local PG superuser. No password verification actually occurs due to trust auth — this is intentional for the dev environment.
- **`~/.local/bin` on PATH**: pip installs scripts (including `streamlit`, `openbb-build`, `ruff`) to `~/.local/bin`. This is added to PATH via `~/.bashrc`. If a new shell doesn't find `streamlit`, run `export PATH="$HOME/.local/bin:$PATH"`.
- **OpenBB build step**: After installing or changing OpenBB provider extensions, run `openbb-build` to regenerate the Python interface. This is included in the update script.
- **No automated test suite**: The codebase has no tests directory or test framework configured. Linting can be done with `ruff check app/`.
- **Module imports use relative-style**: `app/db.py` imports `from models import Base` (not `from app.models`). The app expects to be launched from the project root with `streamlit run app/main.py`, which adds `app/` to `sys.path`. When running ad-hoc scripts, prepend `app/` to `sys.path` (e.g. `sys.path.insert(0, 'app')`).
- **API keys are optional**: The app works with zero API keys; it falls back to yfinance and SEC EDGAR (both free, no key). More keys improve data coverage but are not required for development.
