<!-- /autoplan restore point: /Users/gavinblanusa/.gstack/projects/gavinblanusa-finance_terminal/main-autoplan-restore-20260403-190934.md -->
# Plan: Optional OHLCV read from market-data-warehouse (Approach A)

**Status:** APPROVED (autoplan gate; user chose Approach A now, defer B)  
**Branch:** main  
**Design input:** `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-design-20260404-000659.md`  
**User decision:** Implement **Approach A** now; **Approach B** (lake as authoritative tier, ops-heavy) deferred.

## Goal

Add an **optional** code path so `fetch_ohlcv` can load daily bars from a local **DuckDB** file and/or **bronze Parquet** layout produced by [joemccann/market-data-warehouse](https://github.com/joemccann/market-data-warehouse), then **fall through** to the existing OpenBB → yfinance → backup chain when unset, missing symbol, or read failure.

## Non-goals (this PR)

- No vendoring of MDW; no IB Gateway or ingest schedules inside this repo.
- No change to Postgres schema; no mandatory new dependencies for default installs (DuckDB optional: extra or lazy import).
- **Approach B** deferred: terminal does not treat the lake as sole source of truth; no admin-only network backfill UX in v1.

## Implementation outline

1. **Env config** (names TBD in impl, examples):
   - `GFT_MARKET_WAREHOUSE_DUCKDB` — path to `market.duckdb` (optional).
   - Optional: `GFT_MARKET_WAREHOUSE_BRONZE` — root of `data-lake/bronze` for direct Parquet read if DuckDB absent.
2. **New module** `app/market_warehouse.py` (or `warehouse_ohlcv.py`):
   - `try_load_ohlcv_from_warehouse(ticker, start_date, end_date) -> Optional[pd.DataFrame]`
   - Map equity ticker → MDW paths (`asset_class=equity/symbol={TICKER}/...` per upstream README).
   - Normalize columns to match `fetch_ohlcv`: datetime index named consistently with existing cache (`Open`, `High`, `Low`, `Close`, `Volume`).
   - DuckDB: `SELECT` with date filter; **read-only connection per attempt** (avoid global mutable connection under Streamlit reruns).
   - Parquet fallback: `pandas.read_parquet` / `pyarrow` if we choose file path without DuckDB.
3. **Wire `market_data.fetch_ohlcv`**:
   - After `_load_from_cache` miss: try warehouse **before** OpenBB. Document that **4h JSON cache can hide fresher warehouse**; optional bypass env deferred to `TODOS.md`.
   - If warehouse returns sufficient rows for caller’s `period_years` / `max` rules, return; else fall through. **Reuse or extract** a small helper with the OpenBB path so “enough history” rules do not drift.
4. **Logging:** when warehouse serves a series, log ticker, row count, and **max bar date** (trust/debug). Use existing print-style or `logging` consistent with `gft.openbb` if touched nearby. **Timeout** connect/query so slow disk does not block the UI path.
5. **Tests:** `tests/test_market_warehouse.py` with temp dir, synthetic Parquet or mocked DuckDB (prefer no real DuckDB binary in CI if heavy; use pandas write_parquet + read back, or `unittest.mock`).
6. **Docs:** `README.md` + `docs/ARCHITECTURE.md` one paragraph; env vars in README “optional MDW” section.

## Open decisions (resolve in implementation)

- **Whitelist vs all symbols:** Default: no whitelist (any ticker attempts warehouse read once path set); optional `GFT_MARKET_WAREHOUSE_TICKERS` later if needed.
- **Futures / vol:** Out of scope for v1 unless trivial same schema; document as follow-up.

## Success criteria

- With env unset: identical behavior to today.
- With env set and valid MDW data: `fetch_ohlcv` returns warehouse-shaped DataFrame; charts and factor code unchanged.
- CI passes without DuckDB installed (skip or mock).

---

## /autoplan — Phase 1 CEO review (SELECTIVE EXPANSION, auto-decided)

**Premise gate:** User confirmed **Approach A now, Approach B later** (this session).

### 0A. Premise challenge

| Premise | Verdict |
|--------|---------|
| Right problem: optional local OHLCV reduces API churn for operators who already run MDW | **Valid** for ICP; incremental for everyone else |
| Doing nothing | No harm; APIs and JSON cache remain |
| 10x reframe | Would be authoritative store + freshness UX everywhere (deferred as B) |

### 0B. Existing code leverage

| Sub-problem | Existing code |
|-------------|----------------|
| OHLCV pipeline, cache, sufficiency rules | `app/market_data.py` `fetch_ohlcv`, `_load_from_cache`, `_save_to_cache` |
| Provider fallbacks | `openbb_adapter`, yfinance, `api_clients` |
| Env pattern | `os.environ` + `load_dotenv` in `market_data` |

### 0C. Dream state (12-month)

```
CURRENT: on-demand APIs + 4h JSON cache
   → THIS PLAN: optional DuckDB/Parquet read before APIs
   → IDEAL: versioned daily bars + explicit as-of + optional B as source of truth
```

### 0C-bis. Alternatives (recap)

- **A** (chosen): adapter in `fetch_ohlcv`, fall through.
- **B** (deferred): lake authoritative, ops-owned ingest (see `TODOS.md`).
- **C**: separate research surface only (deferred; lower priority than A for shared OHLCV).

**RECOMMENDATION:** A — matches user decision and smallest blast radius.

### 0D. SELECTIVE EXPANSION — cherry-picks (auto)

| Proposal | Effort | Decision | Principle |
|----------|--------|----------|-----------|
| UI staleness badge | M | **Defer** TODOS | P3 pragmatic; not required for A |
| Pin MDW version + contract tests | S | **Defer** TODOS | P2 boil lake borderline; do in follow-up |
| Shared “covers window + min rows” helper | S | **Accept** | P4 DRY, P5 explicit |
| Log max(bar date) when warehouse hits | S | **Accept** | P1 completeness (debuggability) |
| Fast timeout on warehouse open/query | S | **Accept** | P1 edge case |

### 0E. Temporal interrogation (resolved in plan)

- **Hour 1:** MDW table/column names from real `market.duckdb` or README; path template in one function.
- **Hour 2–3:** Align `period_years` / `max` truncation with existing OpenBB block.
- **Hour 4–5:** Document JSON cache vs warehouse precedence; optional env to prefer warehouse later.
- **Hour 6+:** CI mocks, `requirements.txt` optional extra for `duckdb`.

### 0F. Mode

**SELECTIVE EXPANSION** with baseline scope = Approach A only.

### CEO dual voices

**CODEX SAYS (CEO — strategy challenge):** *Unavailable — `codex exec` exceeded time budget on this machine.*

**CLAUDE SUBAGENT (CEO — strategic independence):** See task output (summary): critical risks are **staleness**, **bar equivalence** (splits/adjustments), **schema drift**, **JSON cache masking fresher warehouse**, and **singleton DuckDB** threading. Mitigations: log max date, document adjustment policy, version contract (deferred), per-call read-only connections, fast timeout.

**CEO DUAL VOICES — CONSENSUS TABLE**

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Premises valid? | Mostly; pin equivalence | N/A | **Partial** — add logging + docs |
| Right problem? | Yes for lake operators | N/A | **Yes** |
| Scope calibration? | A is tight | N/A | **Yes** |
| Alternatives explored? | Yes; B deferred | N/A | **Yes** |
| Market risks? | Low product diff | N/A | **N/A** |
| 6-month trajectory? | Drift + stale data | N/A | **Flag** contract tests + staleness UX in TODOS |

### Section 1 (Architecture) — CEO

New leaf: `market_warehouse` → optional read → `fetch_ohlcv`. No new HTTP. Rollback: unset env + revert.

```
  [JSON cache] --miss--> [warehouse try] --miss--> [OpenBB] --> [yfinance] --> [backups]
```

Nil/empty/error on warehouse → `None`, fall through (same as today).

### Section 2 (Error & rescue)

Warehouse path: catch OSError, duckdb.Error, parse errors → log context → `None`. No bare `except Exception` without re-raise at top level; match `market_data` style (prefer specific + log).

### Section 3 (Security)

Local paths from env; `Path.resolve()`; no shell. Low risk for single-user terminal.

### Section 4 (Data flow / interaction)

No new UI in A. Edge: partial history → fall through.

### Section 5 (Code quality)

Keep warehouse out of circular imports; thin `fetch_ohlcv` glue.

### Section 6 (Tests)

Cross-ref test artifact: `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-test-plan-20260404-mdw.md`

### Section 7 (Performance)

Bounded query by date range; avoid full table scan if MDW allows; timeout on connect.

### Section 8 (Observability)

Log line when warehouse serves: ticker, row count, **max(date)**.

### Section 9 (Deployment)

No migration; feature is env-gated. Rollback: remove env vars.

### Section 10 (Long-term)

A leaves door open for B; ADR pointer in `TODOS.md` list.

### Section 11 (Design)

**Skipped** — plan has no new UI scope (only “layout” in sense of data lake, not screen layout).

### CEO completion summary

| Area | Result |
|------|--------|
| NOT in scope | IB ingest, ClickHouse, futures/vol v1, Approach B |
| Deferred | Staleness UI, MDW pin, cache-bypass env |
| Accepted adds | max-date log, timeout, shared sufficiency helper |

---

## /autoplan — Phase 2 Design review

**Skipped** (no UI scope: fewer than 2 meaningful matches for dashboard/nav/form/etc.).

---

## /autoplan — Phase 3 Eng review

### Scope challenge

Touches ~3 files: new `app/market_warehouse.py`, `app/market_data.py` (`fetch_ohlcv`), tests + docs. Within blast radius.

### Eng dual voices

**CODEX SAYS (eng — architecture challenge):** *Unavailable — codex exec timed out.*

**CLAUDE SUBAGENT (eng — independent review):** Confirms leaf module pattern; risks: **cache vs warehouse freshness**, **partial history** alignment, **optional deps in CI**, **DuckDB connection strategy** (per-call read-only). Tests must mock/patch `fetch_ohlcv` integration.

**ENG DUAL VOICES — CONSENSUS TABLE**

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Architecture sound? | Yes | N/A | **Yes** |
| Tests sufficient? | Needs explicit plan | N/A | **Artifact written** |
| Performance? | Bounded query | N/A | **Yes** |
| Security? | Low | N/A | **Yes** |
| Error paths? | Fall through | N/A | **Yes** |
| Deployment? | Env-only | N/A | **Yes** |

### Section 1 — Architecture ASCII

```
                    +------------------+
                    |  fetch_ohlcv     |
                    +--------+---------+
                             |
              +--------------+--------------+
              v              v              v
      +---------------+ +---------+ +--------------+
      | JSON cache    | |warehouse| | OpenBB / yf  |
      | .market_cache| | optional | | + backups    |
      +---------------+ +---------+ +--------------+
```

### Section 3 — Test diagram

See `gavinblanusa-main-test-plan-20260404-mdw.md` (paths above).

### Failure modes registry

| Mode | Severity | Mitigation |
|------|----------|------------|
| Wrong MDW schema | High | Validate columns; `None` + fall through |
| Stale warehouse + fresh JSON cache | Med | Document; defer bypass env |
| DuckDB locked | Med | Timeout; fall through |

### Eng completion summary

Ship A with tests + docs; monitor optional dep story in CI.

---

## Cross-phase themes

1. **Staleness and trust** (CEO + Eng): log max bar date; future UI (TODOS).
2. **Equivalence** (CEO): document adjustment assumptions vs yfinance.
3. **Codex unavailable** for both phases — consensus marked N/A; Claude subagent carried dual-voice duty.

---

## NOT in scope

- Approach B implementation, IB automation, vendoring MDW, Postgres OHLCV table, new Streamlit pages for MDW.

## What already exists

- `fetch_ohlcv`, JSON OHLCV cache, OpenBB/yfinance chain (`app/market_data.py`).
- Architecture map in `docs/ARCHITECTURE.md`.

---

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Approach A over B for this PR | Mechanical | P1 completeness of *scoped* delivery | User explicit; B is ops ocean | — |
| 2 | CEO | Defer staleness UI to TODOS | Taste | P3 | A ships without chrome | Full freshness UI |
| 3 | CEO | Accept max-date logging | Mechanical | P1 | Cheap trust win | — |
| 4 | CEO | Accept warehouse query timeout | Mechanical | P1 | Avoid slow disk blocking UX | — |
| 5 | CEO | Defer MDW version pin to TODOS | Taste | P2 | <1d but not blocking A | Blocking on upstream manifest |
| 6 | Eng | Per-call read-only DuckDB | Mechanical | P5 | Simpler than singleton | Shared connection pool |
| 7 | Eng | Tests: mock duckdb + parquet fixture | Mechanical | P1 | CI without binary | — |

---

## /autoplan Final state

**User challenges:** None (models do not contradict user direction).

**Taste choices for optional override:** Staleness UI and MDW pin deferred; promote if you want maximum trust before shipping.

**Review scores:** CEO Claude: strategic risks enumerated; CEO Codex: unavailable. Design: skipped. Eng Claude: sound; Eng Codex: unavailable.

**Deferred to TODOS.md:** Approach B line items + cache-bypass + contract tests (see repo `TODOS.md`).

**Test plan file:** `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-test-plan-20260404-mdw.md`
