<!-- /autoplan restore point: /Users/gavinblanusa/.gstack/projects/gavinblanusa-finance_terminal/main-autoplan-restore-20260402-163247.md -->

# Plan: Morning stack Approach B (Dashboard macro + news lift)

## Executive summary

Lift the Dashboard “morning stack” toward Bloomberg-style **context before positions**: add **vol-normalized macro movers**, **richer US rates from FRED**, and **rule-based event tags** on book-scoped ranked news. Data sources stay free-tier friendly (Yahoo daily + FRED). No global TOP, no tick data, no non-US BTMM grid in this plan.

## Problem and user choice

From office-hours comparison (2026-04-02), Approach **B** was selected over (A) docs-only tweaks and (C) separate Global Context page.

## Premises (for implementation)

1. **Movers:** Yahoo Finance daily bars remain the source for cross-asset % change; “vol” means **20 trading days of close-to-close returns** on the same series (or documented exception for indices where we show σ of % moves).
2. **Rates:** FRED remains optional via `FRED_API_KEY`; new series must degrade to empty rows with existing callout patterns, not hard errors.
3. **News:** Ranked feed stays **portfolio + watchlist** scoped; tags are **assistive** (scan + sort hints), not a replacement for score.
4. **Honesty:** UI copy or tooltips flag **daily data**, **FRED as-of lag**, and **ETF proxies** where applicable (DESIGN.md: trust).

## Scope in

| Area | Change |
|------|--------|
| `app/macro_context.py` | Extend mover pipeline: realized vol (20d), optional **change / σ** (or percentile band). Define behavior for **VIX** (level not price vol—document omit or alternate label). FX: σ on % change of spot is meaningful; keep. |
| FRED | Add **SOFR** (e.g. `SOFR`), optional **5Y** `DGS5` and/or **30Y** `DGS30` if stable; extend `FRED_RATES` and `fetch_fred_rates`. |
| `app/relevant_news.py` | Add `event_tags: List[str]` on `RankedNewsItem` via regex/keyword buckets: earnings, M&A, legal/regulatory, FDA/health, ratings, guidance, macro-ish terms. Optional small score bump for “high-impact” tags (configurable constants). |
| `app/main.py` | Dashboard: show new mover columns; news cards/rows show tags (muted pills, DESIGN.md tokens). |
| `app/data_schemas.py` | Extend `MacroContextSchema` / news export if JSON export includes these shapes. |
| Docs | `docs/ARCHITECTURE.md`, `docs/DATA_LAYER_REFERENCE.md` short deltas. |
| Tests | `tests/`: pure functions for σ calculation, tag extraction, empty-history edges. |

## Scope out

- Bloomberg-style global TOP, NLP market-impact ranking, LLM tagging (defer to TODOS).
- Additional macro symbols beyond agreed list (separate PR).
- FI strip vol normalization (^TNX as yield: treat as level change or skip σ—document).
- Real-time or futures “overnight” session logic.

## Implementation alternatives (recap)

- **Minimal:** Only FRED series expansion + docs.
- **Chosen B:** Vol columns + FRED + tags (this plan).
- **Maximal:** + LLM tags + new page (deferred).

## Success criteria

- User can sort/scan movers for “**large vs its own volatility**” in one table.
- Rates block shows SOFR next to EFFR/Treasuries when key present.
- Headlines show **1–3 tags** without cluttering the tape aesthetic.
- `pytest tests/` green; no new required API keys.

## Risks

- **VIX / yield interpretation:** Wrong σ semantics confuse users; mitigate with column labels and omitted rows.
- **Performance:** Extra pass on same `history` window; keep single fetch per symbol (compute σ from same DataFrame as change %).
- **News false positives:** Regex tags; tune word boundaries like existing `relevant_news` keywords.

### Implementation amendment (eng review, auto-approved)

Current `fetch_macro_movers` uses `period="15d"`, which is **insufficient for 20 trading-day realized vol** (need enough closes for 20 returns). **Mandatory:** widen Yahoo history (e.g. `period="3mo"` or `60d`) for mover rows while keeping last vs prior close for display change. Single `history` call feeds both % change and σ.

---

## Phase 1 — CEO review (/autoplan auto-decisions)

_Mode: SELECTIVE EXPANSION (hold book-scoped TOP, add completeness inside blast radius)._

### 0A Premise challenge

| Premise | Verdict |
|---------|---------|
| Yahoo+daily is enough for “GMM-lite” | **Hold** — matches free stack; vol column adds signal without new vendor. |
| FRED expansion low risk | **Hold** — same pattern as existing series. |
| Rule tags vs LLM | **Hold for v1** — completeness in blast radius favors shipping deterministic tags; LLM is ocean. |

### 0B Leverage map (sub-problem → existing code)

- Movers table: `macro_context.fetch_macro_movers`, `macro_context_to_dataframes`, `_style_macro_movers_styler` in `main.py`.
- Rates: `fetch_fred_rates`, `FRED_RATES`.
- News: `build_relevant_news`, `_gft_render_news_hero_cards`, `_cached_relevant_news`.
- Export: `macro_context_to_schema`, `build_dashboard_export_payload`.

### 0C Dream state diagram

```
CURRENT (daily % only, US FRED subset, score-only news)
    → THIS PLAN (σ column, SOFR+curve extension, event tags)
    → 12-MONTH IDEAL (optional: global macro page, curated wire, vol surface hooks) — NOT THIS PR
```

### 0D NOT in scope (deferred)

- Separate “Global context” page (Approach C).
- LLM event classification.
- Non-US sovereign BTMM.

### 0E Temporal check

- **Hour 1:** `macro_context` dataclass + vol helper + tests.
- **Hour 6+:** FRED rows, news tags, Dashboard styling, schema, docs.

### CEO dual voices

#### CLAUDE SUBAGENT (CEO — strategic independence)

- **Critical:** Incremental polish risk; reframe around one **morning outcome** (“60s brief tied to my book + risk”); σ/tags support that story.
- **High:** “Context before positions” may need **interpretation + sequencing**, not only density; add one-line “why it matters” tied to holdings or validate with sessions.
- **High:** Yahoo **daily** is weak for true “morning” semantics; add **as-of / pre-market = prior close** honesty everywhere.
- **High:** Tags can **erode trust**; ship conservative thresholds, “heuristic” tooltips, score-first default.
- **Medium:** σ semantics mixed across VIX/yields; **omit or split** rather than wrong number.
- **Medium:** Approach C under-explored vs dashboard cramming; paper-prototype if scan time suffers.
- **Medium:** No moat in commodity columns; lean on **portfolio linkage + lineage + export**.
- **Verdict (subagent):** **narrow** scope and sharpen narrative before stacking metrics.

#### CODEX SAYS (CEO — strategy challenge)

`codex exec` ran >10 minutes without captured stdout; process was terminated. Treat as **outside voice unavailable** for this run. Themes above overlap expected Codex critique: staleness, tag trust, σ semantics, competitive commodity features.

#### CEO DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude   Codex   Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Premises valid?                   mostly  N/A     HOLD with honesty fixes
  2. Right problem to solve?           narrow  N/A     DISAGREE — user chose full B
  3. Scope calibration correct?        trim    N/A     ADD as-of + history window fix
  4. Alternatives sufficiently explored? partial N/A   FLAG Approach C for later
  5. Competitive/market risks covered? yes   N/A     Moat = book + export not pills
  6. 6-month trajectory sound?         caution N/A     Hinges on tag/σ correctness
═══════════════════════════════════════════════════════════════
```

### CEO completion summary

- **NOT in scope (reiterated):** Global TOP, LLM tags v1, non-US BTMM, tick/overnight session engine.
- **Error & Rescue Registry:** FRED missing → existing empty + callout; Yahoo fail → per-row Note; tag noise → defer LLM, tune regex (TODOS.md).
- **Failure Modes Registry:** Misleading σ on VIX/yields (omit/label); **15d history too short** (fixed in plan amendment); false tag positives (tooltips + conservative rules); score bump surprises (document in UI).
- **Dream state delta:** This PR moves daily context toward GMM-lite credibility; 12-month ideal still needs global page or digest (deferred).

---

## Phase 2 — Design review (UI scope: yes)

### Step 0 — Design scope completeness

**8/10:** Plan names DESIGN.md tokens and existing styler hooks; needs explicit empty-cell and column-order spec before implementation.

### CLAUDE SUBAGENT (design — independent review)

Scorecard:

| Dimension | Score | Note |
|-----------|-------|------|
| Information hierarchy | 7 | σ vs % weight and column order undecided |
| Loading/empty/error | 6 | Short σ history, skeletons unspecified |
| Trust & honesty | 8 | Aligns with DESIGN.md if “rule-based” copy |
| Density vs scan | 6 | More columns + pills strain tape |
| Accessibility | 5 | Non-color direction cues not specified |
| Specificity | 6 | Tooltips, sort, overflow light |
| Haunt list | — | VIX row, ^TNX, >3 tags, rates layout with new FRED rows |

**Top 3 fixes:** (1) UI spec for empty/short-history cells + tooltips, (2) lock column order + default sort (% primary), (3) a11y: +/- or icons, pill contrast/focus.

#### CODEX SAYS (design — UX challenge)

Not run (time budget; design subagent covered primary gaps).

#### Design litmus consensus

Ship with **explicit UI spec addendum** in implementation PR: column order, σ as secondary signal, “heuristic tag” microcopy, a11y pass on movers + pills.

### Design passes 1–7 (auto-decisions)

1. **Hierarchy:** % change remains primary scan column; σ right or optional column — **mechanical, P5 explicit**.
2. **States:** Match existing `_gft_dash_callout` patterns; short history shows em dash + caption — **mechanical**.
3. **Trust:** Subtitle “Daily closes · rule-based tags” in section kicker — **mechanical**.
4. **Density:** Cap tags at 3; truncate headline unchanged — **mechanical**.
5. **A11y:** Add non-color direction cue in styler or column text — **taste** (surface at gate: icon vs +/- prefix).
6. **Motion:** No new animations — **mechanical**.
7. **Responsive:** Streamlit wide layout; table scroll acceptable — **mechanical**.

**Phase 2 complete.** Codex unavailable; Claude subagent: 7 issues worth addressing; consensus: ship B with UI spec + a11y choice.

---

## Phase 3 — Eng review

### Scope challenge (code-backed)

Read `macro_context.py` (`period="15d"`), `relevant_news.py`, `data_schemas.py`, `main.py` dashboard paths. **Finding:** 20d vol **cannot** be correct without widening history; plan amended. **Finding:** Export today focuses on macro `MacroContextResult`; news tags can stay UI-only v1 to limit schema churn — **auto-decide P4 DRY** unless export explicitly needs tags (defer).

### CLAUDE SUBAGENT (eng — independent review)

**Dependency graph:**

```
yfinance ──► fetch_macro_movers ──► MacroMoverRow ──┬──► macro_context_to_dataframes ──► main (Dashboard)
                                                    ├──► build_macro_context ──► MacroContextResult
                                                    └──► macro_context_to_schema ──► build_dashboard_export_payload

FRED_API_KEY ──► requests ──► fetch_fred_rates ──► FredRateRow ──► (same)

portfolio/watchlist ──► build_relevant_news ──► RankedNewsItem ──► main (news UI)
fetch_company_news ──► build_relevant_news
```

**Edge cases:** Short history, VIX semantics, yield Δ%, FRED holes, tag false positives, score+bump coupling, export additive fields, timezone sort (pre-existing).

**Tests:** See artifact `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-test-plan-20260402.md`.

**Security:** N/A new surface; escape tag text in HTML like headlines.

**Hidden complexity:** One history fetch for dual metrics; styler regression; dual keyword systems (`NEWS_KEYWORDS` vs tags) should share boundaries where possible — **taste** (consolidate in one module or document layering).

#### CODEX SAYS (eng — architecture challenge)

Not run (time budget).

#### ENG DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════
  Dimension                           Claude   Codex   Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Architecture sound?             yes     N/A     YES + widen history
  2. Test coverage sufficient?       gaps    N/A     Add unit tests per test plan
  3. Performance risks?              low     N/A     Single fetch mitigates
  4. Security threats?               none    N/A     N/A
  5. Error paths handled?            mostly  N/A     Specify σ-empty cells
  6. Deployment risk?                low     N/A     No infra change
═══════════════════════════════════════════════════════════════
```

### Eng completion summary

- **Architecture:** Extend dataclasses + dataframe columns + optional schema fields; no new services.
- **Test plan file:** `~/.gstack/projects/gavinblanusa-finance_terminal/gavinblanusa-main-test-plan-20260402.md`
- **TODOS.md:** Updated with deferrals (LLM tags, Global page, digest, macro calendar, LLM decision doc).

### Failure modes registry (eng)

| Risk | Severity | Mitigation |
|------|----------|------------|
| σ often null (short history) | Medium | Widen period; clear empty cells |
| Wrong σ on VIX | High | Omit or separate label |
| Tag spam / wrong tag | Medium | Conservative rules + tooltip |
| Export breakage | Low | Additive Pydantic fields only |

---

## Cross-phase themes

1. **Honesty / as-of / “daily not live”** — CEO + Design both stress labeling; eng confirms Yahoo window limits.
2. **Trust in tags** — CEO + Design; eng adds escaping and sort coupling.
3. **σ semantics** — all three phases; unanimous: do not ship misleading blended column.

---

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Keep Approach B scope | User challenge pending | P6 | User explicitly chose B in prior message | “narrow” to metrics-only strip without tags |
| 2 | CEO | Add morning honesty copy + as-of | Mechanical | P1 | CEO phase gap | None |
| 3 | Eng | Widen Yahoo `period` for σ | Mechanical | P2 | 15d insufficient for 20d vol | Keep 15d |
| 4 | Eng | News tags UI-only v1 export | Mechanical | P4 | Avoid schema churn | Export tags day one |
| 5 | Design | % change primary column, σ secondary | Taste | P5 | Scan speed | σ primary |
| 6 | Design | Non-color direction cue required | Taste | P1 a11y | DESIGN.md alignment | Color-only |
| 7 | Eng | Consolidate keyword/tag rules or document layers | Taste | P4 | DRY | Silent duplication |

---

## /autoplan — Final approval gate (for user)

**Premise gate:** You previously chose **B** (vol + FRED + tags). Confirmed for this plan file unless you override.

### User challenge (both outside voices vs your direction)

**Challenge 1: Scope narrative (CEO subagent)**

- **You said:** Ship Approach B as specified (σ column, more FRED, event tags on book news).
- **Outside voice recommends:** **Narrow** to a sharper “single morning outcome” first; treat σ/tags as supporting, consider cutting or sequencing after a honesty banner + book linkage story.
- **Why:** Incremental columns rarely move competitive positioning; risk of busy UI without proven job-to-be-done.
- **Blind spot:** Solo power-user terminal may value density over narrative; no user interview cited.
- **If we are wrong:** You delay B and under-ship visible progress while competitors add AI briefs.

**Your call.** Default in autoplan is **your original direction (full B)** unless you reply to narrow.

### Taste decisions (recommendations)

- **T5:** Non-color direction for movers: recommend **+/- prefix in text column** (simpler than icons in Streamlit styler).
- **T7:** Tag vs `NEWS_KEYWORDS`: recommend **shared boundary helpers** in `relevant_news.py` to avoid drift.

### Review scores (this run)

- **CEO:** Strong critique; Codex **unavailable** (timeout).
- **Design:** Subagent only; scores 5–8 by dimension; Codex skipped.
- **Eng:** Subagent only; **critical** history-window fix merged into plan; Codex skipped.

### Deferred items

See **TODOS.md** section “Morning stack / autoplan deferrals”.

---

**Next step after you approve:** implement per amended plan, then `/ship` when ready.

