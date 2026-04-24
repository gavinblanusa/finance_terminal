# Design System — Gavin Financial Terminal

## Product context

- **What this is:** A personal financial intelligence workspace (Streamlit + PostgreSQL) modeled after professional terminals: portfolio, tax lots, market analysis, macro context, IPO tracking, 13F holdings, partnerships.
- **Who it is for:** A single power user (you) and anyone self-hosting the same stack. Expects data density, honest data caveats, and fast scanning.
- **Space / industry:** Retail + pro-sumer fintech, Bloomberg / TradingView adjacency, not consumer banking marketing.
- **Project type:** Data-dense **app UI** (workspace-first). No public marketing shell inside Streamlit.

## Aesthetic direction

- **Direction:** Industrial utilitarian with a **warm terminal** accent (amber rail). Calm surfaces, sharp type hierarchy, no decorative illustration layer.
- **Decoration level:** Intentional minimal … thin borders, optional soft radial highlights on section cards (`app/main.py` `.gft-dash-section`). No gradient blobs, no purple “AI SaaS” defaults.
- **Mood:** Serious desk software: numbers first, copy explains limits (delayed quotes, proxy instruments, FRED as-of dates).
- **Reference:** In-repo preview dogfooding these tokens: `~/.gstack/projects/Invest/designs/design-consultation-20260331/preview.html` (also `/tmp/design-consultation-preview-gft.html` if present).

## Typography

- **Display / section titles:** **Sora** (500–700). Used for app title, page titles, `.gft-dash-title`, kickers context.
- **Body / UI:** **IBM Plex Sans** (400–600). Descriptions, captions, sidebar copy.
- **UI labels:** Same as body unless Streamlit default overrides; prefer Plex when using custom HTML blocks.
- **Data / tables / tickers:** **JetBrains Mono** (400–600) with **`font-variant-numeric: tabular-nums`** for aligned columns. Adopt via `st.markdown` + HTML/CSS or `column_config` where applicable; goal is numeric alignment on dashboard and tape-style rows.
- **Code / formulas:** JetBrains Mono or project default for any inline code in docs.
- **Loading:** Google Fonts links already embedded in `app/main.py` for Sora + IBM Plex Sans. Add JetBrains Mono to the same block when wiring tabular UI.
- **Scale (reference):**
  - Kicker: ~0.65rem, wide letter-spacing, uppercase
  - Section title: ~1.28rem (`.gft-dash-title`)
  - Body: 0.78–1rem for secondary text; Streamlit base ~16px for widgets

**Do not** use as primary: Inter, Roboto, Arial, Open Sans, Poppins (generic stack). **Avoid** emoji in global chrome (nav titles, primary actions); keep inner copy utility-first.

## Color

- **Approach:** Restrained … one warm accent, neutrals carry the UI, semantic green/red for market direction.

| Token | Hex | Usage |
|--------|-----|--------|
| Background | `#0E1117` | Page background (matches `.streamlit/config.toml` `backgroundColor`) |
| Surface | `#262730` | Secondary surfaces, sidebar (`secondaryBackgroundColor`) |
| Text primary | `#F8FAFC` | Headings, primary body on dark |
| Text muted | `#94A3B8` | Hints, timestamps, secondary labels |
| Accent | `#E8A838` | Brand rail, `.gft-dash-kicker`, primary emphasis in custom HTML |
| Research stack | `#2DD4BF` | Research band accent only (CSS variable `--gft-research-accent`) |
| Execution stack | `#A5B4FC` | Execution band accent only (`--gft-exec-accent`) |
| Positive | `#4ADE80` | Up moves, gains |
| Negative | `#F87171` | Down moves, losses |
| Streamlit primary | `#E8A838` | Native Streamlit chrome (`primaryColor`) after the 2026-04-23 terminal chrome retheme |

- **Semantic:** Success / warning / error in custom panels should align with existing `.gft-dash-msg-*` and alert patterns in `app/main.py` (greens, ambers, reds as already defined).
- **Dark mode:** Primary. Streamlit theme is dark-first.
- **Light mode (optional):** Not a first-class Streamlit theme today; if added, invert neutrals, desaturate accent ~10%, keep semantic green/red.

## Spacing

- **Base unit:** 8px.
- **Density:** Comfortable-dense … readable without feeling like a spreadsheet wall. Section blocks: ~1rem padding vertical, ~1.2rem horizontal (see `.gft-dash-section`).
- **Scale (reference):** 4 / 8 / 12 / 16 / 24 / 32px for gaps between bands.

## Layout

- **Approach:** Grid-disciplined … sidebar navigation + wide main. Inside main: stacked **bands** with kickers (Morning / Research / Execution), not a mosaic of equal cards unless the card is the interaction.
- **Grid:** Follow Streamlit columns; prefer `[1,1,4]` style splits for actions vs content.
- **Max content width:** Streamlit wide layout; avoid full-bleed paragraph text without a max-width in custom HTML.
- **Border radius:** Hierarchy … small controls ~4–6px (`.gft-dash-section` uses 6px). Avoid one uniform “bubble” radius on everything.

## Motion

- **Approach:** Minimal-functional. Short section reveal is acceptable (existing `gftDashReveal`).
- **Easing:** `ease-out` for enter, avoid animating width/height/layout.
- **Duration:** ~150–400ms for UI feedback; respect `prefers-reduced-motion` if adding more animation.

## Streamlit alignment

- Keep `.streamlit/config.toml` in sync with this doc for global theme keys.
- Custom dashboard styling lives in the large `<style>` block in `app/main.py`; prefer extending existing `.gft-*` classes over one-off inline styles.

## Decisions log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-31 | Initial DESIGN.md from /design-consultation | Codifies existing Sora + Plex + amber terminal direction; adds JetBrains Mono target for tabular data |
| 2026-03-31 | Initially kept Streamlit `primaryColor` blue | Avoided accidental widget retheme until a deliberate global chrome pass existed |
| 2026-04-02 | Research · factors attribution strip | Horizontal bars for factor contribution + residual; presets 21/63 TD + MTD + custom; same research band classes; no mockups (plan-design-review, tool off) |
| 2026-04-19 | Market Analysis · Tabbed Terminal | Formally adopt high-density tabbed workspaces for data-heavy pages instead of vertical document scrolls. Context stays locked in global headers. |
| 2026-04-23 | Native Streamlit chrome amber retheme | The retheme is now intentional: amber owns active orientation, focus rings, and true primary actions; slate carries default controls. |
