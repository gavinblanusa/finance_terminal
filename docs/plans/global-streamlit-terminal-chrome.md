<!-- /autoplan restore point: /Users/gavinblanusa/.gstack/projects/gavinblanusa-finance_terminal/main-autoplan-restore-20260423-213420.md -->

# Plan: Global Streamlit Terminal Chrome

## Product Goal

Make Gavin Financial Terminal feel like one intentional financial workspace instead of a custom dashboard embedded inside default Streamlit chrome.

This plan styles Streamlit native controls globally so tabs, sidebar radio navigation, buttons, inputs, metrics, and dataframe chrome inherit the existing GFT terminal language defined in `DESIGN.md`.

## Why Now

The current UI breaks trust in the first 30 seconds because custom terminal sections sit next to plain Streamlit controls. The user feels the app oscillate between "personal financial terminal" and "prototype notebook."

Success after this pass:

- The app feels production-grade on first open, especially in the sidebar, tabs, and primary actions.
- Navigation and action hierarchy are easier to scan because active states, focus states, and primary controls share one visual system.
- Obvious default Streamlit blue/chrome is removed from core workflows by explicitly retheming native chrome to the GFT amber/slate system.

## Premises

- The app is a data-dense Streamlit terminal, not a marketing site.
- The current design system is good enough to extend. This work should not create a second design language.
- The main visual failure is inconsistent native Streamlit chrome, not bad page content.
- Streamlit DOM selectors are somewhat brittle, so the implementation must be compact, documented, and easy to roll back.
- The first pass should improve global consistency without redesigning page-specific workflows.

## UI Scope

In scope:

- Global design tokens in the existing `app/main.py` CSS block.
- Streamlit tabs from `st.tabs`.
- Sidebar radio navigation from `st.sidebar.radio`.
- Buttons and download buttons.
- Text inputs, selectboxes, checkboxes, radios, sliders, number/date inputs, and text areas where covered by shared Streamlit input wrappers.
- Metrics from `st.metric`.
- Dataframe/table outer chrome: container border, radius, background, and toolbar tone where stable selectors expose it.
- Focus-visible, hover, active, disabled, error, and loading-adjacent visual states.

Out of scope:

- Plotly/chart theming.
- Replacing Streamlit widgets with custom components.
- Page-specific layout refactors.
- Market Analysis page-specific content/charts. Global native controls should still converge to GFT chrome on that page.
- Fragile dataframe internals such as row cells, virtualized canvas internals, and generated grid class names unless stable selectors are confirmed during implementation.
- Mobile navigation redesign beyond making current sidebar controls usable.
- Light mode.
- Page favicon/app icon, unless a non-emoji asset is added separately.

## Existing Design System To Reuse

- Fonts: Sora for display, IBM Plex Sans for UI/body, JetBrains Mono for data.
- Colors: `#0E1117`, `#262730`, `#F8FAFC`, `#94A3B8`, `#E8A838`, `#2DD4BF`, `#A5B4FC`, `#4ADE80`, `#F87171`.
- Radius: 4-8px, with small radii preferred for app chrome.
- Layout mood: industrial utilitarian, warm terminal accent, dense but readable.
- Existing `.gft-*` naming and dashboard section treatment in `app/main.py`.

## Implementation Approach

1. Add a documented `:root` token section inside the existing global CSS block in `app/main.py`.
2. Add a single `/* Global Streamlit terminal chrome */` CSS section below the current GFT component classes.
3. Prefer stable Streamlit `data-testid` selectors when targeting native widgets.
4. Keep each selector group short and commented by component.
5. Avoid styling generated class names unless there is no reasonable alternative.
6. Use `!important` sparingly, only when Streamlit inline/runtime styles require it.
7. Do not change `page_icon` in this pass. `DESIGN.md` avoids emoji in global chrome; a real favicon should be a separate asset decision.
8. Update `.streamlit/config.toml` and `DESIGN.md` so theme config, docs, and CSS agree on the amber native chrome retheme.
9. Remove or replace legacy blue Streamlit overrides before adding the new chrome layer.
10. Remove the DCF page's unscoped global metric override before styling metrics globally.

Approved selector patterns:

| Surface | Selector strategy | Fallback |
|---|---|---|
| Tabs | `[data-testid="stTabs"] [role="tab"]`, `[aria-selected="true"]` | If ARIA changes, default tab styling remains usable |
| Sidebar radio | Sidebar-scoped `[data-testid="stRadio"] label:has(input:checked)` | Without `:has`, hover/focus still applies; no brittle generated class fallback |
| Buttons | `[data-testid="stButton"] button`, `[data-testid="stDownloadButton"] button`, `[data-testid="stFormSubmitButton"] button` | Leave untouched if button DOM changes |
| Inputs | Widget `data-testid` wrappers plus direct `input`, `textarea`, BaseWeb select child surfaces | Preserve native affordances if wrapper shape changes |
| Metrics | `[data-testid="stMetric"]`, `[data-testid="stMetricValue"]`, delta labels | Skip page-local metric exceptions |
| Dataframes | `[data-testid="stDataFrame"]` and `[data-testid="stTable"]` outer wrappers only | Do not style virtualized internals |

## Alternatives Considered

| Alternative | Decision | Why |
|---|---|---|
| Streamlit theme config only | Rejected | Theme config cannot express active tab rails, sidebar row hierarchy, or focus treatment. |
| Page-by-page redesign | Deferred | Higher product upside, but too large for this pass and risks mixing workflow redesign with chrome cleanup. |
| Custom component shell | Rejected | Too much maintenance for a single-user Streamlit terminal. |
| Chart/table-first redesign | Deferred | Important for a later visual QA pass, but native chrome inconsistency appears across every workflow today. |
| Global native chrome layer | Accepted | Cheapest credibility upgrade with small code blast radius. |
| Full migration off Streamlit | Rejected | Not justified by this visual issue. |

## Phasing

Phase 1, implement now:

- Tokens, tabs, sidebar radio, buttons, inputs, focus-visible treatment, and dataframe outer chrome.
- Metrics only where stable Streamlit metric wrappers can be styled without brittle generated classes.

Phase 2, defer until visual QA proves Phase 1 is stable:

- Deeper dataframe internals.
- Plotly/chart theme convergence.
- Page-specific Market Analysis mini-theme cleanup.
- Mobile navigation redesign.
- Non-emoji favicon/app icon asset.

## Visual Hierarchy

Global chrome should not make every widget shout.

Priority order:

1. App identity and active sidebar page: strongest persistent orientation. Uses amber left rail plus brighter text.
2. Current page title and section headers: existing Streamlit heading plus `.gft-*` section vocabulary.
3. Active tab: secondary orientation within a page. Uses amber underline, not a large filled pill.
4. Primary action buttons: only Streamlit `type="primary"` actions get strong amber treatment.
5. Default buttons and inputs: quiet slate surfaces with clear focus and hover.
6. Dataframe outer frame: quiet containment. It should feel aligned, not decorated.

Action inventory:

| Action type | Examples | Treatment |
|---|---|---|
| Primary | `Import Trades`, `Follow IPO`, agent generation buttons | Amber fill or amber border with high contrast |
| Secondary | `Refresh`, `Clear cache`, `Save`, `Remove`, `Compare` | Slate surface, muted border, amber hover/focus only |
| Destructive | `Remove` actions | Slate base with red hover/focus accent where reachable |
| Download/export | `Download JSON snapshot` | Secondary treatment unless `type="primary"` is set |

## Color Decision

This pass deliberately rethemes native Streamlit chrome from generic blue to GFT amber/slate.

That reverses the older DESIGN.md decision that kept `primaryColor = #1F77B4` to avoid accidental full retheme. The retheme is now intentional and limited:

- Amber is for active orientation, focus rings, and true primary actions.
- Slate surfaces carry default controls.
- Semantic green/red remain reserved for market direction and destructive/error states.
- No glow, no decorative gradients, no fake terminal effects.

## Component Specifications

### Tabs

User sees:

- Tab list as a compact segmented terminal rail.
- Active tab uses amber text and a thin amber underline or inset border.
- Inactive tabs use muted text with subtle hover surface.
- Focus-visible state has a clear amber outline.

States:

| State | Visual |
|---|---|
| Default | transparent/dark surface, muted text |
| Hover | `rgba(232,168,56,0.08)` background, text brightens |
| Active | amber text, amber bottom border, subtle dark surface |
| Focus | 2px amber outline, 2px offset |
| Disabled | muted opacity, no hover emphasis |

### Sidebar Radio Navigation

User sees:

- Sidebar title remains clear and understated.
- Nav choices look like terminal workspace rows, not plain radio pills.
- Active page has amber left rail and brighter text.
- Hover state previews selection without moving layout.

States:

| State | Visual |
|---|---|
| Default | muted text, transparent/dark row |
| Hover | subtle surface, text primary |
| Active | amber rail, amber-tinted surface, primary text |
| Focus | visible amber outline |

### Buttons

User sees:

- Primary actions are amber-on-dark or dark-on-amber only when Streamlit marks them as `type="primary"`.
- Secondary buttons are dark surface with thin border.
- Buttons feel compact and utilitarian, not bubbly.

States:

| State | Visual |
|---|---|
| Default | 6px radius, IBM Plex Sans, clear border |
| Hover | brighter border/background, no layout shift |
| Active | slight inset/darker background |
| Focus | 2px amber outline |
| Disabled | reduced opacity and muted border |

### Inputs

User sees:

- Inputs blend into dark terminal surfaces with clear border and focus ring.
- Labels are Plex/Sora-like, compact, readable.
- Placeholder text is muted, not low-contrast invisible.

States:

| State | Visual |
|---|---|
| Default | `#0E1117`/`#151922` surface, slate border |
| Hover | border brightens slightly |
| Focus | amber border + soft amber ring |
| Error | red border, red-tinted ring |
| Disabled | muted opacity |

Checkboxes/radios/selectboxes must keep their native affordance recognizable. The styling can recolor borders/focus, but it must not hide the checkmark, selected dot, dropdown arrow, or label.

### Metrics

User sees:

- Metric blocks read like instrument panels.
- Labels are muted and compact.
- Values use JetBrains Mono/tabular numerals.
- Positive/negative delta colors stay semantic.

States:

| State | Visual |
|---|---|
| Default | subtle surface/border if Streamlit container exposes metric block |
| Positive | `#4ADE80` delta |
| Negative | `#F87171` delta |
| Neutral | muted delta |

### Dataframes

User sees:

- Tables feel like tape/screens, not default embedded widgets.
- Outer frame aligns with the terminal surface system.
- Toolbar and fullscreen affordances are toned down.
- Header, body, row hover, and selection internals are only styled if stable selectors expose them without generated class coupling.

States:

| State | Visual |
|---|---|
| Container | thin slate border, 6px radius, dark background |
| Header | opportunistic darker surface if stable selector exists |
| Row hover | opportunistic amber/slate tint if stable selector exists |
| Selection | opportunistic teal/amber tint if stable selector exists |
| Scrollbar | narrow dark scrollbar where supported |

## Accessibility Requirements

- Preserve keyboard focus for all targeted widgets.
- Minimum visible focus ring: 2px solid amber or teal with enough contrast.
- Do not hide native labels.
- Do not remove native affordances for checkboxes, radios, selectboxes, sliders, date inputs, or file-like controls.
- Do not rely on color alone for active state. Active nav/tabs also get rail/underline/border.
- Keep text contrast at least WCAG AA for normal text where practical.
- Check contrast for amber-on-dark, muted text on slate, and focus ring against background during QA.
- Keep click targets near 44px for primary navigation and key controls where Streamlit structure allows.

## Responsive Requirements

- On narrow screens, tabs may scroll horizontally but must not wrap into unreadable multi-line stacks.
- Sidebar nav rows must remain legible with long labels.
- Button text must not overflow its control.
- Dataframes must preserve existing horizontal scrolling behavior.
- Verify at 1440px desktop, 1024px tablet/narrow desktop, and 390px mobile width.
- On 390px, long tab/nav labels may truncate only if the active state remains understandable.

## Verification Plan

Manual visual pass:

- Dashboard: metrics, refresh/download buttons, dataframes.
- Portfolio & Taxes: tabs, form inputs, import button, tax lots dataframe.
- Market Analysis: ticker input, clear cache button, tabs, selectboxes, checkboxes, metrics, dataframes.
- IPO Vintage Tracker: buttons, inputs, metrics, dataframes.
- Partnerships: checkboxes, selectbox, buttons, dataframe.
- 13F Holdings: tabs, selectboxes, buttons, dataframes.
- Macro Dashboard: buttons, metrics/containers.

Automated checks:

- `ruff check app/`
- `pytest tests/`

Design QA checks:

- No obvious default blue Streamlit hover on buttons unless intentionally retained.
- Active tab and active sidebar item are visually obvious.
- Keyboard tabbing still shows focus.
- Dataframes remain usable and scrollable.
- No text clipping inside buttons/tabs on desktop widths.
- At 390px, tabs scroll or compress without clipping labels.
- No page loses an existing action affordance.
- If a selector does not apply cleanly, the implementation leaves that sub-surface default rather than adding brittle generated-class CSS.
- Keyboard traversal visibly reaches sidebar nav, tabs, inputs, buttons, and dataframe toolbar controls.
- Reduced-motion preference remains respected; no new motion is added beyond existing section reveal.

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Streamlit DOM changes break selectors | Styles silently regress after upgrade | Keep CSS grouped/commented and avoid generated class names |
| Over-styling harms accessibility | Keyboard users lose orientation | Explicit focus-visible styles and manual keyboard QA |
| Global CSS creates unintended page-specific effects | Some widgets look cramped or miscolored | Use restrained selectors and verify all pages |
| Dataframe internals resist CSS | Some table chrome stays default | Accept partial improvement; avoid fragile hacks |

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | Intake | Scope global native Streamlit chrome, not page layout refactors | Mechanical | Explicit over clever | The problem is inconsistent native controls; layout refactors would increase blast radius | Redesigning each page now |
| 2 | CEO | Narrow dataframe work to outer chrome in Phase 1 | Taste | Boil lakes, but avoid oceans | Both CEO voices flagged dataframe internals as brittle and lower leverage; outer chrome still improves consistency | Styling virtualized/generated dataframe internals now |
| 3 | CEO | Let global native controls win on Market Analysis while leaving page-specific content alone | Mechanical | DRY | Native controls should not have a separate page language; charts/content can be handled later | Preserving default/neon control styling on Market Analysis |
| 4 | Design | Explicitly retheme native chrome from blue to GFT amber/slate | Taste | Explicit over clever | The user asked for native controls to inherit the terminal language; doing so must update the old DESIGN.md assumption rather than conflict with it silently | Keeping blue primary widgets inside an amber terminal chrome pass |
| 5 | Design | Remove page icon from v1 | Mechanical | Scope restraint | `DESIGN.md` avoids emoji in global chrome; a favicon needs a real asset decision | Adding emoji favicon as cosmetic scope creep |
| 6 | Eng | Update theme config and DESIGN.md with the amber retheme | Mechanical | DRY | Theme config, docs, and CSS must not fight each other | CSS-only retheme over blue config |
| 7 | Eng | Replace existing blue button hover and remove unscoped DCF metric override | Mechanical | Explicit over clever | Existing overrides would defeat the new global layer | Stacking more specific CSS over old rules |
| 8 | Eng | Limit dataframe v1 to outer wrappers only | Mechanical | Pragmatic | Virtualized dataframe internals are high brittleness for low payoff | Styling generated grid internals |

## CEO Review Summary

Initial score: 6/10. Final score after plan edits: 8/10.

Premises reviewed:

- Valid: the app is a data-dense Streamlit terminal and should not look like a generic Streamlit demo.
- Valid: global native chrome is a cheap credibility upgrade with low code blast radius.
- Challenged: "main visual failure" was too broad. The plan now frames this as one visible trust blocker, not the whole terminal experience.

Dream state delta:

```text
CURRENT
  Custom GFT dashboard sections sit beside default Streamlit controls.
    |
THIS PLAN
  Core native controls inherit GFT terminal chrome with stable, documented CSS.
    |
12-MONTH IDEAL
  Full terminal shell: persistent context, unified charts/tables, cross-page entity navigation, visual regression baselines.
```

CEO dual voices consensus:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| Premises valid? | Mostly | Partly | Confirmed with edits |
| Right problem? | Yes, if framed as credibility layer | Not standalone product goal | Confirmed with reframing |
| Scope calibration? | Too broad around dataframes | Too broad around low-value internals | Confirmed |
| Alternatives explored? | Underdeveloped | Underdeveloped | Confirmed |
| Market/competitive risk? | Low | Low | Confirmed |
| 6-month trajectory? | Risk is brittle CSS | Risk is cosmetic-only work | Confirmed |

NOT in scope from CEO review:

- Full terminal workflow redesign, because this pass should stay low blast radius.
- Deep dataframe internals, because selector brittleness can exceed product value.
- Chart theming, because it deserves a dedicated visual QA/design pass.

What already exists:

- `DESIGN.md` defines tokens, typography, radius, and density.
- `app/main.py` already owns the global CSS injection point.
- Existing `.gft-*` classes provide the local naming pattern and visual language.

Failure modes registry:

| Failure | Severity | Mitigation |
|---|---|---|
| CSS pass becomes a broad Streamlit skin | High | Phase 1 scope limits and out-of-scope list |
| Cosmetic work hides workflow debt | Medium | Frame as credibility layer, defer workflow work explicitly |
| Market Analysis keeps divergent native chrome | Medium | Global native controls win on all pages |

## Eng Review Summary

Initial score: 6.5/10. Final score after plan edits: 9/10.

Architecture diagram:

```text
.streamlit/config.toml
  primaryColor/token baseline
        |
        v
app/main.py global <style>
  :root GFT tokens
  existing .gft-* components
  global Streamlit chrome selectors
        |
        v
Streamlit native widgets
  tabs, sidebar radio, buttons, inputs, metrics, dataframe outer wrappers
```

Eng dual voices consensus:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| Architecture sound? | Yes after config/docs alignment | Yes after config/docs alignment | Confirmed |
| Test coverage sufficient? | Needs visual QA | Needs visual QA | Confirmed |
| Performance risks addressed? | CSS only, low risk | CSS only, low risk | Confirmed |
| Security threats covered? | No new attack surface | No new attack surface | Confirmed |
| Error paths handled? | Need fallback selector behavior | Need fallback selector behavior | Confirmed |
| Deployment risk manageable? | Yes if old overrides removed | Yes if old overrides removed | Confirmed |

Test diagram:

| Flow/surface | Risk | Coverage |
|---|---|---|
| Sidebar nav active/hover/focus | Active page not obvious or focus hidden | Browser QA at desktop/tablet/mobile widths |
| Tabs active/scroll/focus | Labels clip or active tab unclear | Browser QA on Portfolio, Market Analysis, 13F |
| Primary/secondary buttons | Every action appears primary | Visual QA against action inventory |
| Inputs/selects/checks | Native affordance hidden | Keyboard/focus QA |
| Metrics | Page-local override fights global styling | Remove DCF override, visual QA |
| Dataframe outer wrapper | Scroll/fullscreen broken | Manual scroll/fullscreen check |

Failure modes from eng review:

| Failure | Severity | Mitigation |
|---|---|---|
| Theme config stays blue | High | Update `.streamlit/config.toml` primary color |
| Legacy blue hover leaks | High | Replace old `.stButton>button:hover` rule |
| DCF metric override wins globally | High | Remove unscoped `[data-testid="stMetricValue"]` style block |
| Sidebar active styling depends on `:has` | Medium | Use it only as progressive enhancement; preserve usable fallback |
| Custom Market/IPO page cards remain blue/purple | Medium | Accepted Phase 2 design debt; Phase 1 targets native chrome only |

Test plan artifact: `~/.gstack/projects/gavinblanusa-finance_terminal/main-test-plan-global-streamlit-terminal-chrome.md`.


## Design Review Summary

Initial score: 5/10. Final score after plan edits: 8.5/10.

Classifier: app UI. Apply calm surface hierarchy, dense but readable controls, utility language, minimal chrome.

Design litmus scorecard:

| Check | Claude | Codex | Consensus |
|---|---|---|---|
| Brand/product unmistakable in first screen? | Needs active chrome hierarchy | Needs hierarchy | Confirmed with hierarchy section |
| One strong visual anchor? | Sidebar active page | Sidebar active page | Confirmed |
| Scannable by headings/chrome? | Needs priority order | Needs action inventory | Confirmed with action inventory |
| Each section has one job? | Mostly | Mostly | Confirmed |
| Cards actually necessary? | N/A | N/A | No issue |
| Motion improves hierarchy? | No new motion | No new motion | Confirmed |
| Premium without decorative shadows? | Yes if no glow/gradients | Yes if no glow/gradients | Confirmed |

Pass scores:

| Pass | Before | After | Fix |
|---|---:|---:|---|
| Information Architecture | 6 | 9 | Added hierarchy and action inventory |
| Interaction States | 7 | 8.5 | Added native affordance and fallback rules |
| User Journey | 6 | 8 | Added why-now and success outcomes |
| AI Slop Risk | 7 | 9 | Added no glow/gradients/fake-terminal constraints |
| Design System Alignment | 5 | 9 | Added explicit amber/slate retheme decision |
| Responsive & Accessibility | 6 | 8.5 | Added viewport, contrast, keyboard traversal checks |
| Unresolved Decisions | 5 | 8 | Deferred favicon, deep dataframe internals, chart theming |

NOT in scope from design review:

- Emoji favicon or page icon.
- Making every button visually primary.
- Hiding native affordances for checkboxes/radios/selectboxes.
- Decorative terminal effects such as glow, scanlines, or heavy gradients.
