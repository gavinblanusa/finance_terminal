# CLAUDE.md

Project pointers: **README.md** (setup, structure, API), **AGENTS.md** (Cursor Cloud / VM notes, **gstack `/learn` paths**, CI/pytest), **docs/ARCHITECTURE.md** (page-to-module map), **docs/OPENBB_COVERAGE.md** (OpenBB inventory + provider order), **docs/DATA_LAYER_REFERENCE.md** (data contracts + optional **`GFT_MARKET_WAREHOUSE_*`** local OHLCV + **`.macro_cache/`** for Macro Dashboard FRED files), **CHANGELOG.md** / **VERSION** (releases).

## Design system

Always read **DESIGN.md** before making visual or UI decisions. Font roles, color tokens, spacing, and layout rules live there.

- Do not introduce new primary fonts or a second accent color without updating DESIGN.md.
- Prefer extending existing `.gft-*` classes in `app/main.py` over scattered inline styles.
- In QA or review, flag UI that contradicts DESIGN.md (for example generic purple gradients, emoji in global chrome, non-tabular number columns on tape-style tables).

When adding features: keep copy utility-first (status, action, caveat). Cards only when the card is the interaction.

## Skill routing

When the user's request matches an available gstack skill, follow that workflow (for example `/gstack-qa`, `/gstack-ship`, `/gstack-investigate`) as the primary path instead of ad-hoc answers when they explicitly invoke a skill.

Key routing (invoke the matching skill or equivalent when the user asks):

- Product ideas, "is this worth building", brainstorming → office-hours
- Bugs, errors, stack traces, "why is this broken" → investigate
- Ship, deploy, push, create PR → ship
- QA, test the site, find bugs → qa (or `/gstack-qa`)
- Code review, check my diff → review
- Update docs after shipping → document-release
- Weekly retro → retro
- Design system, brand → design-consultation
- Visual audit, design polish → design-review
- Architecture / plan review → plan-eng-review
- Project learnings, `/learn`, "what did we learn" → learn (see **AGENTS.md** § Gstack project learnings)
