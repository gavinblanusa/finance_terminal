# CLAUDE.md

Project pointers: **README.md** (setup, structure, API), **AGENTS.md** (Cursor Cloud / VM notes), **docs/ARCHITECTURE.md** (page-to-module map).

## Design system

Always read **DESIGN.md** before making visual or UI decisions. Font roles, color tokens, spacing, and layout rules live there.

- Do not introduce new primary fonts or a second accent color without updating DESIGN.md.
- Prefer extending existing `.gft-*` classes in `app/main.py` over scattered inline styles.
- In QA or review, flag UI that contradicts DESIGN.md (for example generic purple gradients, emoji in global chrome, non-tabular number columns on tape-style tables).

When adding features: keep copy utility-first (status, action, caveat). Cards only when the card is the interaction.
