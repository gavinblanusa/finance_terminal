#!/usr/bin/env python3
"""
Ensure docs/OPENBB_COVERAGE.md still lists every multi-provider chain from
app/openbb_provider_registry.py (drift guard).

Run from repo root:
  python scripts/verify_openbb_coverage_doc.py
CI: invoked by .github/workflows/ci.yml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from openbb_provider_registry import OPENBB_PROVIDER_CHAINS, chain_arrow  # noqa: E402


def main() -> int:
    path = ROOT / "docs" / "OPENBB_COVERAGE.md"
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    missing: list[tuple[str, str]] = []
    for dataset_id, providers in OPENBB_PROVIDER_CHAINS.items():
        if len(providers) < 2:
            continue
        needle = chain_arrow(providers)
        if needle not in text:
            missing.append((dataset_id, needle))
    if missing:
        print("OPENBB_COVERAGE.md is missing these provider chains from openbb_provider_registry.py:", file=sys.stderr)
        for ds, needle in missing:
            print(f"  {ds}: {needle}", file=sys.stderr)
        return 1
    print("OK: all multi-provider chains from registry appear in OPENBB_COVERAGE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
