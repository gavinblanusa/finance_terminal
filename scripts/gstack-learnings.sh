#!/usr/bin/env bash
# Run gstack learnings tools from repo root so gstack-slug matches this project.
# Usage:
#   scripts/gstack-learnings.sh recent [N]
#   scripts/gstack-learnings.sh search <query> [N]
#   scripts/gstack-learnings.sh log '<json>'
# Env: GSTACK_BIN (default ~/.claude/skills/gstack/bin)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GSTACK_BIN="${GSTACK_BIN:-$HOME/.claude/skills/gstack/bin}"
cd "$ROOT"
cmd="${1:-}"
shift || true
case "$cmd" in
  recent)
    limit="${1:-20}"
    "$GSTACK_BIN/gstack-learnings-search" --limit "$limit"
    ;;
  search)
    q="${1:-}"
    limit="${2:-20}"
    "$GSTACK_BIN/gstack-learnings-search" --query "$q" --limit "$limit"
    ;;
  log)
    json="${1:?pass a single JSON object string}"
    "$GSTACK_BIN/gstack-learnings-log" "$json"
    ;;
  *)
    echo "Usage: $0 recent [limit] | search <query> [limit] | log '<json>'" >&2
    exit 1
    ;;
esac
