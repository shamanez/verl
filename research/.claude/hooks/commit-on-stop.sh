#!/usr/bin/env bash
# Stop — autosave uncommitted work under research/.
# Stages research/** only (never anything outside the harness subtree).
# Skips pre-commit because verl's pre-commit is configured for verl/, not for the research harness.
# If git itself fails, exit 2 surfaces back to Claude.
set -euo pipefail

PAYLOAD="$(cat)"
SID=$(echo "$PAYLOAD" | jq -r '.session_id // "unknown"')

cd "${CLAUDE_PROJECT_DIR:-.}"

# Refuse to run outside a research/ project dir as a safety net.
case "$PWD" in
  */verl/research|*/verl/research/*) ;;
  *)
    echo "commit-on-stop: refused — CLAUDE_PROJECT_DIR=$PWD is not under verl/research." >&2
    exit 0
    ;;
esac

# Only stage research-subtree changes, even if other untracked files exist.
if [[ -n "$(git status --porcelain -- :/research 2>/dev/null)" ]]; then
  git add -- :/research
  if ! git -c core.hooksPath=/dev/null commit -m "[autosave] research session $SID stop"; then
    echo "autosave commit failed. Check git status." >&2
    exit 2
  fi
fi
exit 0
