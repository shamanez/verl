#!/usr/bin/env bash
# Stop — autosave uncommitted work under research/ (crash durability).
# NEVER blocks session stop: every failure path exits 0 with a stderr note.
# Worktree-aware: commits on whatever branch THIS session's checkout is on,
# which under the per-issue-worktree convention is that issue's own branch —
# parallel sessions therefore cannot cross-contaminate each other's branches.
set -uo pipefail

PAYLOAD="$(cat)"
SID=$(echo "$PAYLOAD" | jq -r '.session_id // "unknown"' 2>/dev/null || echo unknown)

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Only run inside a checkout that actually has a research/ subtree (main
# checkout or a worktree of the fork). Bail quietly otherwise.
TOP=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[[ -d "$TOP/research" ]] || exit 0

BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)

if [[ -n "$(git status --porcelain -- :/research 2>/dev/null)" ]]; then
  git add -- :/research 2>/dev/null
  if ! git -c core.hooksPath=/dev/null commit -q \
       -m "[autosave] research session $SID stop ($BR)" 2>/dev/null; then
    # index.lock contention with a concurrent session, etc. — log, never block.
    echo "commit-on-stop: autosave commit failed on $BR (non-blocking; will retry next Stop)." >&2
  fi
fi
exit 0
