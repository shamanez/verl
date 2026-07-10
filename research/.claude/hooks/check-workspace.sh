#!/usr/bin/env bash
# check-workspace.sh — session-orientation guard (#63 B4).
#
# Sessions must open in research/ of the CURRENT checkout: custom agents
# (.claude/agents/*), the protect-upstream hook, and the Stop-hook teardown
# backstop all register from there. A session opened anywhere else silently
# loses ALL of them (observed #63: no research-planner agent type, no write
# guard, no teardown backstop — and per-agent effort tiers degraded).
#
# Run at session start (manually or from a SessionStart hook):
#   bash .claude/hooks/check-workspace.sh
# WARN-only: prints what is inactive, never blocks.
set -uo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RESEARCH_DIR="$(cd "$HOOK_DIR/../.." && pwd)"                # <checkout>/research
CHECKOUT="$(cd "$RESEARCH_DIR/.." && pwd)"

CUR_ROOT="$(git -C "$PWD" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')"

if [[ "$PWD" == "$RESEARCH_DIR" || "$PWD" == "$RESEARCH_DIR"/* ]]; then
  echo "check-workspace: OK — session anchored to $RESEARCH_DIR"
  exit 0
fi

cat >&2 <<EOF
WARN [check-workspace]: session cwd is $PWD
                        expected      $RESEARCH_DIR (or below)
Inactive in a session opened here:
  - custom agents (research/.claude/agents/*): research-planner, experiment-runner,
    machine-monitor, ... fall back to generic agents and LOSE their model/effort tiers
  - protect-upstream PreToolUse hook (verl tree write guard)
  - Stop-hook teardown backstop (session-scoped reaper)
Current git root: ${CUR_ROOT:-<none>} — this checkout: $CHECKOUT
Fix: cd $RESEARCH_DIR && claude   (or proceed knowingly).
EOF
exit 0
