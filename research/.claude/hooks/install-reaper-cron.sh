#!/usr/bin/env bash
# install-reaper-cron.sh — install/remove the SESSION-INDEPENDENT money backstop.
# The Stop-hook reaper (teardown-finished-runs.sh) only fires while a Claude
# session is open; if every window is closed with a box live, nothing reaps it.
# This installs an hourly crontab line that runs the same hook headless — the
# hook is idempotent, lock-aware, timeout-bounded, and always exits 0, so it is
# safe alongside live sessions.
#
#   bash .claude/hooks/install-reaper-cron.sh            # install (idempotent)
#   bash .claude/hooks/install-reaper-cron.sh --remove   # uninstall
#   bash .claude/hooks/install-reaper-cron.sh --status   # show the current line
#
# macOS CAVEAT (observed 2026-07-07): the repo lives under ~/Documents, which is
# TCC-protected — cron gets "Operation not permitted" until the OPERATOR grants
# Full Disk Access to /usr/sbin/cron (System Settings → Privacy & Security →
# Full Disk Access → add /usr/sbin/cron). Until then the cron line is installed
# but inert, and the Stop-hook reaper (session-scoped) is the only backstop.
# Verify after granting:  tail /tmp/teardown.cron.log  (should stop erroring).
set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
HOOK="$RESEARCH_DIR/.claude/hooks/teardown-finished-runs.sh"
MARK="# verl-research-reaper"   # marker comment — install/remove key off this, never off the schedule
LINE="17 * * * * CLAUDE_PROJECT_DIR=$RESEARCH_DIR bash $HOOK >> /tmp/teardown.cron.log 2>&1 $MARK"

[[ -f "$HOOK" ]] || { echo "install-reaper-cron: $HOOK not found" >&2; exit 1; }

current="$(crontab -l 2>/dev/null || true)"

case "${1:-install}" in
  --status)
    grep -F "$MARK" <<<"$current" || echo "reaper cron: NOT installed"
    ;;
  --remove)
    if grep -qF "$MARK" <<<"$current"; then
      grep -vF "$MARK" <<<"$current" | crontab -
      echo "reaper cron: removed"
    else
      echo "reaper cron: nothing to remove"
    fi
    ;;
  install|*)
    if grep -qF "$MARK" <<<"$current"; then
      # replace (path may have changed), keep exactly one line
      { grep -vF "$MARK" <<<"$current"; echo "$LINE"; } | crontab -
      echo "reaper cron: refreshed → $LINE"
    else
      { [[ -n "$current" ]] && echo "$current"; echo "$LINE"; } | crontab -
      echo "reaper cron: installed → $LINE"
    fi
    ;;
esac
