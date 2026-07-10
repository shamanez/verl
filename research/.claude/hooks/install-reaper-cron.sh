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

# PATH is the load-bearing fix (#audit 2026-07-10): macOS cron runs with a bare
# PATH=/usr/bin:/bin, but `vastai` lives in ~/.local/bin. Without this, the hook
# scans the ledger fine (jq/perl are in /usr/bin) yet every `vastai destroy`
# exits 127 -> classified FAILED -> the row never flips TORN_DOWN and the box
# BILLS FOREVER — silently, since /tmp/teardown.cron.log looks healthy. Resolve
# vastai's real dir at install time and bake an absolute PATH into the cron line
# ($HOME/$(...) expand here, at assignment, NOT in crontab which never expands).
VASTAI_DIR="$(dirname "$(command -v vastai 2>/dev/null || echo "$HOME/.local/bin/vastai")")"
REAPER_PATH="$VASTAI_DIR:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
LINE="17 * * * * PATH=$REAPER_PATH CLAUDE_PROJECT_DIR=$RESEARCH_DIR bash $HOOK >> /tmp/teardown.cron.log 2>&1 $MARK"

[[ -f "$HOOK" ]] || { echo "install-reaper-cron: $HOOK not found" >&2; exit 1; }

current="$(crontab -l 2>/dev/null || true)"

case "${1:-install}" in
  --status)
    installed_line=$(grep -F "$MARK" <<<"$current" || true)
    if [[ -z "$installed_line" ]]; then
      echo "reaper cron: NOT installed"
    else
      echo "$installed_line"
      # Drift check (#63 B3): the cron must point at THIS checkout, else it
      # reaps the wrong ledger and every box provisioned here is unguarded.
      if ! grep -qF "CLAUDE_PROJECT_DIR=$RESEARCH_DIR " <<<"$installed_line"; then
        echo "reaper cron: WARN — installed path is NOT this checkout ($RESEARCH_DIR)." >&2
        echo "reaper cron: WARN — reinstall from here: bash $0" >&2
      fi
    fi
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
