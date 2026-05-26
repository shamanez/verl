#!/usr/bin/env bash
# PreToolUse — universal pause. Touch ~/.claude-kill-switch to halt every agent on this machine.
set -euo pipefail

if [[ -f "$HOME/.claude-kill-switch" ]]; then
  echo "kill-switch active. rm ~/.claude-kill-switch to resume." >&2
  exit 2
fi
exit 0
