#!/usr/bin/env bash
# SessionStart — heartbeat to ~/.claude-events.log plus current $/hr burn from runs.jsonl.
# Gives the operator immediate visibility on what's spending money the moment they open a session.
set -euo pipefail

PAYLOAD="$(cat)"
SID=$(echo "$PAYLOAD" | jq -r '.session_id // "unknown"')

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"

BURN_DPH="0.00"
RUNNING_COUNT=0
if [[ -f "$LEDGER" ]]; then
  # Sum dph across rows whose status is RUNNING. Each row is one JSON object per line.
  # jq -rs: raw output (so @tsv produces a real tab, not the JSON string "0\t0").
  TSV=$(jq -rs '
    map(select(.status == "RUNNING"))
    | [(map(.dph // 0) | add // 0), length]
    | @tsv
  ' "$LEDGER" 2>/dev/null) || TSV=$'0.00\t0'
  [[ -z "$TSV" ]] && TSV=$'0.00\t0'
  IFS=$'\t' read -r BURN_DPH RUNNING_COUNT <<<"$TSV"
fi

echo "[$(date -Iseconds)] [$SID] session started in $PROJECT_DIR · running=$RUNNING_COUNT · burn=\$$BURN_DPH/hr" \
  >> "$HOME/.claude-events.log"
exit 0
