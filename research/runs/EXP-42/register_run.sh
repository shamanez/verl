#!/usr/bin/env bash
# Append a teardown-SAFE runs.jsonl row for an EXP-42 cell (LAPTOP-side).
#
# The row carries NO instance_id in .handles[] (empty array) so the teardown Stop
# hook (teardown-finished-runs.sh) iterates nothing and can NEVER destroy the
# operator's BYO box — honouring the operator rule "do not tear down my box, ask
# first". The box endpoint is recorded in instance_ref (audit only; the hook does
# not read it). status defaults to RUNNING; flip to COMPLETE when the cell ends.
set -euo pipefail
CELL="${1:?usage: register_run.sh run1|run2|run3 [RUNNING|COMPLETE]}"
STATUS="${2:-RUNNING}"
LEDGER="$(cd "$(dirname "$0")/../../.claude/state" && pwd)/runs.jsonl"
ID="EXP-42-${CELL}"
# If a row for this id already exists, update its status in place; else append.
TMP="$(mktemp)"
if [[ -f "$LEDGER" ]] && jq -e --arg id "$ID" 'select(.id==$id)' "$LEDGER" >/dev/null 2>&1; then
  jq -c --arg id "$ID" --arg st "$STATUS" --arg t "$(date -Iseconds)" \
    'if .id==$id then . + {status:$st, updated_at:$t} else . end' "$LEDGER" > "$TMP" && mv "$TMP" "$LEDGER"
  echo "updated $ID -> $STATUS"
else
  jq -nc --arg id "$ID" --arg cell "$CELL" --arg t "$(date -Iseconds)" --argjson ts "$(date +%s)" --arg st "$STATUS" \
    '{id:$id, cell:$cell, handles:[], instance_ref:"byo-104.202.252.41(operator-managed)",
      external:true, manual_teardown:true, vast_account:"private", per_node_gpus:4, total_gpus:4,
      started_at:$t, started_at_epoch:$ts, status:$st,
      note:"BYO box; empty handles => teardown hook no-op; operator owns teardown"}' >> "$LEDGER"
  echo "registered $ID ($STATUS)"
fi
