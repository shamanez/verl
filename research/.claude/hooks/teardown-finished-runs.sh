#!/usr/bin/env bash
# Stop — tear down Vast.ai instances for finished experiments.
# Triggers (any one):
#   1. runs/<ID>/verdict.md exists (experiment reached a verdict)
#   2. No incoming.log heartbeat for > 30 min (training died on the box)
#   3. Elapsed GPU-hours exceeds plan.max_gpu_hr (budget cap)
# This is the budget backstop: even if every agent dies mid-experiment, the next Stop
# tears down stale Vast.ai rentals so we don't bleed money.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"

[[ -f "$LEDGER" ]] || exit 0

# vast-provision and vast-teardown read VAST_API_KEY from env; the vastai CLI reads the same var.
# Source ~/.config/verl-research/secrets.env at session start to populate it.

NOW=$(date +%s)
TEMP=$(mktemp)
TORN_ANY=0

# Process each ledger row.
# Triggers teardown for RUNNING and PROVISIONED rows. PROVISIONED rows came from
# experiment-runner's "register early" step — they have paid instances but no
# train.log yet, so we add a 4th trigger (in PROVISIONED for >15 min) to catch
# runner crashes between handle capture and successful launch.
while IFS= read -r row || [[ -n "$row" ]]; do
  [[ -z "$row" ]] && continue
  STATUS=$(echo "$row" | jq -r '.status // "UNKNOWN"')
  case "$STATUS" in
    RUNNING|PROVISIONED) ;;
    *) echo "$row" >> "$TEMP"; continue ;;
  esac

  ID=$(echo "$row" | jq -r '.id')

  REASON=""

  # 1. verdict.md present? (only meaningful for RUNNING; PROVISIONED never had a runner)
  if [[ "$STATUS" == "RUNNING" && -f "$PROJECT_DIR/runs/$ID/verdict.md" ]]; then
    REASON="verdict-written"
  fi

  # 2. Heartbeat stale > 30 min? (RUNNING only)
  if [[ "$STATUS" == "RUNNING" && -z "$REASON" ]]; then
    HEARTBEAT="$PROJECT_DIR/runs/$ID/metrics/incoming.log"
    if [[ -f "$HEARTBEAT" ]]; then
      LAST_MOD=$(stat -f %m "$HEARTBEAT" 2>/dev/null || stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
      if (( NOW - LAST_MOD > 1800 )); then
        REASON="no-heartbeat-30min"
      fi
    fi
  fi

  # 3. Budget exceeded?
  # Use total_gpus (sum across all handles) if the runner recorded it; otherwise
  # fall back to per_node_gpus * 1 handle. The single-handle approximation is
  # how the old code worked and undercounts multi-node jobs by gpu_count×.
  if [[ -z "$REASON" ]]; then
    STARTED=$(echo "$row" | jq -r '.started_at_epoch // 0')
    DPH=$(echo "$row" | jq -r '.dph // 0')
    MAX_GPU_HR=$(echo "$row" | jq -r '.max_gpu_hr // 0')
    TOTAL_GPUS=$(echo "$row" | jq -r '.total_gpus // (.per_node_gpus // 1)')
    if (( STARTED > 0 )) && awk -v d="$DPH" -v m="$MAX_GPU_HR" 'BEGIN { exit !(d > 0 && m > 0) }'; then
      ELAPSED_HR=$(awk -v n="$NOW" -v s="$STARTED" 'BEGIN { printf "%.4f", (n-s)/3600 }')
      ELAPSED_GPU_HR=$(awk -v e="$ELAPSED_HR" -v g="$TOTAL_GPUS" 'BEGIN { printf "%.4f", e*g }')
      EXCEEDED=$(awk -v eg="$ELAPSED_GPU_HR" -v mg="$MAX_GPU_HR" 'BEGIN { print (eg > mg) ? 1 : 0 }')
      if [[ "$EXCEEDED" == "1" ]]; then
        REASON="budget-exceeded"
      fi
    fi
  fi

  # 4. PROVISIONED stale > 15 min? (runner died before launch completed)
  if [[ "$STATUS" == "PROVISIONED" && -z "$REASON" ]]; then
    STARTED=$(echo "$row" | jq -r '.started_at_epoch // 0')
    if (( STARTED > 0 )) && (( NOW - STARTED > 900 )); then
      REASON="provisioned-but-never-launched"
    fi
  fi

  if [[ -n "$REASON" ]]; then
    # Tear down all instances in this row. Track success/failure so we don't
    # silently mark a row TORN_DOWN while the instance keeps running and billing.
    DESTROYED=0
    FAILED=0
    while IFS= read -r iid; do
      [[ -z "$iid" ]] && continue
      if vastai destroy instance "$iid" >>/tmp/teardown.err 2>&1; then
        DESTROYED=$((DESTROYED + 1))
      else
        FAILED=$((FAILED + 1))
      fi
    done < <(echo "$row" | jq -r '.handles[]? | .instance_id // empty')

    if (( FAILED == 0 && DESTROYED > 0 )); then
      # Clean teardown — flip to TORN_DOWN.
      echo "$row" \
        | jq -c --arg t "$(date -Iseconds)" --arg r "$REASON" \
          '. + {status: "TORN_DOWN", torn_down_at: $t, teardown_reason: $r}' \
        >> "$TEMP"
      echo "[$(date -Iseconds)] teardown EXP-$ID reason=$REASON destroyed=$DESTROYED" \
        >> "$PROJECT_DIR/PROGRESS.md"
      TORN_ANY=1
    elif (( DESTROYED == 0 && FAILED == 0 )); then
      # No handles at all — odd but treat as torn-down (nothing to destroy).
      echo "$row" \
        | jq -c --arg t "$(date -Iseconds)" --arg r "${REASON}-no-handles" \
          '. + {status: "TORN_DOWN", torn_down_at: $t, teardown_reason: $r}' \
        >> "$TEMP"
      echo "[$(date -Iseconds)] teardown EXP-$ID reason=$REASON no-handles" \
        >> "$PROJECT_DIR/PROGRESS.md"
      TORN_ANY=1
    else
      # Partial or full teardown failure — leave row RUNNING with an annotation
      # so the next Stop hook retries, and emit a loud PROGRESS marker so the
      # operator knows to intervene before money bleeds further.
      echo "$row" \
        | jq -c --arg t "$(date -Iseconds)" --arg r "$REASON" \
          --argjson d "$DESTROYED" --argjson f "$FAILED" \
          '. + {teardown_attempts: ((.teardown_attempts // 0) + 1),
                teardown_last_at: $t,
                teardown_last_reason: $r,
                teardown_partial_destroyed: $d,
                teardown_partial_failed: $f}' \
        >> "$TEMP"
      echo "[$(date -Iseconds)] TEARDOWN_FAILED EXP-$ID reason=$REASON destroyed=$DESTROYED failed=$FAILED — instance(s) may still be running, check vastai show instances" \
        >> "$PROJECT_DIR/PROGRESS.md"
      # Do NOT set TORN_ANY — partial failure shouldn't suppress the warning banner.
    fi
  else
    echo "$row" >> "$TEMP"
  fi
done < "$LEDGER"

mv "$TEMP" "$LEDGER"

if [[ $TORN_ANY -eq 1 ]]; then
  echo "teardown-finished-runs: tore down at least one stale Vast.ai handle. See PROGRESS.md." >&2
fi
exit 0
