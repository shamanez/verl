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

# Teardown auths PER-ROW: a row's vast_account (team|private, default private)
# selects the key, so a team-account box is destroyed with the team key (else the
# personal key 404s and the box leaks). The shared resolver loads both keys; if it
# is somehow absent, fall back to the private VAST_API_KEY (the historical behaviour).
if [[ -f "$PROJECT_DIR/.claude/skills/_vast_account.sh" ]]; then
  # shellcheck disable=SC1090
  source "$PROJECT_DIR/.claude/skills/_vast_account.sh"
  vast_load_secrets
else
  # Degraded-mode fallback must STILL be account-aware, else a team row would be
  # torn down with the private key (silent no-op leak).
  vast_key_for() {
    case "${1:-private}" in
      team) printf '%s' "${VAST_API_KEY_TEAM:-${VAST_API_KEY:-}}" ;;
      *)    printf '%s' "${VAST_API_KEY:-}" ;;
    esac
  }
fi

NOW=$(date +%s)
TEMP=$(mktemp)
TORN_ANY=0

# Hard bound on every vastai CLI call — a hung API call must never hang session
# Stop or the foreground sweep (never-hang guarantee).
VAST_CLI_TIMEOUT=90

# Ledger spinlock (shared with skills/_lib.sh writers) so a concurrent session's
# append is never lost by this whole-file rewrite. Bounded 30s; on timeout we
# proceed WITHOUT the rewrite (skip ledger mutation, still attempt destroys next
# Stop) rather than hang.
LOCKDIR="$PROJECT_DIR/.claude/state/.runs.jsonl.lock"
LOCKED=0; n=0
until mkdir "$LOCKDIR" 2>/dev/null; do
  n=$((n+1)); (( n > 300 )) && break; sleep 0.1
done
(( n <= 300 )) && LOCKED=1
trap '[[ $LOCKED -eq 1 ]] && rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

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

  # 2. Heartbeat stale? (RUNNING only)
  #    - If incoming.log EXISTS: stale > 30 min => dead.
  #    - If incoming.log NEVER appeared (box died before sync-metrics landed a
  #      file): clock from launch with a longer 60-min grace, so a never-heartbeat
  #      RUNNING box tears down at ~1h instead of leaking until the ~24h budget
  #      backstop — while not falsely killing a slow-starting healthy run.
  if [[ "$STATUS" == "RUNNING" && -z "$REASON" ]]; then
    HEARTBEAT="$PROJECT_DIR/runs/$ID/metrics/incoming.log"
    if [[ -f "$HEARTBEAT" ]]; then
      LAST_MOD=$(stat -f %m "$HEARTBEAT" 2>/dev/null || stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
      if (( LAST_MOD > 0 && NOW - LAST_MOD > 1800 )); then
        REASON="no-heartbeat-30min"
      fi
    else
      STARTED=$(echo "$row" | jq -r '.started_at_epoch // 0')
      if (( STARTED > 0 && NOW - STARTED > 3600 )); then
        REASON="no-heartbeat-ever-60min"
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
    # Tear down all instances in this row. VERIFY-AUTHORITATIVE classification:
    # an instance that is GONE counts as DESTROYED even if `destroy` itself
    # errored (e.g. it was already destroyed manually). This fixes the retry-
    # spam regression where an already-gone instance was classified FAILED on
    # every Stop forever (2026-06-04, an already-gone instance hit 8 attempts
    # because the old grep counted "not found" as a failure).
    # Conservative on ambiguity: an auth/network error from `show instance` does
    # NOT count as gone (that would falsely flip a live, billing box to
    # TORN_DOWN) — only a positive not-found does.
    DESTROYED=0
    FAILED=0
    # Resolve the account this row was provisioned on (default private) and its key,
    # so a team-account box is torn down with the team key.
    ROW_ACCT=$(echo "$row" | jq -r '.vast_account // "private"')
    ROW_KEY=$(vast_key_for "$ROW_ACCT")
    while IFS= read -r iid; do
      [[ -z "$iid" ]] && continue
      # Empty key guard: an exported-but-empty VAST_API_KEY makes the CLI fall back
      # to the stored (private) config, so a team box would be destroyed under the
      # wrong account and keep billing. Keep the row for a later retry.
      if [[ -z "$ROW_KEY" ]]; then
        echo "[$(date -Iseconds)] SKIP destroy $iid: no key for account=$ROW_ACCT (set VAST_API_KEY_TEAM)" >> /tmp/teardown.err
        FAILED=$((FAILED + 1)); continue
      fi
      # Co-ownership guard: if another live ledger row (a concurrent /loop session
      # reusing this box) references the same instance_id, do NOT destroy it — that
      # session owns its teardown (the box-reuse teardown race).
      CO=$(jq -s --arg iid "$iid" --arg self "$ID" '
        [ .[] | select(.id != $self)
              | select(.status=="RUNNING" or .status=="PROVISIONED")
              | select(any(.handles[]?.instance_id // empty; (.|tostring)==$iid)) ] | length' \
        "$LEDGER" 2>/dev/null || echo 0)
      if [[ "${CO:-0}" -gt 0 ]]; then
        echo "[$(date -Iseconds)] SKIP destroy $iid: co-owned by $CO other live ledger row(s) — leaving box for the owning session" >> /tmp/teardown.err
        FAILED=$((FAILED + 1)); continue
      fi
      # MUST pass -y (without it a non-TTY prompt collapses to "Aborted" yet the
      # CLI still exits 0 — a silent no-op; observed 2026-06-03 on 39132674).
      DOUT=$(timeout "$VAST_CLI_TIMEOUT" env VAST_API_KEY="$ROW_KEY" vastai destroy instance "$iid" -y 2>&1); DRC=$?
      echo "[$(date -Iseconds)] destroy $iid account=$ROW_ACCT rc=$DRC: $DOUT" >> /tmp/teardown.err
      if (( DRC == 0 )) && ! echo "$DOUT" | grep -qiE 'aborted|traceback|status_code|permission denied|^error'; then
        DESTROYED=$((DESTROYED + 1))                       # clean destroy
      elif echo "$DOUT" | grep -qiE 'not found|no such|does not exist|no longer exists|already (destroyed|gone)'; then
        DESTROYED=$((DESTROYED + 1))                       # already gone — goal achieved, NOT a failure
      else
        # Ambiguous — verify authoritatively. Only a POSITIVE not-found counts as
        # gone; still-listed OR an auth/network error is a conservative FAILED so
        # we never abandon a live, billing box.
        CHECK=$(timeout "$VAST_CLI_TIMEOUT" env VAST_API_KEY="$ROW_KEY" vastai show instance "$iid" --raw 2>&1 || true)
        if echo "$CHECK" | grep -qiE 'not found|no such|does not exist|404'; then
          DESTROYED=$((DESTROYED + 1))
        else
          FAILED=$((FAILED + 1))
        fi
      fi
    done < <(echo "$row" | jq -r '.handles[]? | .instance_id // empty')

    if (( FAILED == 0 )); then
      # Clean teardown, or no handles left to destroy — flip to TORN_DOWN.
      echo "$row" \
        | jq -c --arg t "$(date -Iseconds)" --arg r "$REASON" \
          '. + {status: "TORN_DOWN", torn_down_at: $t, teardown_reason: $r}' \
        >> "$TEMP"
      echo "[$(date -Iseconds)] teardown EXP-$ID reason=$REASON destroyed=$DESTROYED" \
        >> "$PROJECT_DIR/PROGRESS.md"
      TORN_ANY=1
    else
      # Genuine failure: instance(s) still listed. Keep the row for the next Stop
      # to retry (so a transient Vast outage eventually clears), but THROTTLE the
      # PROGRESS warning to once/hour so a stuck box can't spam the audit log on
      # every Stop (the 2026-06-04 regression that bloated PROGRESS.md).
      LAST_FAIL=$(echo "$row" | jq -r '.teardown_last_epoch // 0')
      echo "$row" \
        | jq -c --arg t "$(date -Iseconds)" --argjson te "$NOW" --arg r "$REASON" \
          --argjson d "$DESTROYED" --argjson f "$FAILED" \
          '. + {teardown_attempts: ((.teardown_attempts // 0) + 1),
                teardown_last_at: $t,
                teardown_last_epoch: $te,
                teardown_last_reason: $r,
                teardown_partial_destroyed: $d,
                teardown_partial_failed: $f}' \
        >> "$TEMP"
      if (( NOW - LAST_FAIL > 3600 )); then
        echo "[$(date -Iseconds)] TEARDOWN_FAILED EXP-$ID reason=$REASON destroyed=$DESTROYED failed=$FAILED — instance(s) still listed, check 'vastai show instances' (throttled 1/hr)" \
          >> "$PROJECT_DIR/PROGRESS.md"
      fi
    fi
  else
    echo "$row" >> "$TEMP"
  fi
done < "$LEDGER"

if [[ $LOCKED -eq 1 ]]; then
  mv "$TEMP" "$LEDGER"
else
  rm -f "$TEMP"
  echo "[$(date -Iseconds)] teardown: ledger lock timeout — destroys attempted, ledger rewrite skipped" >> /tmp/teardown.err
fi

if [[ $TORN_ANY -eq 1 ]]; then
  echo "teardown-finished-runs: tore down at least one stale Vast.ai handle. See PROGRESS.md." >&2
fi
exit 0
