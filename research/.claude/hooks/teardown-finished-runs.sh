#!/usr/bin/env bash
# Stop — tear down Vast.ai instances for finished/stale runs. The money backstop.
# Triggers per RUNNING row: verdict.md written · heartbeat stale >30 min ·
# never-any-heartbeat >60 min · budget (gpu-hr) exceeded. Per PROVISIONED row:
# stale >15 min. Plus an ORPHAN-HANDLE sweep: any harness-provisioned handle
# whose instance has NO ledger row (runner died mid-provision) is destroyed
# after a 45-min grace. EXTERNAL rows are operator-managed — never touched.
#
# Concurrency: destroys run UNLOCKED against a snapshot; the ledger lock is
# taken only for the final rewrite, so concurrent appends are never starved or
# lost. Every vastai call is hard-bounded; rc 124/126/127 (timeout / not
# executable / not found) is a hard FAILURE — never "already gone" (a live box
# must never flip TORN_DOWN because a wrapper failed to run).
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
# Re-anchor to the PRIMARY checkout's research dir, exactly like _lib.sh does.
# Skills (via _lib.sh) register the ledger + create runs/<id>/ in the PRIMARY
# checkout even from a worktree session; hooks get $CLAUDE_PROJECT_DIR = the
# worktree root, where .claude/state/runs.jsonl is gitignored/absent. Without
# this the reaper reads the wrong (empty) ledger and heartbeat dir — leaking
# boxes on window-close and false-reaping healthy ones. Degrades to a no-op
# when git is unavailable or this already IS the primary checkout.
_main=$(git -C "$PROJECT_DIR" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')
[[ -n "$_main" && -d "$_main/research" ]] && PROJECT_DIR="$_main/research"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"
ERRLOG="/tmp/teardown.err"
VAST_CLI_TIMEOUT=90
NOW=$(date +%s)

# macOS has no timeout(1); perl alarm survives execve (same shim as _lib.sh).
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

# Account → key resolution (team box must be destroyed with the team key).
if [[ -f "$PROJECT_DIR/.claude/skills/_vast_account.sh" ]]; then
  # shellcheck disable=SC1090
  source "$PROJECT_DIR/.claude/skills/_vast_account.sh"
  vast_load_secrets
else
  vast_key_for() {
    case "${1:-private}" in
      team) printf '%s' "${VAST_API_KEY_TEAM:-}" ;;   # NO private fallback — empty key = skip, never wrong-account destroy
      *)    printf '%s' "${VAST_API_KEY:-}" ;;
    esac
  }
fi

# destroy_one <iid> <key> — prints DESTROYED or FAILED. Verify-authoritative:
# already-gone counts as destroyed; wrapper/CLI failure (rc 124/126/127) and
# ambiguous-still-listed count as FAILED so we never abandon a live box.
destroy_one() {
  local iid="$1" key="$2" dout drc check crc
  dout=$(timeout "$VAST_CLI_TIMEOUT" env VAST_API_KEY="$key" vastai destroy instance "$iid" -y 2>&1); drc=$?
  echo "[$(date -Iseconds)] destroy $iid rc=$drc: $dout" >> "$ERRLOG"
  # rc 124/126/127 = GNU timeout / not-executable / not-found; 142 = 128+SIGALRM,
  # the macOS perl-alarm shim's timeout code (GNU's 124 never appears on this
  # laptop). All mean the destroy NEVER RAN -> hard FAILED, never "already gone".
  if (( drc == 124 || drc == 126 || drc == 127 || drc == 142 )); then echo FAILED; return 0; fi
  # Explicit already-gone phrasing = idempotent success.
  if grep -qiE 'instance not found|no such instance|does not exist|no longer exists|already (destroyed|gone)' <<<"$dout"; then
    echo DESTROYED; return 0
  fi
  # VERIFY-AUTHORITATIVE (mirror the manual vast-teardown, #63): a clean rc=0 is
  # NOT trusted on its own — the CLI can exit 0 without destroying. Always
  # re-query show-instance; DESTROYED only when the box is provably gone.
  sleep 2
  check=$(timeout "$VAST_CLI_TIMEOUT" env VAST_API_KEY="$key" vastai show instance "$iid" --raw 2>&1); crc=$?
  if (( crc == 124 || crc == 126 || crc == 127 || crc == 142 )); then echo FAILED; return 0; fi
  if grep -qiE 'instance not found|no such instance|does not exist|404' <<<"$check" \
     || ! jq -e 'type=="object" and has("id")' >/dev/null 2>&1 <<<"$check"; then
    echo DESTROYED; return 0
  fi
  echo FAILED
}

# heartbeat_alive <row> <id> — 0 if the box is SSH-reachable AND its remote
# training log advanced since the reaper's last probe. Spares a healthy box
# whose in-session sync-metrics simply isn't running (all windows closed, or a
# step slower than the stale threshold). UNREACHABLE or NOT-advancing => 1
# (let teardown proceed). Bounded; any error => 1 so it never blocks a real reap.
heartbeat_alive() {
  local row="$1" id="$2" h host port ident rlog sig prev sigfile
  h=$(jq -c '.handles[0] // empty' <<<"$row" 2>/dev/null); [[ -n "$h" && "$h" != "null" ]] || return 1
  host=$(jq -r '.ssh_host // empty' <<<"$h"); port=$(jq -r '.ssh_port // 22' <<<"$h")
  ident=$(jq -r '.ssh_identity // empty' <<<"$h"); ident="${ident/#\~/$HOME}"
  [[ -n "$host" && -n "$ident" && -r "$ident" ]] || return 1
  rlog=$(jq -r '.remote_log // empty' <<<"$row")
  [[ -z "$rlog" && -r "$PROJECT_DIR/runs/$id/run.json" ]] && \
    rlog=$(jq -r '.remote_log // empty' "$PROJECT_DIR/runs/$id/run.json" 2>/dev/null)
  rlog="${rlog:-/workspace/train.log}"
  sig=$(timeout 25 ssh -n -i "$ident" -o ConnectTimeout=8 -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" \
        "tail -n 3 '$rlog' 2>/dev/null | cksum" 2>/dev/null) || return 1
  [[ -n "$sig" ]] || return 1
  sigfile="$PROJECT_DIR/runs/$id/metrics/.reaper-probe-sig"
  prev=$(cat "$sigfile" 2>/dev/null || echo "")
  mkdir -p "$(dirname "$sigfile")" 2>/dev/null || true
  echo "$sig" > "$sigfile" 2>/dev/null || true
  # No prior probe, or the tail advanced => alive; refresh the heartbeat mtime so
  # we don't re-SSH every cycle. Unchanged since last probe => genuinely stalled.
  if [[ -z "$prev" || "$sig" != "$prev" ]]; then
    touch "$PROJECT_DIR/runs/$id/metrics/incoming.log" 2>/dev/null || true
    return 0
  fi
  return 1
}

[[ -f "$LEDGER" ]] || exit 0

# ── Phase A (UNLOCKED): decide + destroy against a snapshot ──────────────────
SNAP=$(mktemp); cp "$LEDGER" "$SNAP"
RESULTS=$(mktemp)   # jsonl: {id, reason, destroyed, failed}
TORN_ANY=0

while IFS= read -r row || [[ -n "$row" ]]; do
  [[ -z "$row" ]] && continue
  STATUS=$(jq -r '.status // "UNKNOWN"' <<<"$row" 2>/dev/null) || continue
  case "$STATUS" in RUNNING|PROVISIONED) ;; *) continue ;; esac
  ID=$(jq -r '.id' <<<"$row")
  REASON=""

  # 1. verdict written (RUNNING only)
  if [[ "$STATUS" == "RUNNING" && -f "$PROJECT_DIR/runs/$ID/verdict.md" ]]; then
    REASON="verdict-written"
  fi
  # 2. heartbeat stale / never appeared (RUNNING only). sync-metrics refreshes
  #    incoming.log mtime ONLY on content advance, so mtime == training progress.
  #    OPERATOR_STOP sentinel (#63 B10): the operator stopped this run ON PURPOSE
  #    (reconfigure/inspect) — heartbeat silence is EXPECTED, do NOT reap on it.
  #    Budget trigger (3.) still applies: a stopped box still bills.
  if [[ "$STATUS" == "RUNNING" && -z "$REASON" && -f "$PROJECT_DIR/runs/$ID/OPERATOR_STOP" ]]; then
    :  # heartbeat triggers suppressed
  elif [[ "$STATUS" == "RUNNING" && -z "$REASON" ]]; then
    HB="$PROJECT_DIR/runs/$ID/metrics/incoming.log"
    if [[ -f "$HB" ]]; then
      LAST_MOD=$(stat -f %m "$HB" 2>/dev/null || stat -c %Y "$HB" 2>/dev/null || echo 0)
      (( LAST_MOD > 0 && NOW - LAST_MOD > 1800 )) && REASON="no-heartbeat-30min"
    else
      STARTED=$(jq -r '.started_at_epoch // 0' <<<"$row")
      (( STARTED > 0 && NOW - STARTED > 3600 )) && REASON="no-heartbeat-ever-60min"
    fi
  fi
  # 3. budget exceeded. Gate on max_gpu_hr ONLY — the gpu-hr cap is elapsed×gpus,
  #    it does not use dph. The old `d > 0` gate meant a row with an unresolved
  #    dph=0 (e.g. a --instance-id attach whose price didn't resolve) was NEVER
  #    budget-capped, so only the heartbeat path could ever stop it billing.
  if [[ -z "$REASON" ]]; then
    STARTED=$(jq -r '.started_at_epoch // 0' <<<"$row")
    MAX_GPU_HR=$(jq -r '.max_gpu_hr // 0' <<<"$row")
    TOTAL_GPUS=$(jq -r '.total_gpus // (.per_node_gpus // 1)' <<<"$row")
    if (( STARTED > 0 )) && awk -v m="$MAX_GPU_HR" 'BEGIN { exit !(m > 0) }'; then
      EXCEEDED=$(awk -v n="$NOW" -v s="$STARTED" -v g="$TOTAL_GPUS" -v mg="$MAX_GPU_HR" \
        'BEGIN { print (((n-s)/3600)*g > mg) ? 1 : 0 }')
      [[ "$EXCEEDED" == "1" ]] && REASON="budget-exceeded"
    fi
  fi
  # 4. PROVISIONED stale >15 min (runner died between handle capture and launch)
  if [[ "$STATUS" == "PROVISIONED" && -z "$REASON" ]]; then
    STARTED=$(jq -r '.started_at_epoch // 0' <<<"$row")
    (( STARTED > 0 && NOW - STARTED > 900 )) && REASON="provisioned-but-never-launched"
  fi

  # Heartbeat reasons only: actively re-probe before reaping. The stale-heartbeat
  # signal can be a FALSE alarm (no session open to run sync-metrics, or a step
  # slower than the threshold). budget/verdict/provisioned reasons are NOT spared
  # — an over-budget box must die even while advancing.
  if [[ "$REASON" == no-heartbeat-* ]] && heartbeat_alive "$row" "$ID"; then
    echo "[$(date -Iseconds)] SPARE $ID: $REASON but box reachable + log advancing" >> "$ERRLOG"
    REASON=""
  fi
  [[ -z "$REASON" ]] && continue

  ROW_ACCT=$(jq -r '.vast_account // "team"' <<<"$row")
  # Match provisioning + vast-teardown, which default a missing account to TEAM
  # (private-default here would destroy a team box with the wrong key -> 404 ->
  # misread as already-gone -> silent leak). Guard the empty-string case too
  # (jq's // only fills null/absent, not "").
  [[ -z "$ROW_ACCT" || "$ROW_ACCT" == "null" ]] && ROW_ACCT="team"
  ROW_KEY=$(vast_key_for "$ROW_ACCT")
  DESTROYED=0; FAILED=0
  while IFS= read -r iid; do
    [[ -z "$iid" ]] && continue
    # Synthetic / non-numeric id (vast-attach --ssh-login box with no resolvable
    # Vast id): un-destroyable via the API, and a "no such instance" reply would
    # FALSELY flip a live box to TORN_DOWN. Skip (count FAILED so the row stays
    # live) — the throttled TEARDOWN_FAILED line surfaces it for MANUAL teardown.
    if [[ ! "$iid" =~ ^[0-9]+$ ]]; then
      echo "[$(date -Iseconds)] SKIP destroy $iid: non-numeric/synthetic id — needs MANUAL teardown" >> "$ERRLOG"
      FAILED=$((FAILED + 1)); continue
    fi
    if [[ -z "$ROW_KEY" ]]; then
      echo "[$(date -Iseconds)] SKIP destroy $iid: no key for account=$ROW_ACCT" >> "$ERRLOG"
      FAILED=$((FAILED + 1)); continue
    fi
    # Co-ownership guard: another live row (concurrent session) references this
    # instance — that session owns its teardown.
    CO=$(jq -s --arg iid "$iid" --arg self "$ID" '
      [ .[] | select(.id != $self)
            | select(.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL")
            | select(any(.handles[]?.instance_id // empty; (.|tostring)==$iid)) ] | length' \
      "$SNAP" 2>/dev/null || echo 0)
    if [[ "${CO:-0}" -gt 0 ]]; then
      echo "[$(date -Iseconds)] SKIP destroy $iid: co-owned by $CO other live row(s)" >> "$ERRLOG"
      FAILED=$((FAILED + 1)); continue
    fi
    if [[ "$(destroy_one "$iid" "$ROW_KEY")" == "DESTROYED" ]]; then
      DESTROYED=$((DESTROYED + 1))
    else
      FAILED=$((FAILED + 1))
    fi
  done < <(jq -r '.handles[]? | .instance_id // empty' <<<"$row")

  jq -nc --arg id "$ID" --arg r "$REASON" --argjson d "$DESTROYED" --argjson f "$FAILED" \
    '{id:$id, reason:$r, destroyed:$d, failed:$f}' >> "$RESULTS"
done < "$SNAP"

# ── Orphan-handle sweep: harness-provisioned boxes with NO ledger row ────────
# Covers the runner-died-mid-provision window. external:true handles are
# operator-provided — never swept. 45-min grace covers the normal capture→
# register gap (seconds) with a wide margin.
for h in "$PROJECT_DIR"/runs/*/handles/*.json "$PROJECT_DIR"/.claude/state/vast-handles/*.json; do
  [[ -e "$h" ]] || continue
  [[ "$h" == *.reaped ]] && continue
  IID=$(jq -r '.instance_id // empty' "$h" 2>/dev/null); [[ -z "$IID" ]] && continue
  EXT=$(jq -r '.external // false' "$h" 2>/dev/null); [[ "$EXT" == "true" ]] && continue
  H_MOD=$(stat -f %m "$h" 2>/dev/null || stat -c %Y "$h" 2>/dev/null || echo 0)
  (( H_MOD == 0 || NOW - H_MOD < 2700 )) && continue
  # Any ledger row referencing this instance (any status) means it's accounted for.
  KNOWN=$(jq -s --arg iid "$IID" \
    '[ .[] | select(any(.handles[]?.instance_id // empty; (.|tostring)==$iid)) ] | length' \
    "$SNAP" 2>/dev/null || echo 0)
  [[ "${KNOWN:-0}" -gt 0 ]] && continue
  H_ACCT=$(jq -r '.vast_account // "private"' "$h" 2>/dev/null)
  H_KEY=$(vast_key_for "$H_ACCT"); [[ -z "$H_KEY" ]] && continue
  if [[ "$(destroy_one "$IID" "$H_KEY")" == "DESTROYED" ]]; then
    mv "$h" "$h.reaped" 2>/dev/null || true
    echo "[$(date -Iseconds)] teardown ORPHAN handle $IID (no ledger row) destroyed" \
      >> "$PROJECT_DIR/PROGRESS.md"
    TORN_ANY=1
  fi
done

# ── Phase B (LOCKED, bounded): apply results to the LIVE ledger ──────────────
if [[ -s "$RESULTS" ]]; then
  LOCKDIR="$PROJECT_DIR/.claude/state/.runs.jsonl.lock"; n=0; LOCKED=1
  until mkdir "$LOCKDIR" 2>/dev/null; do
    AGE=$(( NOW - $(stat -f %m "$LOCKDIR" 2>/dev/null || stat -c %Y "$LOCKDIR" 2>/dev/null || echo "$NOW") ))
    (( AGE > 300 )) && { rmdir "$LOCKDIR" 2>/dev/null || true; continue; }   # stale lock from a crashed holder
    n=$((n+1)); (( n > 300 )) && { LOCKED=0; break; }; sleep 0.1
  done
  if [[ "$LOCKED" == 1 ]]; then
    TEMP=$(mktemp)
    jq -c --slurpfile res "$RESULTS" --arg t "$(date -Iseconds)" --argjson now "$NOW" '
      . as $row
      | ($res | map(select(.id == $row.id)) | first) as $r
      | if ($r == null) or ($row.status != "RUNNING" and $row.status != "PROVISIONED") then $row
        elif $r.failed == 0 then
          $row + {status:"TORN_DOWN", torn_down_at:$t, teardown_reason:$r.reason}
        else
          $row + {teardown_attempts: (($row.teardown_attempts // 0) + 1),
                  teardown_last_at:$t, teardown_last_epoch:$now,
                  teardown_last_reason:$r.reason,
                  teardown_partial_destroyed:$r.destroyed,
                  teardown_partial_failed:$r.failed}
        end' "$LEDGER" > "$TEMP" && mv "$TEMP" "$LEDGER"
    rmdir "$LOCKDIR" 2>/dev/null || true
  else
    echo "[$(date -Iseconds)] teardown: ledger lock timeout — destroys done, rewrite deferred to next Stop" >> "$ERRLOG"
  fi

  # PROGRESS lines (clean flips now; failures throttled to 1/hr via the row fields)
  while IFS= read -r r; do
    RID=$(jq -r '.id' <<<"$r"); RF=$(jq -r '.failed' <<<"$r")
    if [[ "$RF" == "0" ]]; then
      echo "[$(date -Iseconds)] teardown $RID reason=$(jq -r '.reason' <<<"$r") destroyed=$(jq -r '.destroyed' <<<"$r")" \
        >> "$PROJECT_DIR/PROGRESS.md"
      TORN_ANY=1
    else
      LAST_FAIL=$(jq -r --arg id "$RID" 'select(.id==$id) | .teardown_last_epoch // 0' "$SNAP" 2>/dev/null | tail -1)
      if (( NOW - ${LAST_FAIL:-0} > 3600 )); then
        echo "[$(date -Iseconds)] TEARDOWN_FAILED $RID — instance(s) still listed, check 'vastai show instances' (throttled 1/hr)" \
          >> "$PROJECT_DIR/PROGRESS.md"
      fi
    fi
  done < "$RESULTS"
fi

rm -f "$SNAP" "$RESULTS"
if [[ $TORN_ANY -eq 1 ]]; then
  echo "teardown-finished-runs: tore down at least one stale Vast.ai handle. See PROGRESS.md." >&2
fi
exit 0
