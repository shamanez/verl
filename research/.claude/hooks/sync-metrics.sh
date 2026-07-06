#!/usr/bin/env bash
# PostToolUse (matcher: Bash) — pull remote train.log tails for each RUNNING run.
# Debounced to once per 5 min. Best-effort: failures never block Claude.
#
# HEARTBEAT CONTRACT (the GPU-never-stale core): incoming.log's mtime is the
# teardown reaper's liveness signal, so we refresh it ONLY when the remote log
# CONTENT ADVANCED since the last sync. A dead-but-reachable box that serves the
# same stale tail forever therefore goes heartbeat-stale and is reaped at 30 min
# (this closes the "monitors keep mtime fresh" defeat, memory 2026-06).
set -euo pipefail

# macOS has no timeout(1); perl alarm survives execve (same shim as _lib.sh).
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"
DEBOUNCE="$PROJECT_DIR/.claude/state/.last-sync"
DEBOUNCE_SEC=300
HARD_TIMEOUT=40   # seconds per ssh/rsync — a hung connection must never hang the hook

SSH_IDENTITY="$(awk '/^[[:space:]]*identity_file:/ {
  sub(/^[[:space:]]*identity_file:[[:space:]]*/, "", $0)
  sub(/[[:space:]]*#.*$/, "", $0); gsub(/["'"'"']/, "", $0); print; exit
}' "$PROJECT_DIR/.claude/project.yaml" 2>/dev/null)"
SSH_IDENTITY="${SSH_IDENTITY:-~/.ssh/vast_ai_name}"
SSH_IDENTITY="${SSH_IDENTITY/#~/$HOME}"
SSH_OPTS=(-i "$SSH_IDENTITY" -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

[[ -f "$LEDGER" ]] || exit 0

NOW=$(date +%s)
if [[ -f "$DEBOUNCE" ]]; then
  LAST=$(cat "$DEBOUNCE" 2>/dev/null || echo 0)
  (( NOW - LAST < DEBOUNCE_SEC )) && exit 0
fi
echo "$NOW" > "$DEBOUNCE"

jq -c 'select(.status == "RUNNING")' "$LEDGER" 2>/dev/null | while IFS= read -r row; do
  ID=$(echo "$row" | jq -r '.id')
  RUN_DIR="$PROJECT_DIR/runs/$ID"
  # Graceful degradation: a deleted run dir is NOT resurrected. The run then goes
  # heartbeat-stale and the reaper tears the box down — the correct backstop for
  # an operator-deleted dir under a live row (de-bloat refuses live rows anyway).
  [[ -d "$RUN_DIR" ]] || continue
  OUT_DIR="$RUN_DIR/metrics"
  mkdir -p "$OUT_DIR"
  # Per-run remote log path (run.json/ledger may override the default).
  REMOTE_LOG=$(echo "$row" | jq -r '.remote_log // empty')
  [[ -z "$REMOTE_LOG" && -f "$RUN_DIR/run.json" ]] \
    && REMOTE_LOG=$(jq -r '.remote_log // empty' "$RUN_DIR/run.json" 2>/dev/null)
  REMOTE_LOG="${REMOTE_LOG:-/workspace/train.log}"

  echo "$row" | jq -c '.handles[]?' 2>/dev/null | while IFS= read -r h; do
    HOST=$(echo "$h" | jq -r '.ssh_host // empty')
    PORT=$(echo "$h" | jq -r '.ssh_port // "22"')
    [[ -z "$HOST" ]] && continue

    TMP_TAIL=$(mktemp -t sync-metrics.XXXXXX)
    SSH_RC=0
    # -n: never let ssh drain the while-loop's stdin (it would eat the
    # remaining handle rows and silently skip every node after the first).
    timeout "$HARD_TIMEOUT" ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
        "tail -n 200 '$REMOTE_LOG' 2>/dev/null" \
        > "$TMP_TAIL" 2>>"$OUT_DIR/sync-errors.log" || SSH_RC=$?

    if (( SSH_RC == 0 )) && [[ -s "$TMP_TAIL" ]]; then
      # PROGRESS CHECK: append (= refresh heartbeat) only if the tail moved.
      # Sig is PER HOST — with multi-node handles a shared sig would alternate
      # every cycle and mask a single frozen node.
      SIG=$(tail -n 3 "$TMP_TAIL" | cksum | awk '{print $1}')
      SIG_FILE="$OUT_DIR/.last-tail-sig.$HOST.$PORT"
      PREV=$(cat "$SIG_FILE" 2>/dev/null || echo "")
      if [[ "$SIG" != "$PREV" ]]; then
        echo "$SIG" > "$SIG_FILE"
        { echo "--- $(date -Iseconds) host=$HOST port=$PORT ---"; cat "$TMP_TAIL"; } \
          >> "$OUT_DIR/incoming.log" 2>/dev/null || true
      else
        echo "[$(date -Iseconds)] no-progress host=$HOST (tail unchanged — heartbeat NOT refreshed)" \
          >> "$OUT_DIR/sync-errors.log" 2>/dev/null || true
      fi
    else
      echo "[$(date -Iseconds)] sync failed host=$HOST rc=$SSH_RC" \
        >> "$OUT_DIR/sync-errors.log" 2>/dev/null || true
    fi
    rm -f "$TMP_TAIL"

    # Pull any in-container hotfix patches back (Vast volatility safety).
    HOTFIX_LOCAL="$RUN_DIR/hotfix-patches"
    mkdir -p "$HOTFIX_LOCAL"
    timeout "$HARD_TIMEOUT" rsync -a --ignore-existing \
      -e "ssh -i $SSH_IDENTITY -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" \
      "root@$HOST:/workspace/runs/$ID/hotfix-patches/" "$HOTFIX_LOCAL/" \
      >> "$OUT_DIR/sync-errors.log" 2>&1 || true
  done
done

exit 0
