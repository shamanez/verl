#!/usr/bin/env bash
# PostToolUse (matcher: Bash) — periodically pull remote train.log tails for each
# RUNNING experiment in runs.jsonl. Debounced to once per 5 min so we don't ssh on
# every Bash call. Best-effort: ssh failures never block Claude.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"
DEBOUNCE="$PROJECT_DIR/.claude/state/.last-sync"
DEBOUNCE_SEC=300

# SSH identity — sourced from project.yaml (vast_ssh.identity_file). Falls back
# to ~/.ssh/vast_ai_name if project.yaml is missing/unparseable. Bare `ssh root@host`
# silently picks id_rsa/id_ed25519 and gets `Permission denied (publickey)` from
# Vast.ai boxes, which is what kept this hook broken before.
SSH_IDENTITY="$(awk '/^[[:space:]]*identity_file:/ {
  sub(/^[[:space:]]*identity_file:[[:space:]]*/, "", $0)
  sub(/[[:space:]]*#.*$/, "", $0)
  gsub(/["'"'"']/, "", $0)
  print; exit
}' "$PROJECT_DIR/.claude/project.yaml" 2>/dev/null)"
SSH_IDENTITY="${SSH_IDENTITY:-~/.ssh/vast_ai_name}"
SSH_IDENTITY="${SSH_IDENTITY/#~/$HOME}"   # expand leading ~
SSH_OPTS=(-i "$SSH_IDENTITY" -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

[[ -f "$LEDGER" ]] || exit 0

NOW=$(date +%s)
if [[ -f "$DEBOUNCE" ]]; then
  LAST=$(cat "$DEBOUNCE" 2>/dev/null || echo 0)
  if (( NOW - LAST < DEBOUNCE_SEC )); then
    exit 0
  fi
fi
echo "$NOW" > "$DEBOUNCE"

# Walk RUNNING rows. Each row's handles[] is the list of Vast.ai instances for this run.
# We expect each handle to carry ssh_host + ssh_port (the vast-provision skill writes these).
jq -c 'select(.status == "RUNNING")' "$LEDGER" 2>/dev/null | while IFS= read -r row; do
  ID=$(echo "$row" | jq -r '.id')
  OUT_DIR="$PROJECT_DIR/runs/$ID/metrics"
  mkdir -p "$OUT_DIR"

  echo "$row" | jq -c '.handles[]?' 2>/dev/null | while IFS= read -r h; do
    HOST=$(echo "$h" | jq -r '.ssh_host // empty')
    PORT=$(echo "$h" | jq -r '.ssh_port // "22"')
    [[ -z "$HOST" ]] && continue

    # CRITICAL: collect SSH output into a temp file FIRST. Only append it to
    # incoming.log (and thereby refresh its mtime — which the teardown hook
    # uses as the heartbeat signal) if SSH actually succeeded with non-empty
    # content. A failed SSH must NOT refresh the heartbeat, otherwise dead
    # training jobs stay "alive" in the harness and never get torn down.
    TMP_TAIL=$(mktemp -t sync-metrics.XXXXXX)
    SSH_RC=0
    ssh "${SSH_OPTS[@]}" \
        -p "$PORT" "root@$HOST" \
        'tail -n 200 /workspace/train.log 2>/dev/null' \
        > "$TMP_TAIL" 2>>"$OUT_DIR/sync-errors.log" || SSH_RC=$?

    if (( SSH_RC == 0 )) && [[ -s "$TMP_TAIL" ]]; then
      {
        echo "--- $(date -Iseconds) host=$HOST port=$PORT ---"
        cat "$TMP_TAIL"
      } >> "$OUT_DIR/incoming.log" 2>/dev/null || true
    else
      echo "[$(date -Iseconds)] sync failed host=$HOST rc=$SSH_RC empty=$([[ -s $TMP_TAIL ]] && echo no || echo yes)" \
        >> "$OUT_DIR/sync-errors.log" 2>/dev/null || true
    fi
    rm -f "$TMP_TAIL"

    # Vast-volatility safety: also pull any /workspace/runs/<ID>/hotfix-patches/ back.
    # The in-container commit-hotfix.sh helper drops `.patch` files there; if the
    # instance dies before the experiment finishes, the patches survive here.
    HOTFIX_LOCAL="$PROJECT_DIR/runs/$ID/hotfix-patches"
    mkdir -p "$HOTFIX_LOCAL"
    rsync -av --ignore-existing \
      -e "ssh -i $SSH_IDENTITY -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" \
      "root@$HOST:/workspace/runs/$ID/hotfix-patches/" \
      "$HOTFIX_LOCAL/" \
      >> "$OUT_DIR/sync-errors.log" 2>&1 || true
  done
done

exit 0
