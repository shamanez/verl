#!/usr/bin/env bash
# vast-teardown — destroy Vast.ai instances and patch runs.jsonl.
# See SKILL.md for usage. Safe to call repeatedly; idempotent on already-TORN_DOWN rows.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"
REASON="manual"
IDS=()
HANDLE_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reason)   REASON="$2"; shift 2 ;;
    --handles)  HANDLE_PATHS+=("$2"); shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$(dirname "$0")/SKILL.md"
      exit 0 ;;
    *)          IDS+=("$1"); shift ;;
  esac
done

# Expand handle paths (files or dirs of *.json) into instance ids.
for hp in "${HANDLE_PATHS[@]:-}"; do
  [[ -z "$hp" ]] && continue
  if [[ -d "$hp" ]]; then
    while IFS= read -r f; do
      iid=$(jq -r '.instance_id // empty' "$f" 2>/dev/null || true)
      [[ -n "$iid" ]] && IDS+=("$iid")
    done < <(find "$hp" -maxdepth 2 -type f -name '*.json')
  elif [[ -f "$hp" ]]; then
    iid=$(jq -r '.instance_id // empty' "$hp" 2>/dev/null || true)
    [[ -n "$iid" ]] && IDS+=("$iid")
  fi
done

if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "vast-teardown: no instance ids provided." >&2
  exit 1
fi

# Auth: team vs private, resolved PER-INSTANCE from the ledger's vast_account
# field — a team-account box must be destroyed with the team key, else the
# personal key 404s and the box leaks. The shared resolver loads both keys.
# An explicit VAST_ACCOUNT env var, if set, forces that account for ALL ids.
SECRETS_FILE="${VERL_SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
# shellcheck disable=SC1090
source "$(dirname "$0")/../_vast_account.sh"
vast_load_secrets
if [[ -z "${VAST_API_KEY:-}" && -z "${VAST_API_KEY_TEAM:-}" ]]; then
  echo "vast-teardown: no Vast API key (VAST_API_KEY / VAST_API_KEY_TEAM) in $SECRETS_FILE" >&2
  exit 1
fi

# Account a given instance id was provisioned on (default private). VAST_ACCOUNT
# env, if set, forces that account for every id (manual override).
acct_for_iid() {
  local iid="$1" a=""
  if [[ -n "${VAST_ACCOUNT:-}" ]]; then vast_account_norm "$VAST_ACCOUNT"; return; fi
  [[ -f "$LEDGER" ]] && a=$(jq -r --arg i "$iid" '
    select(any(.handles[]?.instance_id // empty; (.|tostring) == $i)) | .vast_account // empty' \
    "$LEDGER" 2>/dev/null | tail -1)
  vast_account_norm "${a:-private}"
}

ERR_LOG="/tmp/teardown.err"
: > "$ERR_LOG"
DESTROYED=()
FAILED=()
for iid in "${IDS[@]}"; do
  # MUST pass -y: `vastai destroy instance <id>` prompts interactively for
  # confirmation, and when stdin isn't a TTY the prompt collapses to "Aborted"
  # — but the CLI STILL EXITS 0. Without -y the destroy silently does nothing.
  ACCT=$(acct_for_iid "$iid"); KEY=$(vast_key_for "$ACCT")
  OUT=$(VAST_API_KEY="$KEY" vastai destroy instance "$iid" -y 2>&1) || true
  echo "[$iid] account=$ACCT $OUT" >>"$ERR_LOG"
  # Belt-and-braces: even with -y, treat "Aborted" / "error" anywhere in stdout
  # as a hard failure. The CLI's exit code alone is not trustworthy.
  if grep -qiE 'aborted|^error[: ]|status_code' <<<"$OUT"; then
    FAILED+=("$iid")
    continue
  fi
  # Verify the instance is actually gone (or marked stopping/destroyed).
  # `vastai show instance <id>` returns an object while it exists; once
  # destroyed it returns either an HTTP error or no payload.
  sleep 2
  CHECK=$(VAST_API_KEY="$KEY" vastai show instance "$iid" --raw 2>&1 || true)
  if echo "$CHECK" | grep -qiE 'error|not found|404' \
     || ! echo "$CHECK" | jq -e 'type=="object" and has("id")' >/dev/null 2>&1; then
    DESTROYED+=("$iid")
  else
    REMAINING_STATUS=$(echo "$CHECK" | jq -r '.actual_status // .cur_state // "unknown"' 2>/dev/null || echo "unknown")
    echo "[$iid] post-destroy still listed as $REMAINING_STATUS" >>"$ERR_LOG"
    FAILED+=("$iid")
  fi
done

# Patch the ledger: any row whose handles contain a destroyed id flips to TORN_DOWN.
if [[ -f "$LEDGER" && ${#DESTROYED[@]} -gt 0 ]]; then
  TS=$(date -Iseconds)
  TEMP=$(mktemp)
  IDS_JSON=$(printf '%s\n' "${DESTROYED[@]}" | jq -R . | jq -s .)
  while IFS= read -r row; do
    [[ -z "$row" ]] && continue
    NEW=$(jq -c --argjson ids "$IDS_JSON" --arg t "$TS" --arg r "$REASON" '
      # Patch RUNNING *and* PROVISIONED rows (an abandoned PROVISIONED box —
      # e.g. one that failed SSH-key injection before launch — must also flip
      # to TORN_DOWN once its instance is destroyed, else the Stop hook keeps
      # reaping it; 2026-06-04 EXP-20-dense-DEAD-39407768).
      if (.status == "RUNNING" or .status == "PROVISIONED")
         and (any(.handles[]?.instance_id // empty; . as $i | $ids | index($i)))
      then . + {status: "TORN_DOWN", torn_down_at: $t, teardown_reason: $r}
      else .
      end
    ' <<<"$row")
    echo "$NEW" >> "$TEMP"
  done < "$LEDGER"
  mv "$TEMP" "$LEDGER"
fi

echo "VAST_TORN_DOWN: destroyed=${#DESTROYED[@]} failed=${#FAILED[@]} reason=$REASON"
[[ ${#FAILED[@]} -gt 0 ]] && echo "vast-teardown: see $ERR_LOG for failures." >&2

# Never block: exit 0 even if some destroys failed.
exit 0
