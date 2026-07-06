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

# Expand handle paths (files or dirs of *.json) into instance ids. The vast_account
# for each id is resolved later by acct_for_iid, which re-scans these same paths —
# kept map-free for bash 3.2 (macOS default ships no associative-array support).
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
# Scan the --handles paths for a handle whose instance_id matches $1, echo its
# vast_account. bash 3.2 has no associative arrays, so we re-scan instead of a map.
acct_from_handles() {
  local iid="$1" hp f
  for hp in "${HANDLE_PATHS[@]:-}"; do
    [[ -z "$hp" ]] && continue
    if [[ -d "$hp" ]]; then
      while IFS= read -r f; do
        if [[ "$(jq -r '.instance_id // empty' "$f" 2>/dev/null)" == "$iid" ]]; then
          jq -r '.vast_account // empty' "$f" 2>/dev/null; return
        fi
      done < <(find "$hp" -maxdepth 2 -type f -name '*.json')
    elif [[ -f "$hp" ]]; then
      if [[ "$(jq -r '.instance_id // empty' "$hp" 2>/dev/null)" == "$iid" ]]; then
        jq -r '.vast_account // empty' "$hp" 2>/dev/null; return
      fi
    fi
  done
}

acct_for_iid() {
  local iid="$1" a=""
  if [[ -n "${VAST_ACCOUNT:-}" ]]; then vast_account_norm "$VAST_ACCOUNT"; return; fi
  # 1) account stamped on a handle JSON passed via --handles (most authoritative)
  a="$(acct_from_handles "$iid")"
  # 2) the provision handle dir (handles there are keyed by instance id)
  if [[ -z "$a" ]]; then
    local hf="$PROJECT_DIR/.claude/state/vast-handles/${iid}.json"
    [[ -r "$hf" ]] && a=$(jq -r '.vast_account // empty' "$hf" 2>/dev/null || true)
  fi
  # 3) the ledger row that references this instance id
  if [[ -z "$a" && -f "$LEDGER" ]]; then
    a=$(jq -r --arg i "$iid" '
      select(any(.handles[]?.instance_id // empty; (.|tostring) == $i)) | .vast_account // empty' \
      "$LEDGER" 2>/dev/null | tail -1)
  fi
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
  # Never call vastai with an EMPTY key: an exported-but-empty VAST_API_KEY makes
  # the CLI silently fall back to ~/.config/vastai (the PRIVATE key), so a team box
  # would be "destroyed" under the wrong account, 404, and keep billing.
  if [[ -z "$KEY" ]]; then
    echo "[$iid] no API key for account=$ACCT — skipping (set VAST_API_KEY_TEAM in $SECRETS_FILE)" >>"$ERR_LOG"
    FAILED+=("$iid")
    continue
  fi
  OUT=$(timeout 90 env VAST_API_KEY="$KEY" vastai destroy instance "$iid" -y 2>&1) || true
  echo "[$iid] account=$ACCT $OUT" >>"$ERR_LOG"
  # Already-gone = goal achieved (idempotent). `destroy` errors on a missing
  # instance, so check this BEFORE the generic error->FAILED guard (mirrors the
  # Stop hook's verify-authoritative classification).
  if grep -qiE 'not found|no such|does not exist|no longer (exists|gone)|already (destroyed|gone)|"?404"?' <<<"$OUT"; then
    DESTROYED+=("$iid")
    continue
  fi
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
  CHECK=$(timeout 90 env VAST_API_KEY="$KEY" vastai show instance "$iid" --raw 2>&1 || true)
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
# Locked (shared spinlock with _lib.sh / the Stop hook) so a concurrent append is
# never lost by this whole-file rewrite; bounded 30s, then skip rather than hang.
if [[ -f "$LEDGER" && ${#DESTROYED[@]} -gt 0 ]]; then
  LOCKDIR="$PROJECT_DIR/.claude/state/.runs.jsonl.lock"; n=0; LOCKED=1
  until mkdir "$LOCKDIR" 2>/dev/null; do
    n=$((n+1)); (( n > 300 )) && { LOCKED=0; break; }; sleep 0.1
  done
  if [[ "$LOCKED" == 1 ]]; then
    TS=$(date -Iseconds)
    TEMP=$(mktemp)
    IDS_JSON=$(printf '%s\n' "${DESTROYED[@]}" | jq -R . | jq -s .)
    while IFS= read -r row; do
      [[ -z "$row" ]] && continue
      NEW=$(jq -c --argjson ids "$IDS_JSON" --arg t "$TS" --arg r "$REASON" '
        # RUNNING, PROVISIONED *and* EXTERNAL rows all flip once their instance is
        # destroyed (EXTERNAL = operator-managed lifecycle, but a destroyed box must
        # never linger as a live-looking row).
        if (.status == "RUNNING" or .status == "PROVISIONED" or .status == "EXTERNAL")
           and (any(.handles[]?.instance_id // empty; . as $i | $ids | index($i)))
        then . + {status: "TORN_DOWN", torn_down_at: $t, teardown_reason: $r}
        else .
        end
      ' <<<"$row")
      echo "$NEW" >> "$TEMP"
    done < "$LEDGER"
    mv "$TEMP" "$LEDGER"
    rmdir "$LOCKDIR" 2>/dev/null || true
  else
    echo "vast-teardown: ledger lock timeout — instances destroyed but rows not flipped (next sweep will reconcile)" >&2
  fi
fi

echo "VAST_TORN_DOWN: destroyed=${#DESTROYED[@]} failed=${#FAILED[@]} reason=$REASON"
[[ ${#FAILED[@]} -gt 0 ]] && echo "vast-teardown: see $ERR_LOG for failures." >&2

# Never block: exit 0 even if some destroys failed.
exit 0
