#!/usr/bin/env bash
# vast-attach — register an ALREADY-RUNNING, operator-provided box so the harness
# can use it WITHOUT provisioning. Companion to vast-provision. See SKILL.md.
#
# Two lifecycles, chosen explicitly:
#   default        -> ledger status:"RUNNING", external:true, max_gpu_hr backstop.
#                     Reaped by the teardown hook like any provisioned box.
#   --manual       -> ledger status:"EXTERNAL". Tracked (vast-cost sees it) but
#                     NEVER auto-reaped: for operator-managed analysis/download
#                     boxes (the box-43495538 incident class). Teardown is the
#                     operator's explicit act (vast-teardown handles EXTERNAL).
set -euo pipefail

# macOS has no timeout(1); perl alarm survives execve (same shim as _lib.sh).
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SKILL_DIR/../../.." && pwd)}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"

EXP_ID=""; INSTANCE_ID=""; SSH_HOST=""; SSH_PORT=""; NUM_GPUS=""
GPU_NAME=""; GPU_RAM="0"; DPH="0"; ACCOUNT="private"; REGISTER=1
MANUAL=0; MAX_GPU_HR="24"; NO_PROBE=0; ISSUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id)      EXP_ID="$2"; shift 2 ;;
    --issue)       ISSUE="$2"; shift 2 ;;   # issue number — REQUIRED for /launch-driven attaches (ledger_row_by_issue keys on it)
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --ssh-host)    SSH_HOST="$2"; shift 2 ;;
    --ssh-port)    SSH_PORT="$2"; shift 2 ;;
    --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
    --gpu-name)    GPU_NAME="$2"; shift 2 ;;
    --gpu-ram)     GPU_RAM="$2"; shift 2 ;;
    --dph)         DPH="$2"; shift 2 ;;
    --account)     ACCOUNT="$2"; shift 2 ;;
    --max-gpu-hr)  MAX_GPU_HR="$2"; shift 2 ;;
    --manual)      MANUAL=1; shift ;;        # status EXTERNAL: tracked, never auto-reaped
    --no-probe)    NO_PROBE=1; shift ;;      # skip the ssh reachability probe (non-standard boxes)
    --no-register) REGISTER=0; shift ;;      # handle only, no ledger row at all
    -h|--help)     sed -n '1,80p' "$SKILL_DIR/SKILL.md"; exit 0 ;;
    *) echo "vast-attach: unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -n "$INSTANCE_ID" ]] || { echo "vast-attach: --instance-id required" >&2; exit 2; }

# Resolve missing ssh params from the Vast API (bounded).
if [[ -z "$SSH_HOST" || -z "$SSH_PORT" || -z "$NUM_GPUS" ]]; then
  if command -v vastai >/dev/null 2>&1 && [[ -f "$SKILL_DIR/../_vast_account.sh" ]]; then
    # shellcheck disable=SC1090
    source "$SKILL_DIR/../_vast_account.sh"; vast_load_secrets
    KEY=$(vast_key_for "$(vast_account_norm "$ACCOUNT")")
    RAW=$(timeout 60 env VAST_API_KEY="$KEY" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
    if echo "$RAW" | jq -e 'type=="object"' >/dev/null 2>&1; then
      [[ -z "$SSH_HOST" ]] && SSH_HOST=$(echo "$RAW" | jq -r '.ssh_host // .public_ipaddr // empty')
      [[ -z "$SSH_PORT" ]] && SSH_PORT=$(echo "$RAW" | jq -r '.ssh_port // empty')
      [[ -z "$NUM_GPUS" ]] && NUM_GPUS=$(echo "$RAW" | jq -r '.num_gpus // empty')
      [[ -z "$GPU_NAME" ]] && GPU_NAME=$(echo "$RAW" | jq -r '.gpu_name // empty')
      { [[ -z "$DPH" || "$DPH" == "0" ]]; } && DPH=$(echo "$RAW" | jq -r '.dph_total // 0')
    fi
  fi
fi

[[ -n "$SSH_HOST" && -n "$SSH_PORT" && -n "$NUM_GPUS" ]] || {
  echo "vast-attach: need --ssh-host, --ssh-port, --num-gpus (could not resolve from the Vast API)" >&2; exit 2; }

EXP_ID="${EXP_ID:-ATTACH-$INSTANCE_ID}"
GPU_NAME="${GPU_NAME:-unknown}"
IDENTITY="${VAST_SSH_IDENTITY:-$HOME/.ssh/vast_ai_name}"
[[ "$ACCOUNT" == "team" && -f "$HOME/.ssh/Vast-Team" ]] && IDENTITY="$HOME/.ssh/Vast-Team"
SSH_LOGIN="ssh -i $IDENTITY -o StrictHostKeyChecking=accept-new -p $SSH_PORT root@$SSH_HOST"

# REACHABILITY PROBE — never hand the harness an unreachable box (bounded 30s).
if [[ "$NO_PROBE" != "1" ]]; then
  if ! timeout 30 ssh -i "$IDENTITY" -o ConnectTimeout=8 -o BatchMode=yes \
       -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST" 'true' 2>/dev/null; then
    echo "vast-attach: ssh probe FAILED for root@$SSH_HOST:$SSH_PORT (identity $IDENTITY)." >&2
    echo "vast-attach: fix connectivity or pass --no-probe to force. NOT registering." >&2
    exit 4
  fi
fi

HANDLE=$(jq -nc \
  --arg iid "$INSTANCE_ID" --arg host "$SSH_HOST" --argjson port "$SSH_PORT" \
  --argjson ng "$NUM_GPUS" --arg gn "$GPU_NAME" --argjson gr "$GPU_RAM" \
  --argjson dph "$DPH" --arg login "$SSH_LOGIN" --arg acct "$ACCOUNT" \
  --arg label "$EXP_ID" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1", instance_id:$iid, ssh_host:$host, ssh_port:$port,
    num_gpus:$ng, gpu_name:$gn, gpu_ram:$gr, dph_total:$dph, ssh_login:$login,
    label:$label, vast_account:$acct, created_at:$t, external:true}')

# Handle lands in BOTH homes: the run dir (runner contract) and the shared
# state dir (vast-teardown's account-resolution fallback scans it).
HANDLE_DIR="$PROJECT_DIR/runs/$EXP_ID/handles"
mkdir -p "$HANDLE_DIR" "$PROJECT_DIR/.claude/state/vast-handles"
echo "$HANDLE" | jq . > "$HANDLE_DIR/$INSTANCE_ID.json"
echo "$HANDLE" | jq . > "$PROJECT_DIR/.claude/state/vast-handles/$INSTANCE_ID.json"

echo "VAST_HANDLE: $HANDLE"

if [[ "$REGISTER" == "1" ]]; then
  # Dedup against LIVE rows only — a TORN_DOWN row under the same id is history,
  # not a duplicate; re-attaching after teardown must register a fresh row or the
  # new box has no budget backstop and vast-cost flags it as a leak.
  if [[ -f "$LEDGER" ]] && jq -e --arg id "$EXP_ID" \
      'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"))' \
      "$LEDGER" >/dev/null 2>&1; then
    echo "vast-attach: ledger already has a LIVE row for $EXP_ID — not duplicating." >&2
  else
    STATUS="RUNNING"; (( MANUAL )) && STATUS="EXTERNAL"
    ROW=$(jq -nc --arg id "$EXP_ID" --arg t "$(date -Iseconds)" --argjson ts "$(date +%s)" \
      --argjson ng "$NUM_GPUS" --argjson dph "$DPH" --arg acct "$ACCOUNT" \
      --argjson mgh "$MAX_GPU_HR" --arg st "$STATUS" --argjson iss "${ISSUE:-null}" \
      --argjson h "$(echo "$HANDLE" | jq -s .)" \
      '{id:$id, issue:$iss, handles:$h, started_at:$t, started_at_epoch:$ts,
        per_node_gpus:$ng, total_gpus:$ng, dph:$dph, max_gpu_hr:$mgh,
        vast_account:$acct, external:true, status:$st}')
    # Locked append (shared spinlock with _lib.sh writers).
    LOCK="$PROJECT_DIR/.claude/state/.runs.jsonl.lock"; n=0
    until mkdir "$LOCK" 2>/dev/null; do n=$((n+1)); (( n > 300 )) && break; sleep 0.1; done
    mkdir -p "$(dirname "$LEDGER")"; echo "$ROW" >> "$LEDGER"
    rmdir "$LOCK" 2>/dev/null || true
    echo "vast-attach: registered $STATUS ledger row for $EXP_ID (max_gpu_hr=$MAX_GPU_HR)." >&2
    (( MANUAL )) && echo "vast-attach: EXTERNAL = operator-managed — the reaper will NOT touch it; tear down explicitly." >&2
  fi
fi

echo "vast-attach: attached box $INSTANCE_ID (${NUM_GPUS}×$GPU_NAME, account=$ACCOUNT) as $EXP_ID." >&2
echo "vast-attach: ssh    -> $SSH_LOGIN" >&2
echo "vast-attach: handle -> runs/$EXP_ID/handles/$INSTANCE_ID.json" >&2
