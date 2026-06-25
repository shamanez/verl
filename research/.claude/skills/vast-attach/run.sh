#!/usr/bin/env bash
# vast-attach — register an ALREADY-RUNNING, operator-provided box as an EXTERNAL
# handle the harness can use WITHOUT provisioning and WITHOUT ever auto-destroying it.
# Companion to vast-provision. See SKILL.md.
#
# The box is marked external:true on BOTH the handle JSON and the ledger row, so the
# teardown Stop hook AND the vast-teardown skill refuse to destroy it (the operator
# owns its lifecycle). Output mirrors vast-provision (a VAST_HANDLE: {...} line) so the
# experiment-runner's rsync+launch path is unchanged.
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"

EXP_ID=""; INSTANCE_ID=""; SSH_HOST=""; SSH_PORT=""; NUM_GPUS=""
GPU_NAME=""; GPU_RAM="0"; DPH="0"; ACCOUNT="private"; REGISTER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id)      EXP_ID="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --ssh-host)    SSH_HOST="$2"; shift 2 ;;
    --ssh-port)    SSH_PORT="$2"; shift 2 ;;
    --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
    --gpu-name)    GPU_NAME="$2"; shift 2 ;;
    --gpu-ram)     GPU_RAM="$2"; shift 2 ;;
    --dph)         DPH="$2"; shift 2 ;;
    --account)     ACCOUNT="$2"; shift 2 ;;
    --no-register) REGISTER=0; shift ;;     # write the handle only, do NOT add a ledger row
    -h|--help)     sed -n '1,80p' "$(dirname "$0")/SKILL.md"; exit 0 ;;
    *) echo "vast-attach: unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -n "$INSTANCE_ID" ]] || { echo "vast-attach: --instance-id required (use a label for a non-Vast box)" >&2; exit 2; }

# If ssh params are missing AND this is a real Vast box, resolve them from the API.
if [[ -z "$SSH_HOST" || -z "$SSH_PORT" || -z "$NUM_GPUS" ]]; then
  if command -v vastai >/dev/null 2>&1 && [[ -f "$(dirname "$0")/../_vast_account.sh" ]]; then
    # shellcheck disable=SC1090
    source "$(dirname "$0")/../_vast_account.sh"; vast_load_secrets
    KEY=$(vast_key_for "$(vast_account_norm "$ACCOUNT")")
    RAW=$(VAST_API_KEY="$KEY" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
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
SSH_LOGIN="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p $SSH_PORT root@$SSH_HOST"

HANDLE=$(jq -nc \
  --arg iid "$INSTANCE_ID" --arg host "$SSH_HOST" --argjson port "$SSH_PORT" \
  --argjson ng "$NUM_GPUS" --arg gn "$GPU_NAME" --argjson gr "$GPU_RAM" \
  --argjson dph "$DPH" --arg login "$SSH_LOGIN" --arg acct "$ACCOUNT" --arg label "$EXP_ID" \
  '{schema_version:"1", instance_id:$iid, ssh_host:$host, ssh_port:$port,
    num_gpus:$ng, gpu_name:$gn, gpu_ram:$gr, dph_total:$dph, ssh_login:$login,
    label:$label, vast_account:$acct, external:true}')

HANDLE_DIR="$PROJECT_DIR/runs/$EXP_ID/handles"
mkdir -p "$HANDLE_DIR"
echo "$HANDLE" | jq . > "$HANDLE_DIR/$INSTANCE_ID.json"

echo "VAST_HANDLE: $HANDLE"

if [[ "$REGISTER" == "1" ]]; then
  if [[ -f "$LEDGER" ]] && jq -e --arg id "$EXP_ID" 'select(.id==$id)' "$LEDGER" >/dev/null 2>&1; then
    echo "vast-attach: ledger already has a row for $EXP_ID — not duplicating." >&2
  else
    HANDLES_ARR=$(echo "$HANDLE" | jq -s .)
    ROW=$(jq -nc --arg id "$EXP_ID" --arg t "$(date -Iseconds)" --argjson ts "$(date +%s)" \
      --argjson ng "$NUM_GPUS" --argjson dph "$DPH" --arg acct "$ACCOUNT" --argjson h "$HANDLES_ARR" \
      '{id:$id, handles:$h, started_at:$t, started_at_epoch:$ts,
        per_node_gpus:$ng, total_gpus:$ng, dph:$dph, vast_account:$acct,
        external:true, status:"RUNNING"}')
    mkdir -p "$(dirname "$LEDGER")"; echo "$ROW" >> "$LEDGER"
    echo "vast-attach: registered RUNNING+external ledger row for $EXP_ID." >&2
  fi
fi

echo "vast-attach: EXTERNAL box $INSTANCE_ID ($NUM_GPUS×$GPU_NAME, account=$ACCOUNT) attached as $EXP_ID — NEVER auto-torn-down." >&2
echo "vast-attach: ssh  -> $SSH_LOGIN" >&2
echo "vast-attach: handle -> runs/$EXP_ID/handles/$INSTANCE_ID.json" >&2
