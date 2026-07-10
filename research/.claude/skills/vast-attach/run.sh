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
# shellcheck disable=SC1090
source "$SKILL_DIR/../_seed_secrets.sh"   # seed_secrets_to_box: HF+WandB+R2 push, VAST withheld

EXP_ID=""; INSTANCE_ID=""; SSH_HOST=""; SSH_PORT=""; NUM_GPUS=""
GPU_NAME=""; GPU_RAM="0"; DPH="0"; ACCOUNT="private"; REGISTER=1
MANUAL=0; MAX_GPU_HR="24"; NO_PROBE=0; ISSUE=""; SSH_IDENTITY_FLAG=""; NEED_R2=0
SSH_LOGIN_STR=""; SYNTHETIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id)      EXP_ID="$2"; shift 2 ;;
    --issue)       ISSUE="$2"; shift 2 ;;   # issue number — REQUIRED for /launch-driven attaches (ledger_row_by_issue keys on it)
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --ssh-login)   SSH_LOGIN_STR="$2"; shift 2 ;;  # a full "ssh -i <key> -p <port> root@<host> …" string (endpoint parsed from it; id reverse-resolved best-effort)
    --ssh-host)    SSH_HOST="$2"; shift 2 ;;
    --ssh-port)    SSH_PORT="$2"; shift 2 ;;
    --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
    --gpu-name)    GPU_NAME="$2"; shift 2 ;;
    --gpu-ram)     GPU_RAM="$2"; shift 2 ;;
    --dph)         DPH="$2"; shift 2 ;;
    --account)     ACCOUNT="$2"; shift 2 ;;
    --max-gpu-hr)  MAX_GPU_HR="$2"; shift 2 ;;
    --manual)      MANUAL=1; shift ;;        # status EXTERNAL: tracked, never auto-reaped
    --ssh-identity) SSH_IDENTITY_FLAG="$2"; shift 2 ;;  # explicit key for THIS box (beats VAST_SSH_IDENTITY + team default)
    --need-r2)     NEED_R2=1; shift ;;       # preflight aws CLI + R2 creds on the box (checkpoint→R2 runs)
    --no-probe)    NO_PROBE=1; shift ;;      # skip the ssh reachability probe (non-standard boxes)
    --no-register) REGISTER=0; shift ;;      # handle only, no ledger row at all
    -h|--help)     sed -n '1,80p' "$SKILL_DIR/SKILL.md"; exit 0 ;;
    *) echo "vast-attach: unknown arg $1" >&2; exit 2 ;;
  esac
done

# Accept EITHER a bare instance-id OR a full SSH login string. The operator
# usually has a working `ssh -i <key> -p <port> root@<host> …` line in hand (the
# Vast "Direct/Proxy" connect string) — parse the endpoint straight out of it and
# probe ONCE, instead of reverse-scanning `vastai show instances` for a host:port
# match before we can do anything. A bare --instance-id that LOOKS like an ssh
# string is auto-routed here too (so `--attach "ssh …"` works either way).
if [[ -z "$SSH_LOGIN_STR" && ( "$INSTANCE_ID" == ssh\ * || "$INSTANCE_ID" == *@* ) ]]; then
  SSH_LOGIN_STR="$INSTANCE_ID"; INSTANCE_ID=""
fi

parse_ssh_login() {  # "<ssh … root@host …>" -> sets SSH_HOST SSH_PORT IDENTITY_PARSED (trailing -L/-D/-R forwards ignored)
  local -a toks; read -ra toks <<<"$1"
  local i=0 n=${#toks[@]} t
  SSH_HOST=""; SSH_PORT=""; IDENTITY_PARSED=""
  while (( i < n )); do
    t="${toks[$i]}"
    case "$t" in
      ssh)                     ;;                                # the command word
      -i)  IDENTITY_PARSED="${toks[$((i+1))]:-}"; i=$((i+1)) ;;
      -p)  SSH_PORT="${toks[$((i+1))]:-}";        i=$((i+1)) ;;
      -L|-R|-D|-o|-J|-W|-b|-c|-l|-m|-F|-E|-Q|-e)  i=$((i+1)) ;;  # option consumes its NEXT token — skip both (incl. -L/-D/-R port-forwards)
      -*)                      ;;                                # bare flag (-A -T -N -q -v -C -X …) — ignore
      *@*) SSH_HOST="${t#*@}"  ;;                                # user@host -> host
      *)   [[ -z "$SSH_HOST" ]] && SSH_HOST="$t" ;;              # bare host fallback
    esac
    i=$((i+1))
  done
  SSH_PORT="${SSH_PORT:-22}"
  IDENTITY_PARSED="${IDENTITY_PARSED/#\~/$HOME}"
}

if [[ -n "$SSH_LOGIN_STR" ]]; then
  parse_ssh_login "$SSH_LOGIN_STR"
  [[ -n "$SSH_HOST" ]] || { echo "vast-attach: could not parse a host from --ssh-login '$SSH_LOGIN_STR'" >&2; exit 2; }
  # The key named IN the login string is the operator's explicit choice for THIS
  # box — treat it as --ssh-identity (an explicit --ssh-identity flag still wins).
  [[ -z "$SSH_IDENTITY_FLAG" && -n "$IDENTITY_PARSED" ]] && SSH_IDENTITY_FLAG="$IDENTITY_PARSED"
fi

[[ -n "$INSTANCE_ID" || -n "$SSH_LOGIN_STR" ]] || {
  echo "vast-attach: need --instance-id <id> OR --ssh-login \"ssh … root@host …\"" >&2; exit 2; }

# Load the account's Vast key once (used by BOTH the id<-endpoint reverse-resolve
# and the id->endpoint forward-resolve below).
if command -v vastai >/dev/null 2>&1 && [[ -f "$SKILL_DIR/../_vast_account.sh" ]]; then
  # shellcheck disable=SC1090
  source "$SKILL_DIR/../_vast_account.sh"; vast_load_secrets
  VAST_KEY=$(vast_key_for "$(vast_account_norm "$ACCOUNT")")
fi

if [[ -n "$SSH_LOGIN_STR" && -z "$INSTANCE_ID" ]]; then
  # Best-effort REVERSE-RESOLVE the numeric Vast id from the endpoint (teardown
  # needs it): match the parsed host:port (or public IP) against the account's
  # instances. Bounded; a miss is non-fatal (synthetic fallback below).
  if [[ -n "${VAST_KEY:-}" ]]; then
    OBJ=$(timeout 60 env VAST_API_KEY="$VAST_KEY" vastai show instances --raw 2>/dev/null \
      | jq -c --arg h "$SSH_HOST" --arg p "$SSH_PORT" \
          'if type=="array" then . else [.] end
           | map(select(((.ssh_host // "")==$h and ((.ssh_port // ""|tostring)==$p))
                        or ((.public_ipaddr // "")==$h))) | first // empty' 2>/dev/null || true)
    if [[ -n "${OBJ:-}" ]]; then
      INSTANCE_ID=$(echo "$OBJ" | jq -r '.id // empty')
      [[ -z "$NUM_GPUS" ]] && NUM_GPUS=$(echo "$OBJ" | jq -r '.num_gpus // empty')
      [[ -z "$GPU_NAME" ]] && GPU_NAME=$(echo "$OBJ" | jq -r '.gpu_name // empty')
      { [[ -z "$DPH" || "$DPH" == "0" ]]; } && DPH=$(echo "$OBJ" | jq -r '.dph_total // 0')
    fi
  fi
  if [[ -z "$INSTANCE_ID" ]]; then
    # Could not resolve — register with a SYNTHETIC id so the box is still tracked
    # + probed. Teardown then DEGRADES gracefully: the reaper and vast-teardown
    # both SKIP a non-numeric id (they cannot `vastai destroy` it) and surface it
    # for MANUAL teardown, rather than risk a false "already-gone" TORN_DOWN flip.
    SYNTHETIC=1
    INSTANCE_ID="ATTACH-${SSH_HOST}-${SSH_PORT}"
    NUM_GPUS="${NUM_GPUS:-1}"
    echo "vast-attach: WARN could not reverse-resolve a Vast instance-id for ${SSH_HOST}:${SSH_PORT} (account=$ACCOUNT)." >&2
    echo "vast-attach:   -> registering synthetic id '$INSTANCE_ID'; AUTO-TEARDOWN IS DISABLED for this row." >&2
    echo "vast-attach:   -> tear the box down by hand (or re-attach with --instance-id) when done." >&2
  fi
elif [[ -z "$SSH_HOST" || -z "$SSH_PORT" || -z "$NUM_GPUS" ]]; then
  # Bare instance-id path: FORWARD-resolve missing ssh/gpu params from the API.
  if [[ -n "${VAST_KEY:-}" ]]; then
    RAW=$(timeout 60 env VAST_API_KEY="$VAST_KEY" vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
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
  echo "vast-attach: need host+port+num-gpus — parse them from --ssh-login, or pass --ssh-host/--ssh-port/--num-gpus" >&2; exit 2; }

EXP_ID="${EXP_ID:-ATTACH-$INSTANCE_ID}"
GPU_NAME="${GPU_NAME:-unknown}"
# SSH identity precedence (#63 B14): --ssh-identity > VAST_SSH_IDENTITY env >
# team-convention key (only when NOTHING explicit was given) > project default.
# The API account (teardown auth) and the ssh key are INDEPENDENT — an operator's
# team box may use any key; never couple the two.
IDENTITY="${SSH_IDENTITY_FLAG:-${VAST_SSH_IDENTITY:-}}"
if [[ -z "$IDENTITY" ]]; then
  IDENTITY="$HOME/.ssh/vast_ai_name"
  [[ "$ACCOUNT" == "team" && -f "$HOME/.ssh/Vast-Team" ]] && IDENTITY="$HOME/.ssh/Vast-Team"
fi
SSH_LOGIN="ssh -i $IDENTITY -o StrictHostKeyChecking=accept-new -p $SSH_PORT root@$SSH_HOST"

# REACHABILITY PROBE — never hand the harness an unreachable box (bounded 30s).
if [[ "$NO_PROBE" != "1" ]]; then
  if ! timeout 30 ssh -i "$IDENTITY" -o ConnectTimeout=8 -o BatchMode=yes \
       -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST" 'true' 2>/dev/null; then
    echo "vast-attach: ssh probe FAILED for root@$SSH_HOST:$SSH_PORT (identity $IDENTITY)." >&2
    echo "vast-attach: fix connectivity or pass --no-probe to force. NOT registering." >&2
    exit 4
  fi
  # Box is reachable — seed HF+WandB+R2 secrets NOW (deterministic; no agent
  # step), BEFORE the --need-r2 preflight so that preflight validates the creds
  # we just pushed. Best-effort: a failure WARNs but never blocks the attach.
  seed_secrets_to_box "$SSH_HOST" "$SSH_PORT" "$IDENTITY" || true
else
  echo "vast-attach: NOTE --no-probe set — secrets NOT auto-seeded; ensure /root/.config/verl-research/secrets.env exists on the box by hand." >&2
fi

# R2-dependency preflight (#63 B11, hardened B5 2026-07-10): checkpoint→R2 fails
# at the step-100 SAVE — mid-run, killing the cell — when the box lacks the aws
# CLI, lacks/has-wrong R2 creds, the bucket is unwritable, OR R2_BUCKET mismatches
# the verl code guard (r2_sink.py R2_REQUIRED_BUCKET). A `command -v aws` presence
# check misses ALL of the last three. So do a REAL write test + a guard-match check.
# Surface HERE, at attach time. WARN-only (the operator may fix later), but LOUD.
if (( NEED_R2 )) && [[ "$NO_PROBE" != "1" ]]; then
  R2CHK=$(timeout 60 ssh -i "$IDENTITY" -o ConnectTimeout=8 -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST" '
      command -v aws >/dev/null && echo AWS_OK || echo AWS_MISSING
      S="$HOME/.config/verl-research/secrets.env"
      grep -qE "^(export )?R2_ACCESS_KEY_ID=" "$S" 2>/dev/null && echo R2CREDS_OK || echo R2CREDS_MISSING
      source "$S" 2>/dev/null
      echo "R2_BUCKET=${R2_BUCKET:-<unset>}"
      # code guard: the ckpt sink refuses any bucket != R2_REQUIRED_BUCKET
      G=$(grep -hoE "R2_REQUIRED_BUCKET *= *\"[^\"]+\"" /workspace/verl/verl/workers/comm_eff/r2_sink.py 2>/dev/null | grep -oE "\"[^\"]+\"" | tr -d \")
      [ -n "$G" ] && { [ "$G" = "${R2_BUCKET:-}" ] && echo "GUARD_MATCH" || echo "GUARD_MISMATCH(code=$G)"; } || echo "GUARD_UNKNOWN(no checkout yet)"
      # REAL write test to R2_BUCKET via the exact path the sink uses (aws s3 cp)
      if command -v aws >/dev/null && [ -n "${R2_BUCKET:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
        EP="${R2_ENDPOINT:-https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com}"
        echo t > /tmp/_r2pf.txt
        AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
          aws s3 cp /tmp/_r2pf.txt "s3://$R2_BUCKET/autonomous-harness-rlvr-compression/_attach_writetest.txt" --endpoint-url "$EP" >/dev/null 2>&1 \
          && echo R2_WRITE_OK || echo R2_WRITE_FAIL
      else echo R2_WRITE_SKIPPED; fi
      ' 2>/dev/null || echo "R2CHK_UNREACHABLE")
  if echo "$R2CHK" | grep -qE "MISSING|UNREACHABLE|GUARD_MISMATCH|R2_WRITE_FAIL"; then
    echo "vast-attach: WARN --need-r2 preflight FAILED: $(echo "$R2CHK" | tr '\n' ' ')" >&2
    echo "vast-attach:   -> fix BEFORE the run reaches a checkpoint save (step SAVE_FREQ): install aws (pip install awscli), set R2_BUCKET to the code-guard bucket, verify creds/writability. A save crash kills the cell mid-run." >&2
  else
    echo "vast-attach: --need-r2 preflight OK: $(echo "$R2CHK" | tr '\n' ' ')" >&2
  fi
elif (( NEED_R2 )); then
  # --no-probe silences the preflight; say so — silence must not read as "clean" (#63 B11 review).
  echo "vast-attach: NOTE --need-r2 requested but --no-probe set — R2 preflight NOT run; verify aws CLI + R2 creds on the box by hand." >&2
fi

HANDLE=$(jq -nc \
  --arg iid "$INSTANCE_ID" --arg host "$SSH_HOST" --argjson port "$SSH_PORT" \
  --argjson ng "$NUM_GPUS" --arg gn "$GPU_NAME" --argjson gr "$GPU_RAM" \
  --argjson dph "$DPH" --arg login "$SSH_LOGIN" --arg acct "$ACCOUNT" \
  --arg label "$EXP_ID" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg ident "$IDENTITY" --argjson synth "$SYNTHETIC" \
  '{schema_version:"1", instance_id:$iid, ssh_host:$host, ssh_port:$port,
    num_gpus:$ng, gpu_name:$gn, gpu_ram:$gr, dph_total:$dph, ssh_login:$login,
    ssh_identity:$ident, label:$label, vast_account:$acct, created_at:$t,
    external:true, synthetic_instance_id:($synth==1)}')

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
      --argjson synth "$SYNTHETIC" \
      --argjson h "$(echo "$HANDLE" | jq -s .)" \
      '{id:$id, issue:$iss, handles:$h, started_at:$t, started_at_epoch:$ts,
        per_node_gpus:$ng, total_gpus:$ng, dph:$dph, max_gpu_hr:$mgh,
        vast_account:$acct, external:true, status:$st,
        synthetic_instance_id:($synth==1)}')
    # Locked append (shared spinlock with _lib.sh writers).
    LOCK="$PROJECT_DIR/.claude/state/.runs.jsonl.lock"; n=0; LOCKED=1
    mkdir -p "$(dirname "$LEDGER")"
    # Steal a stale lock (>300s = crashed holder), mirroring _lib.sh, so a dead
    # session can't wedge us for 30s into the UNLOCKED append below — that path
    # let a concurrent whole-file rewrite silently drop this live box's row.
    until mkdir "$LOCK" 2>/dev/null; do
      AGE=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
      (( AGE > 300 )) && { rmdir "$LOCK" 2>/dev/null || true; continue; }
      n=$((n+1)); (( n > 300 )) && { LOCKED=0; break; }; sleep 0.1
    done
    echo "$ROW" >> "$LEDGER"
    if (( LOCKED )); then
      rmdir "$LOCK" 2>/dev/null || true
    else
      echo "vast-attach: WARN registered $EXP_ID WITHOUT the ledger lock (30s contention) — a concurrent rewrite may race it; verify with: jq -c 'select(.id==\"$EXP_ID\")' $LEDGER" >&2
    fi
    echo "vast-attach: registered $STATUS ledger row for $EXP_ID (max_gpu_hr=$MAX_GPU_HR)." >&2
    (( MANUAL )) && echo "vast-attach: EXTERNAL = operator-managed — the reaper will NOT touch it; tear down explicitly." >&2
  fi
fi

echo "vast-attach: attached box $INSTANCE_ID (${NUM_GPUS}×$GPU_NAME, account=$ACCOUNT) as $EXP_ID." >&2
echo "vast-attach: ssh    -> $SSH_LOGIN" >&2
echo "vast-attach: handle -> runs/$EXP_ID/handles/$INSTANCE_ID.json" >&2
(( SYNTHETIC )) && echo "vast-attach: NOTE synthetic instance-id ('$INSTANCE_ID') — auto-teardown DISABLED; tear this box down by hand when done." >&2
