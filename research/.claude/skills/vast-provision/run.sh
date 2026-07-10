#!/usr/bin/env bash
# vast-provision — provision the cheapest qualifying vast.ai instance(s),
# wait for SSH-routable state, emit a handle JSON per instance.
#
# See SKILL.md for the full contract. Output:
#   VAST_HANDLE: <json>   (one per instance, stdout)
#   VAST_PROVISIONED: count=<N> total_dph=<X.XXXX>
#
# Companion: vast-teardown reads the same schema_version="1" handle JSON.
set -euo pipefail

PROG="vast-provision"
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

# macOS has no timeout(1); perl alarm survives execve (same shim as _lib.sh).
command -v timeout >/dev/null 2>&1 || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

# Every vastai CLI call in this script is hard-bounded — a hung API call must
# never hang the skill. Defined BEFORE any call site (incl. the ssh-keys
# preflight). Inside the function, `timeout` resolves to the binary or the shim
# above; the exec'd vastai binary can never re-enter this shell function.
vastai() { timeout "${VAST_CLI_TIMEOUT:-120}" "$(command -v vastai)" "$@"; }

# CLAUDE_PROJECT_DIR resolves to the research/ directory the harness was
# launched from; fall back to walking three levels up from this script.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SKILL_DIR/../../.." && pwd)}"

SCHEMA_VERSION="1"
DEFAULT_HANDLE_DIR="${VERL_VAST_HANDLE_DIR:-$PROJECT_DIR/.claude/state/vast-handles}"

# ---- defaults --------------------------------------------------------------
# Tuned for agent-self-sufficient invocation (Claude /vast-provision with no
# explicit budget): safe per-instance cost, modest disk, generous wait for the
# verlai image pull + onstart script. Larger multi-GPU plans (4×H200 /
# 8×H100 shapes, explicit operator request only) pass explicit overrides — see
# the per-experiment plan (GitHub issue body; local cache .claude/state/plan-cache/<N>.md).
COUNT=1
QUERY=""
IMAGE=""
DISK_GB=200                  # see SKILL.md "Disk sizing" — covers image + model + dataset + checkpoints for a typical research run
MAX_PRICE="1.0"              # single-GPU ad-hoc safety; bump for multi-GPU production
MIN_RELIABILITY="0.99"       # default floor — the project ladder requires reliability strictly >0.99 on every rung (project.yaml default_compute)
GPU_COUNT_FILTER=""          # optional per-host GPU sanity check
EXTRA_ENV=""                 # vastai --env raw string; off by default
GHCR_LOGIN=""
ONSTART_CMD=""
TIMEOUT=1500                 # 25 min: covers ~30 GB verlai image pull on slow hosts
POLL_INTERVAL=15             # one less API call per minute vs the old 10s
HANDLE_DIR="$DEFAULT_HANDLE_DIR"
LABEL_PREFIX="verl-research"
SESSION_ID=""
TEMPLATE_HASH=""             # optional vast.ai Template hash_id
NO_DEFAULT_FILTERS=false
DRY_RUN=false

# SSH identity offered to the provisioned box (per project.yaml vast_ssh).
# Override with VAST_SSH_IDENTITY to rotate keys without editing this file.
# Both ~/.ssh/vast_ai_name (id 890294) and the older ~/.ssh/vast_ai (id 835115)
# are registered on the account, so the box accepts either; we OFFER this one.
SSH_IDENTITY="${VAST_SSH_IDENTITY:-~/.ssh/vast_ai_name}"

# ---- usage ----------------------------------------------------------------
usage() {
  if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    sed -n '1,160p' "$SKILL_DIR/SKILL.md"
  else
    echo "$PROG: SKILL.md missing — see source comments." >&2
  fi
  exit "${1:-0}"
}

# ---- arg parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --count|-n)            COUNT="$2"; shift 2 ;;
    --gpu-count)           GPU_COUNT_FILTER="$2"; shift 2 ;;
    --query|-q)            QUERY="$2"; shift 2 ;;
    --image|-i)            IMAGE="$2"; shift 2 ;;
    --disk-gb|--disk)      DISK_GB="$2"; shift 2 ;;
    --max-price)           MAX_PRICE="$2"; shift 2 ;;
    --min-reliability)     MIN_RELIABILITY="$2"; shift 2 ;;
    --env)                 EXTRA_ENV="$2"; shift 2 ;;
    --login)               GHCR_LOGIN="$2"; shift 2 ;;
    --onstart-cmd)         ONSTART_CMD="$2"; shift 2 ;;
    --timeout)             TIMEOUT="$2"; shift 2 ;;
    --poll-interval)       POLL_INTERVAL="$2"; shift 2 ;;
    --handle-dir)          HANDLE_DIR="$2"; shift 2 ;;
    --session-id)          SESSION_ID="$2"; shift 2 ;;
    --label)               LABEL_OVERRIDE="$2"; shift 2 ;;   # exact instance label (run id) — beats --label-prefix
    --label-prefix)        LABEL_PREFIX="$2"; shift 2 ;;
    --template-hash)       TEMPLATE_HASH="$2"; shift 2 ;;
    --no-default-filters)  NO_DEFAULT_FILTERS=true; shift ;;
    --dry-run)             DRY_RUN=true; shift ;;
    -h|--help)             usage 0 ;;
    *) echo "$PROG: unknown argument: $1" >&2; usage 2 ;;
  esac
done

# ---- auto-default template ------------------------------------------------
# If neither --image nor --template-hash was passed, pick the active research
# template from templates.json (the single source of truth for what we
# provision). This is what makes "skill is the single source of truth" work
# without every agent having to know the hash: agents call the skill with
# just --query + --max-price + --count, and the locked image / onstart /
# disk-defaults all come from the template record on Vast.ai.
TEMPLATES_JSON="$SKILL_DIR/templates.json"
if [[ -z "$IMAGE" && -z "$TEMPLATE_HASH" && -r "$TEMPLATES_JSON" ]]; then
  NUM_TEMPLATES=$(jq 'length' "$TEMPLATES_JSON" 2>/dev/null || echo 0)
  if [[ "$NUM_TEMPLATES" == "1" ]]; then
    TEMPLATE_NAME=$(jq -r 'keys[0]' "$TEMPLATES_JSON")
    TEMPLATE_IMAGE=$(jq -r '.[keys[0]].image // "?"' "$TEMPLATES_JSON")
    if [[ "${VAST_ACCOUNT:-team}" == "team" ]]; then
      # A Vast Template is owned by ONE account; the private-owned template is NOT
      # accessible to the team account (create 400s). Use the recorded team-owned
      # copy (team_hash_id) when provisioning on the team account.
      TEMPLATE_HASH=$(jq -r '.[keys[0]].team_hash_id // empty' "$TEMPLATES_JSON")
      if [[ -n "$TEMPLATE_HASH" ]]; then
        echo "$PROG: auto-selected TEAM template '$TEMPLATE_NAME' hash=$TEMPLATE_HASH image=$TEMPLATE_IMAGE (VAST_ACCOUNT=team)" >&2
      else
        # FAIL FAST: the private hash is guaranteed to 400 on the team account —
        # submitting a doomed create just wastes a round-trip and muddies the log.
        echo "$PROG: WARNING VAST_ACCOUNT=team but templates.json has no team_hash_id for '$TEMPLATE_NAME'." >&2
        echo "$PROG: MANUAL_REVIEW unrecoverable create error: record a team-owned template copy (SKILL.md 'Team-account templates') before provisioning on the team account." >&2
        exit 4
      fi
    else
      TEMPLATE_HASH=$(jq -r '.[keys[0]].hash_id' "$TEMPLATES_JSON")
      echo "$PROG: auto-selected template '$TEMPLATE_NAME' hash=$TEMPLATE_HASH image=$TEMPLATE_IMAGE" >&2
    fi
    echo "$PROG: (pass --template-hash explicitly to override, or --image to bypass the template entirely)" >&2
  elif [[ "$NUM_TEMPLATES" -gt 1 ]]; then
    echo "$PROG: templates.json has $NUM_TEMPLATES entries; pass --template-hash explicitly to pick one" >&2
    jq -r 'to_entries[] | "  - \(.key): \(.value.hash_id)"' "$TEMPLATES_JSON" >&2
    exit 2
  fi
fi

# ---- validation -----------------------------------------------------------
[[ -z "$QUERY" ]] && { echo "$PROG: --query is required" >&2; exit 2; }
# --image is required UNLESS a --template-hash is provided (the template carries the image)
if [[ -z "$IMAGE" && -z "$TEMPLATE_HASH" ]]; then
  echo "$PROG: pass --image, or pass --template-hash to inherit it from a vast.ai Template" >&2
  echo "$PROG: (templates.json is empty or missing — add a template record to enable auto-default)" >&2
  exit 2
fi
[[ "$COUNT" =~ ^[0-9]+$ && "$COUNT" -ge 1 ]] || { echo "$PROG: --count must be a positive integer" >&2; exit 2; }
[[ "$DISK_GB" =~ ^[0-9]+$ && "$DISK_GB" -ge 1 ]] || { echo "$PROG: --disk-gb must be a positive integer" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "$PROG: --timeout must be an integer" >&2; exit 2; }
[[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] || { echo "$PROG: --poll-interval must be an integer" >&2; exit 2; }

# ---- auth -----------------------------------------------------------------
# Agent self-sufficiency: if VAST_API_KEY isn't already in env, source the
# canonical secrets file. The file is chmod 600 in the user's own home and
# is the documented single store; this lets `/vast-provision` work from any
# fresh Claude session without an explicit `source` step.
SECRETS_FILE="${VERL_SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
# Resolve the Vast.ai account + API key (team vs private) via the shared resolver.
# VAST_ACCOUNT=team bills the shared "Pluralis Research" team account
# (VAST_API_KEY_TEAM); default/private uses the personal VAST_API_KEY. The chosen
# account is stamped on the handle (vast_account) so teardown auths against it.
# shellcheck disable=SC1090
source "$SKILL_DIR/../_vast_account.sh"
# shellcheck disable=SC1090
source "$SKILL_DIR/../_seed_secrets.sh"   # seed_secrets_to_box: HF+WandB+R2 push, VAST withheld
vast_load_secrets
VAST_ACCOUNT="$(vast_account_norm "${VAST_ACCOUNT:-team}")"
VAST_API_KEY="$(vast_key_for "$VAST_ACCOUNT")"; export VAST_API_KEY
if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "$PROG: no Vast API key for account=$VAST_ACCOUNT (set VAST_API_KEY / VAST_API_KEY_TEAM in $SECRETS_FILE)" >&2
  exit 1
fi
echo "$PROG: auth: using VAST_API_KEY (account=$VAST_ACCOUNT)" >&2

command -v vastai >/dev/null || { echo "$PROG: 'vastai' CLI not on PATH — pip install vastai" >&2; exit 1; }
command -v jq    >/dev/null || { echo "$PROG: 'jq' not on PATH" >&2; exit 1; }

# ---- pre-flight: ssh key available? ---------------------------------------
# A `--ssh` box is reachable only via an attached key. PRIVATE accounts use an
# account-level uploaded key (Vast auto-attaches it to every create). TEAM
# accounts CANNOT hold account-level keys ("SSH keys can only be created in
# personal context"), so for team we attach the harness key per-instance right
# after create (see create loop) — here we only verify a local public key exists.
if [[ "$VAST_ACCOUNT" == "team" ]]; then
  if [[ ! -r "$HOME/.ssh/Vast-Team.pub" && ! -r "$HOME/.ssh/vast_ai_name.pub" && ! -r "$HOME/.ssh/vast_ai.pub" ]]; then
    echo "$PROG: VAST_ACCOUNT=team needs a local harness public key to attach per-instance" >&2
    echo "$PROG: (~/.ssh/Vast-Team.pub, vast_ai_name.pub, or vast_ai.pub) — none readable." >&2
    exit 1
  fi
else
  # vastai create instance --ssh creates a box accessible only via keys the
  # user has uploaded to https://cloud.vast.ai/account/. With zero uploaded
  # keys, the instance comes up unreachable; we'd discover that only after
  # the wait_for_ready timeout, ~25 min and several cents in.
  SSH_KEYS_RAW=$(vastai show ssh-keys 2>/dev/null || true)
  if ! grep -q "ssh-" <<<"$SSH_KEYS_RAW"; then
    echo "$PROG: no SSH key uploaded to your vast.ai account — provisioning would yield an unreachable box." >&2
    echo "$PROG: upload one via: vastai create ssh-key \"\$(cat ~/.ssh/id_ed25519.pub)\"" >&2
    echo "$PROG: or via the console: https://cloud.vast.ai/account/" >&2
    exit 1
  fi
fi

# ---- session id / label / handle dir --------------------------------------
if [[ -z "$SESSION_ID" ]]; then
  if command -v uuidgen >/dev/null; then
    SESSION_ID="$(uuidgen | tr 'A-Z' 'a-z')"
  else
    SESSION_ID="$(date +%s)-$$"
  fi
fi
LABEL="${LABEL_OVERRIDE:-${LABEL_PREFIX}:${SESSION_ID}}"
mkdir -p "$HANDLE_DIR"

# ---- search offers --------------------------------------------------------
SEARCH_CMD=(vastai search offers "$QUERY" -o dph_total --raw)
$NO_DEFAULT_FILTERS && SEARCH_CMD+=(-n)

echo "$PROG: searching offers query=\"$QUERY\" max_price=$MAX_PRICE min_reliability=$MIN_RELIABILITY count=$COUNT" >&2
SEARCH_ERR="$(mktemp)"
trap 'rm -f "$SEARCH_ERR"' EXIT

RAW_OFFERS="$("${SEARCH_CMD[@]}" 2>"$SEARCH_ERR")" || {
  echo "$PROG: vastai search offers failed" >&2
  head -20 "$SEARCH_ERR" >&2 || true
  exit 1
}

if ! echo "$RAW_OFFERS" | jq -e 'type=="array"' >/dev/null 2>&1; then
  echo "$PROG: vastai search offers did not return a JSON array (was the CLI updated?)" >&2
  head -20 "$SEARCH_ERR" >&2 || true
  exit 1
fi

NUM_OFFERS=$(echo "$RAW_OFFERS" | jq 'length')
echo "$PROG: $NUM_OFFERS raw offers returned" >&2

# ---- filter (price + reliability + optional num_gpus) ---------------------
# Price precedence mirrors the legacy streams helper:
#   dph_total → totalHour → dph_base → search.totalHour → search.discountedTotalPerHour
PRICE_EXPR='(.dph_total // .totalHour // .dph_base // (.search.totalHour // .search.discountedTotalPerHour))'

FILTER_JQ="
  map(
    select(
      ${PRICE_EXPR} != null
      and (${PRICE_EXPR} | tonumber) <= (\$mp|tonumber)
      and ((.reliability2 // 0) | tonumber) >= (\$mr|tonumber)
"
# We always 'create --direct', so require the host to actually have a direct
# port available (direct_port_count>=1). Without this the cheapest offer can be a
# host that can never give a direct route — you silently fall to the slower proxy
# path the poll warns about. Skipped under --no-default-filters.
if ! $NO_DEFAULT_FILTERS; then
  FILTER_JQ+="
      and ((.direct_port_count // 0) | tonumber) >= 1
  "
fi
if [[ -n "$GPU_COUNT_FILTER" ]]; then
  [[ "$GPU_COUNT_FILTER" =~ ^[0-9]+$ ]] || { echo "$PROG: --gpu-count must be a non-negative integer" >&2; exit 2; }
  FILTER_JQ+="
      and ((.num_gpus // 0) | tonumber) == (\$ng|tonumber)
  "
fi
FILTER_JQ+="
    )
  )
"

FILTERED=$(echo "$RAW_OFFERS" | jq \
  --arg mp "$MAX_PRICE" \
  --arg mr "$MIN_RELIABILITY" \
  --arg ng "${GPU_COUNT_FILTER:-0}" \
  "$FILTER_JQ")

NUM_FILTERED=$(echo "$FILTERED" | jq 'length')
echo "$PROG: $NUM_FILTERED offers qualify after filters" >&2

if [[ "$NUM_FILTERED" -lt "$COUNT" ]]; then
  CHEAPEST_REJECTED=$(echo "$RAW_OFFERS" | jq -c "
    map(select(${PRICE_EXPR} != null))
    | sort_by(${PRICE_EXPR} | tonumber)
    | .[0] // null
  ")
  if [[ "$NUM_FILTERED" -eq 0 ]]; then
    if [[ "$CHEAPEST_REJECTED" != "null" ]]; then
      REJ_PRICE=$(echo "$CHEAPEST_REJECTED" | jq -r "${PRICE_EXPR} // \"?\"")
      REJ_ID=$(echo    "$CHEAPEST_REJECTED" | jq -r '.id // "?"')
      REJ_REL=$(echo   "$CHEAPEST_REJECTED" | jq -r '.reliability2 // "?"')
      REJ_GPU=$(echo   "$CHEAPEST_REJECTED" | jq -r '.gpu_name // "?"')
      REJ_NGPU=$(echo  "$CHEAPEST_REJECTED" | jq -r '.num_gpus // "?"')
      echo "$PROG: NO_OFFERS no offers qualify under max_price=$MAX_PRICE min_reliability=$MIN_RELIABILITY${GPU_COUNT_FILTER:+ num_gpus=$GPU_COUNT_FILTER}; cheapest rejected: id=$REJ_ID dph_total=$REJ_PRICE reliability2=$REJ_REL gpu_name=$REJ_GPU num_gpus=$REJ_NGPU" >&2
    else
      echo "$PROG: NO_OFFERS vastai returned zero offers for query=\"$QUERY\" (no price field)" >&2
    fi
    exit 3
  fi
  echo "$PROG: only $NUM_FILTERED qualifying offers; --count=$COUNT requested. Loosen --max-price ($MAX_PRICE), --min-reliability ($MIN_RELIABILITY), or your query." >&2
  exit 3
fi

# ---- pick a candidate POOL (COUNT + spares), cheapest first, distinct by host_id ----
# We take more than COUNT so the create loop can DESTROY an unreachable/failed
# host and advance to the next cheapest candidate instead of aborting the run.
POOL_SIZE=$(( COUNT + 8 ))
CHOSEN=$(echo "$FILTERED" | jq --argjson n "$POOL_SIZE" "
  sort_by(${PRICE_EXPR} | tonumber)
  | reduce .[] as \$o ([];
      if any(.[]; (.host_id // .id) == (\$o.host_id // \$o.id))
      then .
      else . + [\$o]
      end)
  | .[0:\$n]
")
NUM_CHOSEN=$(echo "$CHOSEN" | jq 'length')
if [[ "$NUM_CHOSEN" -lt "$COUNT" ]]; then
  echo "$PROG: only $NUM_CHOSEN distinct hosts after de-duplication; need --count=$COUNT" >&2
  exit 3
fi
echo "$PROG: candidate pool: $NUM_CHOSEN distinct hosts (provisioning $COUNT; unreachable/failed hosts are destroyed and skipped)" >&2

# ---- dry run --------------------------------------------------------------
if $DRY_RUN; then
  while IFS= read -r offer; do
    OID=$(echo    "$offer" | jq -r '.id')
    OPRICE=$(echo "$offer" | jq -r "${PRICE_EXPR}")
    OGPU=$(echo   "$offer" | jq -r '.gpu_name // ""')
    ONUMG=$(echo  "$offer" | jq -r '.num_gpus // 0')
    OREL=$(echo   "$offer" | jq -r '.reliability2 // "?"')
    echo "$PROG: dry-run offer id=$OID dph_total=$OPRICE gpu_name=$OGPU num_gpus=$ONUMG reliability2=$OREL" >&2
    # Build the same argv the actual create path uses, so dry-run is a faithful preview.
    CMD=(vastai create instance "$OID" --disk "$DISK_GB" --ssh --direct --cancel-unavail --label "$LABEL")
    [[ -n "$TEMPLATE_HASH" ]] && CMD+=(--template_hash "$TEMPLATE_HASH")
    [[ -n "$IMAGE" ]]         && CMD+=(--image "$IMAGE")
    [[ -n "$EXTRA_ENV" ]]     && CMD+=(--env "$EXTRA_ENV")
    [[ -n "$GHCR_LOGIN" ]]    && CMD+=(--login "***REDACTED***")
    [[ -n "$ONSTART_CMD" ]]   && CMD+=(--onstart-cmd "<inline ${#ONSTART_CMD}-byte script>")
    echo "dry-run vastai command: ${CMD[*]}"
  done < <(echo "$CHOSEN" | jq -c '.[]')
  exit 0
fi

# ---- budget announcement (cost-control gate) ------------------------------
# Even when the caller has Bash globally allowlisted, this line lands in the
# session log BEFORE any `vastai create instance` runs. A reviewer (or the
# user reading the transcript later) can see the upper bound on spend.
MAX_TOTAL_DPH=$(awk -v c="$COUNT" -v p="$MAX_PRICE" 'BEGIN { printf "%.4f", c * p }')
echo "$PROG: BUDGET: about to create $COUNT instance(s) at up to \$${MAX_PRICE}/hr each (ceiling: \$${MAX_TOTAL_DPH}/hr aggregate). Pass --dry-run to inspect without spend." >&2

# ---- ssh-reachability probe + orphan-cleanup helpers ----------------------
# A mapped 22/tcp port is NECESSARY but NOT SUFFICIENT: sshd may not be up,
# a host firewall may block the direct port, the proxy may not be forwarding
# yet, or (team) the per-instance key attach may not have landed. So we run a
# REAL ssh command before declaring success. And every instance we create is
# tracked until verified, so any failure (timeout, host error, failed probe,
# set -e on an unguarded command, or a signal) destroys the orphan instead of
# leaving a billing leak — the #1 money-leak class this skill must own, because
# provision is the ONLY component that knows the instance id before a ledger row.
SSH_IDENTITY_PATH="${SSH_IDENTITY/#\~/$HOME}"   # expand ~ for programmatic ssh
ACTIVE_UNVERIFIED=()

_sleep_quiet() { sleep "$1"; }

_destroy_instance() {
  local iid="$1"
  [[ -z "$iid" ]] && return 0
  if VAST_API_KEY="$VAST_API_KEY" vastai destroy instance "$iid" -y >/dev/null 2>&1; then
    echo "$PROG: destroyed instance $iid (account=$VAST_ACCOUNT)" >&2
  else
    echo "$PROG: WARNING could not destroy $iid — verify with: vastai show instances (account=$VAST_ACCOUNT)" >&2
  fi
}

_drop_unverified() {   # remove an id from ACTIVE_UNVERIFIED once it is verified
  local keep=() x
  for x in "${ACTIVE_UNVERIFIED[@]:-}"; do
    [[ -z "$x" || "$x" == "$1" ]] && continue
    keep+=("$x")
  done
  if [[ ${#keep[@]} -gt 0 ]]; then ACTIVE_UNVERIFIED=("${keep[@]}"); else ACTIVE_UNVERIFIED=(); fi
}

_on_exit() {           # EXIT trap: never leave a created-but-unverified box billing
  local x
  for x in "${ACTIVE_UNVERIFIED[@]:-}"; do
    [[ -z "$x" ]] && continue
    echo "$PROG: cleanup — destroying unverified instance $x to avoid a billing leak" >&2
    _destroy_instance "$x"
  done
  ACTIVE_UNVERIFIED=()
  rm -f "${SEARCH_ERR:-}" "${CREATE_ERR:-}" 2>/dev/null || true
}
trap '_on_exit' EXIT

_ssh_probe() {         # host port -> 0 iff a REAL ssh command succeeds with the offered key
  local host="$1" port="$2" i
  for i in 1 2 3 4 5; do
    # timeout 30 bounds a connect-then-wedge sshd; ConnectTimeout alone only
    # bounds the TCP handshake, not a hung remote command.
    if timeout 30 ssh -n -i "$SSH_IDENTITY_PATH" \
           -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
           -o BatchMode=yes -o ConnectTimeout=10 \
           -p "$port" "root@$host" true >/dev/null 2>&1; then
      return 0
    fi
    _sleep_quiet "$POLL_INTERVAL"
  done
  return 1
}

# Some create failures are UNRECOVERABLE across every candidate (the template is
# not visible to this account, or the API key is bad) — looping the whole pool
# just wastes time, so detect these and abort with an actionable message.
_unrecoverable_create_err() {   # blob -> 0 iff the error is account/template-fatal
  grep -qiE 'not accessible by user|invalid template hash|template not (accessible|found)|unauthorized|invalid api key|\b40[13]\b' <<<"${1:-}"
}

# ---- create + wait loop ---------------------------------------------------
# Iterate the candidate pool (COUNT + spares). On ANY per-offer failure we
# destroy that instance and advance to the NEXT cheapest candidate, only failing
# the whole run when the pool is exhausted before COUNT boxes are SSH-verified.
TOTAL_DPH="0"
PROVISIONED=0

while IFS= read -r offer; do
  [[ "$PROVISIONED" -ge "$COUNT" ]] && break
  OID=$(echo    "$offer" | jq -r '.id')
  OPRICE=$(echo "$offer" | jq -r "${PRICE_EXPR}")
  OGPU=$(echo   "$offer" | jq -r '.gpu_name // ""')
  ONUMG=$(echo  "$offer" | jq -r '.num_gpus // 0')

  CREATE_CMD=(vastai create instance "$OID"
              --disk "$DISK_GB"
              --ssh --direct --cancel-unavail
              --label "$LABEL"
              --raw)
  # Image OR template hash. If both, request-level --image overrides per
  # vast.ai's documented merge semantics (scalar request wins over template).
  [[ -n "$TEMPLATE_HASH" ]] && CREATE_CMD+=(--template_hash "$TEMPLATE_HASH")
  [[ -n "$IMAGE" ]]         && CREATE_CMD+=(--image "$IMAGE")
  [[ -n "$EXTRA_ENV" ]]     && CREATE_CMD+=(--env "$EXTRA_ENV")
  [[ -n "$GHCR_LOGIN" ]]    && CREATE_CMD+=(--login "$GHCR_LOGIN")
  [[ -n "$ONSTART_CMD" ]]   && CREATE_CMD+=(--onstart-cmd "$ONSTART_CMD")

  echo "$PROG: creating instance from offer $OID (dph=$OPRICE gpu=$OGPU num_gpus=$ONUMG)" >&2
  CREATE_ERR="$(mktemp)"
  if ! CREATE_OUT="$("${CREATE_CMD[@]}" 2>"$CREATE_ERR")"; then
    CREATE_BLOB="$(cat "$CREATE_ERR" 2>/dev/null)${CREATE_OUT:-}"
    head -20 "$CREATE_ERR" >&2 || true
    rm -f "$CREATE_ERR"
    if _unrecoverable_create_err "$CREATE_BLOB"; then
      echo "$PROG: MANUAL_REVIEW unrecoverable create error for account=$VAST_ACCOUNT (template/api-key) — aborting; every candidate would fail identically." >&2
      [[ -n "$TEMPLATE_HASH" ]] && echo "$PROG: template $TEMPLATE_HASH is likely owned by a DIFFERENT account. Remedy: re-run with VAST_ACCOUNT=private (the owner), OR recreate/share the template on this account (see SKILL.md 'Team-account templates')." >&2
      exit 4
    fi
    echo "$PROG: vastai create instance failed for offer $OID — trying next candidate" >&2
    continue
  fi
  rm -f "$CREATE_ERR"

  # Parse new_contract: prefer JSON; fall back to a tolerant regex on text output.
  INSTANCE_ID=""
  if echo "$CREATE_OUT" | jq -e 'type=="object"' >/dev/null 2>&1; then
    INSTANCE_ID=$(echo "$CREATE_OUT" | jq -r '.new_contract // .instance_id // empty')
  fi
  if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    INSTANCE_ID=$(echo "$CREATE_OUT" | grep -oE '"new_contract"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1 || true)
  fi
  if [[ -z "$INSTANCE_ID" ]]; then
    if _unrecoverable_create_err "$CREATE_OUT"; then
      echo "$PROG: MANUAL_REVIEW unrecoverable create error for account=$VAST_ACCOUNT (template/api-key):" >&2
      echo "$CREATE_OUT" >&2
      [[ -n "$TEMPLATE_HASH" ]] && echo "$PROG: template $TEMPLATE_HASH not accessible by account=$VAST_ACCOUNT — re-run with VAST_ACCOUNT=private OR recreate the template on this account (see SKILL.md 'Team-account templates')." >&2
      exit 4
    fi
    echo "$PROG: could not parse new_contract from 'vastai create instance' output — trying next candidate:" >&2
    echo "$CREATE_OUT" >&2
    continue
  fi

  # Track for cleanup the INSTANT the instance exists, so a crash/timeout/probe
  # failure from here on destroys it instead of leaking.
  ACTIVE_UNVERIFIED+=("$INSTANCE_ID")
  echo "$PROG: instance $INSTANCE_ID created, waiting up to ${TIMEOUT}s for running + ssh-routable" >&2

  # TEAM account: account-level keys aren't supported, so Vast attaches nothing
  # to a team-key create — the box would be unreachable. Attach the harness
  # key(s) to THIS instance now. (Private boxes get account keys auto-attached.)
  if [[ "$VAST_ACCOUNT" == "team" ]]; then
    ATTACHED=0
    for pk in "$HOME/.ssh/Vast-Team.pub" "$HOME/.ssh/vast_ai_name.pub" "$HOME/.ssh/vast_ai.pub"; do
      [[ -r "$pk" ]] || continue
      if vastai attach ssh "$INSTANCE_ID" "$(cat "$pk")" >/dev/null 2>&1; then
        echo "$PROG: attached $(basename "$pk") to team instance $INSTANCE_ID" >&2
        ATTACHED=$((ATTACHED + 1))
      else
        echo "$PROG: WARNING: failed to attach $(basename "$pk") to $INSTANCE_ID" >&2
      fi
    done
    # Zero keys attached => the box is GUARANTEED unreachable. Don't wait out the
    # timeout on a doomed box — destroy it now and advance to the next candidate.
    if [[ "$ATTACHED" -eq 0 ]]; then
      echo "$PROG: no SSH key attached to team instance $INSTANCE_ID — destroying + next candidate" >&2
      _destroy_instance "$INSTANCE_ID"; _drop_unverified "$INSTANCE_ID"; continue
    fi
  fi

  # ---- poll for ready ----
  #
  # The load-bearing readiness signal is `.ports["22/tcp"][0].HostPort`. Both
  # the direct route (public_ipaddr + that HostPort) AND the proxy route
  # (ssh*.vast.ai + ssh_port) depend on the host having published the
  # container's 22/tcp mapping. While `ports == null`, the proxy address
  # is pre-allocated and LOOKS present — but `ssh ssh1.vast.ai:17076`
  # gets "Connection refused" because the proxy has nothing to forward to.
  # So we require `ports["22/tcp"]` populated regardless of the chosen
  # route. Endpoint choice is just direct-preferred-over-proxy:
  #   DIRECT: (public_ipaddr, ports["22/tcp"][0].HostPort)
  #   PROXY : (ssh_host == ssh*.vast.ai, ssh_port)
  # NEVER mix public_ipaddr with ssh_port — that combination is unreachable.
  # vast.ai's `actual_status` is non-monotonic during image pull (can flip
  # running→loading→running); ports["22/tcp"] is monotonic so we lean on it.
  # vast.ai's API sometimes returns `status_msg` with raw \n inside the JSON
  # string (invalid JSON). We sanitize via python before jq sees it.
  _vast_show_sanitized() {
    vastai show instance "$1" --raw 2>/dev/null | python3 -c '
import sys, json, re
raw = sys.stdin.read()
raw = re.sub(r"\"status_msg\"\s*:\s*\"([^\"\\\\]|\\\\.)*\"", "\"status_msg\":\"<sanitized>\"", raw)
try:
    json.loads(raw); sys.stdout.write(raw)
except Exception:
    sys.stdout.write("{}")
' 2>/dev/null || true
  }

  START_TS=$(date +%s)
  DEADLINE=$(( START_TS + TIMEOUT ))
  LAST_STATUS="unknown"
  INSTANCE_JSON=""
  ENDPOINT_MODE=""
  POLLS=0
  POLL_OUTCOME="timeout"     # ready | host-failed | timeout
  while [[ $(date +%s) -lt $DEADLINE ]]; do
    SHOW_RAW="$(_vast_show_sanitized "$INSTANCE_ID")"
    if [[ -n "$SHOW_RAW" ]] && echo "$SHOW_RAW" | jq -e 'type=="object" and has("id")' >/dev/null 2>&1; then
      ACTUAL=$(echo "$SHOW_RAW" | jq -r '.actual_status // .status // .cur_state // "unknown"')
      LAST_STATUS="$ACTUAL"
      PORT22=$(echo "$SHOW_RAW" | jq -r '
        if .ports and .ports["22/tcp"] and (.ports["22/tcp"]|length>0) and .ports["22/tcp"][0].HostPort
        then (.ports["22/tcp"][0].HostPort | tonumber) | tostring
        else ""
        end
      ')

      # Live progress on stderr — agents call this as a blocking step, and a
      # 30 GB image pull on a slow host can run 20+ min. Without this they'd
      # see nothing for the entire wait. Pull the latest `status_msg` line
      # straight from the raw API output (bypassing our sanitized copy) so
      # we get the actual docker-pull progress, even though the raw JSON
      # is technically malformed when status_msg has newlines.
      LATEST_MSG=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c 'import sys,re;m=re.search(r"\"status_msg\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"", sys.stdin.read(), re.S);print((m.group(1) if m else "")[-80:].replace("\\n","|").replace("\n","|"))' 2>/dev/null \
        || echo "")
      ELAPSED=$(( $(date +%s) - START_TS ))
      POLLS=$(( POLLS + 1 ))
      echo "$PROG: [+${ELAPSED}s poll #${POLLS}] actual_status=$ACTUAL port22=${PORT22:-none} msg='${LATEST_MSG}'" >&2

      # Fail-fast on host-side errors. When vast.ai's host gives up trying to
      # start the container (e.g. unresolvable CDI device, OCI runtime failure,
      # broken docker daemon on the host), `intended_status` flips from
      # `running` to `stopped`. That's deterministic and unrecoverable from the
      # client — the only remedy is to destroy + re-provision (vast scheduler
      # picks a different host). Without this, we'd block for the full
      # --timeout (~25 min) on a box that's already dead.
      INTENDED=$(echo "$SHOW_RAW" | jq -r '.intended_status // ""')
      LATEST_MSG_RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c 'import sys,re;m=re.search(r"\"status_msg\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"", sys.stdin.read(), re.S);print(m.group(1) if m else "")' \
        2>/dev/null || echo "")
      if [[ "$INTENDED" == "stopped" || "$ACTUAL" == "stopped" \
            || "$LATEST_MSG_RAW" == *"Error response from daemon"* \
            || "$LATEST_MSG_RAW" == *"OCI runtime"* \
            || "$LATEST_MSG_RAW" == *"unresolvable CDI"* ]]; then
        echo "$PROG: instance $INSTANCE_ID failed host-side (intended_status=$INTENDED actual_status=$ACTUAL)" >&2
        echo "$PROG: status_msg tail: ${LATEST_MSG_RAW: -300}" >&2
        echo "$PROG: this host is unrecoverable — will destroy + advance to the next candidate." >&2
        LAST_STATUS="host-failed"
        POLL_OUTCOME="host-failed"
        break
      fi

      if [[ "$ACTUAL" == "running" && -n "$PORT22" && "$PORT22" != "0" ]]; then
        DIRECT_IP=$(echo  "$SHOW_RAW" | jq -r '.public_ipaddr // ""')
        PROXY_HOST=$(echo "$SHOW_RAW" | jq -r '.ssh_host // ""')
        PROXY_PORT=$(echo "$SHOW_RAW" | jq -r '(.ssh_port // 0) | tostring')

        if [[ -n "$DIRECT_IP" ]]; then
          ENDPOINT_MODE="direct"
          INSTANCE_JSON="$SHOW_RAW"
          POLL_OUTCOME="ready"
          break
        elif [[ -n "$PROXY_HOST" && "$PROXY_HOST" != "$DIRECT_IP" && "$PROXY_PORT" != "0" ]]; then
          ENDPOINT_MODE="proxy"
          INSTANCE_JSON="$SHOW_RAW"
          POLL_OUTCOME="ready"
          break
        fi
        LAST_STATUS="port-mapped-but-no-host"
      elif [[ "$ACTUAL" == "running" ]]; then
        LAST_STATUS="running-but-port22-not-mapped"
      fi
    fi
    _sleep_quiet "$POLL_INTERVAL"
  done

  # Port-mapped never reached (timeout) or host gave up: destroy + next candidate.
  if [[ "$POLL_OUTCOME" != "ready" || -z "$INSTANCE_JSON" ]]; then
    echo "$PROG: instance $INSTANCE_ID not ssh-routable (outcome=$POLL_OUTCOME last=$LAST_STATUS within ${TIMEOUT}s) — destroying + next candidate" >&2
    _destroy_instance "$INSTANCE_ID"; _drop_unverified "$INSTANCE_ID"; continue
  fi

  # ---- finalize fields ----
  # Emit the endpoint pair we actually validated above — never mix the two.
  if [[ "$ENDPOINT_MODE" == "direct" ]]; then
    SSH_HOST=$(echo "$INSTANCE_JSON" | jq -r '.public_ipaddr')
    SSH_PORT=$(echo "$INSTANCE_JSON" | jq -r '.ports["22/tcp"][0].HostPort | tonumber')
  else
    SSH_HOST=$(echo "$INSTANCE_JSON" | jq -r '.ssh_host')
    SSH_PORT=$(echo "$INSTANCE_JSON" | jq -r '.ssh_port')
  fi

  # ---- REAL SSH reachability gate (the critical check) ----
  # The 22/tcp mapping only proves the host PUBLISHED the port. Prove we can
  # actually log in with the offered key before writing a "success" handle.
  echo "$PROG: port 22 mapped ($ENDPOINT_MODE $SSH_HOST:$SSH_PORT); probing real SSH ..." >&2
  if _ssh_probe "$SSH_HOST" "$SSH_PORT"; then
    SSH_OK=1
  else
    SSH_OK=0
    # Team boxes: the per-instance key may have raced container start — re-attach once and retry.
    if [[ "$VAST_ACCOUNT" == "team" ]]; then
      echo "$PROG: SSH probe failed; re-attaching key to $INSTANCE_ID and retrying ..." >&2
      for pk in "$HOME/.ssh/Vast-Team.pub" "$HOME/.ssh/vast_ai_name.pub" "$HOME/.ssh/vast_ai.pub"; do
        [[ -r "$pk" ]] && vastai attach ssh "$INSTANCE_ID" "$(cat "$pk")" >/dev/null 2>&1 || true
      done
      _ssh_probe "$SSH_HOST" "$SSH_PORT" && SSH_OK=1
    fi
  fi
  if [[ "$SSH_OK" != "1" ]]; then
    echo "$PROG: $INSTANCE_ID port-mapped but SSH probe FAILED — unreachable box; destroying + next candidate" >&2
    _destroy_instance "$INSTANCE_ID"; _drop_unverified "$INSTANCE_ID"; continue
  fi
  echo "$PROG: SSH verified to $INSTANCE_ID ($SSH_HOST:$SSH_PORT)" >&2

  # ---- seed HF+WandB+R2 secrets onto the box (deterministic; no agent step) --
  # The on-box launcher FATALs without /root/.config/verl-research/secrets.env;
  # push it the moment SSH is verified so training "just works". Best-effort:
  # a failure WARNs but never aborts provisioning (the launcher FATAL is the
  # backstop). VAST keys are structurally withheld (see _seed_secrets.sh).
  seed_secrets_to_box "$SSH_HOST" "$SSH_PORT" "$SSH_IDENTITY" || true

  # ---- pids.max host-lottery gate (promoted from SKILL.md prose) ----
  # Hosts capping the container at <= 2048 pids deterministically SIGABRT at the
  # FSDP->vLLM boundary (the verl stack needs ~1700+ threads). Catch it NOW,
  # while advancing to the next candidate is cheap.
  PIDS_MAX=$(timeout 30 ssh -i "$SSH_IDENTITY" -o ConnectTimeout=8 -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST" \
      'cat /sys/fs/cgroup/pids.max /sys/fs/cgroup/pids/pids.max 2>/dev/null | head -1' 2>/dev/null || echo "")
  if [[ "$PIDS_MAX" =~ ^[0-9]+$ ]] && (( PIDS_MAX <= 2048 )); then
    echo "$PROG: $INSTANCE_ID pids.max=$PIDS_MAX (<=2048) — host would SIGABRT under FSDP+vLLM; destroying + next candidate" >&2
    _destroy_instance "$INSTANCE_ID"; _drop_unverified "$INSTANCE_ID"; continue
  fi
  [[ -n "$PIDS_MAX" ]] && echo "$PROG: pids.max=$PIDS_MAX ok" >&2

  PUB_IP=$(echo "$INSTANCE_JSON"  | jq -r '.public_ipaddr // ""')
  INST_GPU_NAME=$(echo "$INSTANCE_JSON" | jq -r '.gpu_name // ""')
  INST_NUM_GPUS=$(echo "$INSTANCE_JSON" | jq -r '.num_gpus // 0')

  # Fall back to chosen-offer values if the running instance omitted them
  [[ -z "$INST_GPU_NAME" || "$INST_GPU_NAME" == "null" ]] && INST_GPU_NAME="$OGPU"
  [[ -z "$INST_NUM_GPUS" || "$INST_NUM_GPUS" == "0" || "$INST_NUM_GPUS" == "null" ]] && INST_NUM_GPUS="$ONUMG"

  CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # Paste-ready login command. The fixed SSH form (per project.yaml vast_ssh): always
  # `-i $SSH_IDENTITY` (bare ssh falls back to id_rsa/id_ed25519 -> publickey fail) and
  # `-o StrictHostKeyChecking=accept-new` (reused Vast IPs otherwise trip host-key verify).
  # The -L 8080:localhost:8080 tunnel exposes the on-box vLLM/WandB UI on the laptop.
  SSH_LOGIN="ssh -i ${SSH_IDENTITY} -o StrictHostKeyChecking=accept-new -p ${SSH_PORT:-22} root@${SSH_HOST} -L 8080:localhost:8080"

  HANDLE=$(jq -cn --sort-keys \
    --arg sv  "$SCHEMA_VERSION" \
    --arg iid "$INSTANCE_ID" \
    --arg oid "$OID" \
    --arg sh  "$SSH_HOST" \
    --argjson sp "${SSH_PORT:-0}" \
    --arg pip "$PUB_IP" \
    --arg gn  "$INST_GPU_NAME" \
    --argjson ng "${INST_NUM_GPUS:-0}" \
    --argjson dph "$OPRICE" \
    --arg ca  "$CREATED_AT" \
    --arg lb  "$LABEL" \
    --arg sid "$SESSION_ID" \
    --arg sl  "$SSH_LOGIN" \
    --arg va  "$VAST_ACCOUNT" \
    '{schema_version:$sv, instance_id:$iid, offer_id:$oid,
      ssh_host:$sh, ssh_port:$sp, public_ipaddr:$pip,
      gpu_name:$gn, num_gpus:$ng, dph_total:$dph,
      created_at:$ca, label:$lb, session_id:$sid, ssh_login:$sl,
      vast_account:$va}')

  # ---- atomic handle write ----
  HANDLE_PATH="$HANDLE_DIR/${INSTANCE_ID}.json"
  TMP_HANDLE="$(mktemp "$HANDLE_DIR/.${INSTANCE_ID}.XXXXXX")"
  echo "$HANDLE" > "$TMP_HANDLE"
  mv "$TMP_HANDLE" "$HANDLE_PATH"

  # Verified + handle written: this box is no longer an orphan — drop it from the
  # cleanup set so the EXIT trap leaves it running.
  _drop_unverified "$INSTANCE_ID"

  echo "VAST_HANDLE: $HANDLE"
  echo "$PROG: instance $INSTANCE_ID running at $SSH_HOST:$SSH_PORT (handle: $HANDLE_PATH)" >&2
  echo "$PROG: log in FIRST, then work — copy this verbatim:" >&2
  echo "  $SSH_LOGIN" >&2

  TOTAL_DPH=$(awk -v a="$TOTAL_DPH" -v b="$OPRICE" 'BEGIN { printf "%.4f", a + b }')
  PROVISIONED=$((PROVISIONED + 1))
done < <(echo "$CHOSEN" | jq -c '.[]')

# Candidate pool exhausted before COUNT boxes were SSH-verified. Any instances
# created along the way were already destroyed on their failure; the EXIT trap is
# the backstop for any that slipped through. Fail non-zero so the caller retries.
if [[ "$PROVISIONED" -lt "$COUNT" ]]; then
  echo "$PROG: provisioned only $PROVISIONED of $COUNT requested — candidate pool exhausted (unreachable/failed hosts were destroyed)" >&2
  exit 5
fi

echo "VAST_PROVISIONED: count=${PROVISIONED} total_dph=${TOTAL_DPH}"
exit 0
