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

# CLAUDE_PROJECT_DIR resolves to the research/ directory the harness was
# launched from; fall back to walking three levels up from this script.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SKILL_DIR/../../.." && pwd)}"

SCHEMA_VERSION="1"
DEFAULT_HANDLE_DIR="${VERL_VAST_HANDLE_DIR:-$PROJECT_DIR/.claude/state/vast-handles}"

# ---- defaults --------------------------------------------------------------
# Tuned for agent-self-sufficient invocation (Claude /vast-provision with no
# explicit budget): safe per-instance cost, modest disk, generous wait for the
# verlai image pull + onstart script. Production plans (8×H100, etc.) pass
# explicit overrides — see the per-experiment plan in .claude/plans/<N>.md.
COUNT=1
QUERY=""
IMAGE=""
DISK_GB=200                  # see SKILL.md "Disk sizing" — covers image + model + dataset + checkpoints for a typical research run
MAX_PRICE="1.0"              # single-GPU ad-hoc safety; bump for multi-GPU production
MIN_RELIABILITY="0.95"
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
# Both ~/.ssh/vast_ai_name (id 890294) and the legacy ~/.ssh/vast_ai (id 835115)
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
    TEMPLATE_HASH=$(jq -r '.[keys[0]].hash_id' "$TEMPLATES_JSON")
    TEMPLATE_IMAGE=$(jq -r '.[keys[0]].image // "?"' "$TEMPLATES_JSON")
    echo "$PROG: auto-selected template '$TEMPLATE_NAME' hash=$TEMPLATE_HASH image=$TEMPLATE_IMAGE" >&2
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
vast_load_secrets
VAST_ACCOUNT="$(vast_account_norm "${VAST_ACCOUNT:-private}")"
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
  if [[ ! -r "$HOME/.ssh/vast_ai_name.pub" && ! -r "$HOME/.ssh/vast_ai.pub" ]]; then
    echo "$PROG: VAST_ACCOUNT=team needs a local harness public key to attach per-instance" >&2
    echo "$PROG: (~/.ssh/vast_ai_name.pub or ~/.ssh/vast_ai.pub) — none readable." >&2
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
LABEL="${LABEL_PREFIX}:${SESSION_ID}"
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

# ---- pick N cheapest, distinct by host_id ---------------------------------
CHOSEN=$(echo "$FILTERED" | jq --argjson n "$COUNT" "
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

# ---- create + wait loop ---------------------------------------------------
TOTAL_DPH="0"
PROVISIONED=0

# stderr-only helper for sleep so progress doesn't pollute stdout
_sleep_quiet() { sleep "$1"; }

while IFS= read -r offer; do
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
  CREATE_OUT="$("${CREATE_CMD[@]}" 2>"$CREATE_ERR")" || {
    echo "$PROG: vastai create instance failed for offer $OID" >&2
    head -20 "$CREATE_ERR" >&2 || true
    rm -f "$CREATE_ERR"
    exit 4
  }
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
    echo "$PROG: could not parse new_contract from 'vastai create instance' output:" >&2
    echo "$CREATE_OUT" >&2
    exit 4
  fi

  echo "$PROG: instance $INSTANCE_ID created, waiting up to ${TIMEOUT}s for running + ssh-routable" >&2

  # TEAM account: account-level keys aren't supported, so Vast attaches nothing
  # to a team-key create — the box would be unreachable. Attach the harness
  # key(s) to THIS instance now. (Private boxes get account keys auto-attached.)
  if [[ "$VAST_ACCOUNT" == "team" ]]; then
    for pk in "$HOME/.ssh/vast_ai_name.pub" "$HOME/.ssh/vast_ai.pub"; do
      [[ -r "$pk" ]] || continue
      if vastai attach ssh "$INSTANCE_ID" "$(cat "$pk")" >/dev/null 2>&1; then
        echo "$PROG: attached $(basename "$pk") to team instance $INSTANCE_ID" >&2
      else
        echo "$PROG: WARNING: failed to attach $(basename "$pk") to $INSTANCE_ID — box may be unreachable" >&2
      fi
    done
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
        echo "$PROG: this host is unrecoverable. Destroy and re-provision; vast scheduler should pick a different machine." >&2
        LAST_STATUS="host-failed"
        exit 6
      fi

      if [[ "$ACTUAL" == "running" && -n "$PORT22" && "$PORT22" != "0" ]]; then
        DIRECT_IP=$(echo  "$SHOW_RAW" | jq -r '.public_ipaddr // ""')
        PROXY_HOST=$(echo "$SHOW_RAW" | jq -r '.ssh_host // ""')
        PROXY_PORT=$(echo "$SHOW_RAW" | jq -r '(.ssh_port // 0) | tostring')

        if [[ -n "$DIRECT_IP" ]]; then
          ENDPOINT_MODE="direct"
          INSTANCE_JSON="$SHOW_RAW"
          break
        elif [[ -n "$PROXY_HOST" && "$PROXY_HOST" != "$DIRECT_IP" && "$PROXY_PORT" != "0" ]]; then
          ENDPOINT_MODE="proxy"
          INSTANCE_JSON="$SHOW_RAW"
          break
        fi
        LAST_STATUS="port-mapped-but-no-host"
      elif [[ "$ACTUAL" == "running" ]]; then
        LAST_STATUS="running-but-port22-not-mapped"
      fi
    fi
    _sleep_quiet "$POLL_INTERVAL"
  done

  if [[ -z "$INSTANCE_JSON" ]]; then
    echo "$PROG: instance $INSTANCE_ID did not become running+ssh-routable within ${TIMEOUT}s (last status=$LAST_STATUS)" >&2
    exit 5
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

  echo "VAST_HANDLE: $HANDLE"
  echo "$PROG: instance $INSTANCE_ID running at $SSH_HOST:$SSH_PORT (handle: $HANDLE_PATH)" >&2
  echo "$PROG: log in FIRST, then work — copy this verbatim:" >&2
  echo "  $SSH_LOGIN" >&2

  TOTAL_DPH=$(awk -v a="$TOTAL_DPH" -v b="$OPRICE" 'BEGIN { printf "%.4f", a + b }')
  PROVISIONED=$((PROVISIONED + 1))
done < <(echo "$CHOSEN" | jq -c '.[]')

echo "VAST_PROVISIONED: count=${PROVISIONED} total_dph=${TOTAL_DPH}"
exit 0
