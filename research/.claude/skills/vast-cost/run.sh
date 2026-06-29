#!/usr/bin/env bash
# vast-cost — live Vast.ai spend + leak detector (READ-ONLY; never destroys).
#
# Sums the burn rate across LIVE (running) instances on BOTH accounts (private +
# team), reports $/hr and projected 24h spend, and — the part the official
# /vastai:cost lacks — cross-checks every live instance against the ledger and
# FLAGS any box with no owning RUNNING/PROVISIONED row (an untracked box = a
# likely billing leak, e.g. a provision orphan or a teardown that silently
# no-opped under the wrong account).
#
# Output (stdout, machine-readable):
#   VAST_COST: burn_rate_dph=<X.XXXX> projected_24h_usd=<Y.YY> untracked=<0|1>
set -uo pipefail

PROG="vast-cost"
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SKILL_DIR/../.." && pwd)}"
LEDGER="$PROJECT_DIR/.claude/state/runs.jsonl"

command -v vastai >/dev/null || { echo "$PROG: 'vastai' not on PATH (pip install vastai)" >&2; exit 1; }
command -v jq    >/dev/null || { echo "$PROG: 'jq' not on PATH" >&2; exit 1; }

# shellcheck disable=SC1090
source "$SKILL_DIR/../_vast_account.sh"; vast_load_secrets

# Instance ids the ledger considers live (RUNNING|PROVISIONED) — the leak baseline.
LEDGER_IDS=""
if [[ -f "$LEDGER" ]]; then
  LEDGER_IDS="$(jq -r 'select(.status=="RUNNING" or .status=="PROVISIONED") | .handles[]?.instance_id // empty' "$LEDGER" 2>/dev/null | sort -u)"
fi

TOTAL_DPH=0
ANY=0
LEAK=0

for acct in private team; do
  KEY="$(vast_key_for "$acct")"
  [[ -z "$KEY" ]] && continue
  RAW="$(VAST_API_KEY="$KEY" vastai show instances --raw 2>/dev/null || true)"
  echo "$RAW" | jq -e 'type=="array"' >/dev/null 2>&1 || continue
  N=$(echo "$RAW" | jq 'length')
  if [[ "${N:-0}" -eq 0 ]]; then
    echo "$PROG: [$acct] 0 instances" >&2
    continue
  fi
  ANY=1
  # Burn rate counts RUNNING instances only (stopped ones bill storage, not $/hr).
  ACCT_DPH=$(echo "$RAW" | jq '[.[] | select((.actual_status // "")=="running") | (.dph_total // 0 | tonumber)] | add // 0')
  TOTAL_DPH=$(awk -v a="$TOTAL_DPH" -v b="$ACCT_DPH" 'BEGIN{printf "%.4f", a+b}')
  echo "$PROG: [$acct] $N instance(s) listed, running burn \$${ACCT_DPH}/hr" >&2
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    iid=$(echo "$line" | jq -r '.id')
    st=$(echo  "$line" | jq -r '.actual_status // .cur_state // "?"')
    dph=$(echo "$line" | jq -r '.dph_total // 0')
    gpu=$(echo "$line" | jq -r '.num_gpus // "?"')
    gn=$(echo  "$line" | jq -r '.gpu_name // "?"')
    tag=""
    if ! grep -qx "$iid" <<<"$LEDGER_IDS"; then
      tag="  <-- UNTRACKED (no live ledger row; possible LEAK)"
      LEAK=1
    fi
    echo "$PROG:   id=$iid status=$st ${gpu}x${gn} \$${dph}/hr$tag" >&2
  done < <(echo "$RAW" | jq -c '.[]')
done

PROJ24=$(awk -v t="$TOTAL_DPH" 'BEGIN{printf "%.2f", t*24}')
echo "VAST_COST: burn_rate_dph=${TOTAL_DPH} projected_24h_usd=${PROJ24} untracked=${LEAK}"
[[ "$ANY" -eq 0 ]] && echo "$PROG: no instances on any account — \$0/hr" >&2
[[ "$LEAK" -eq 1 ]] && echo "$PROG: WARNING untracked live instance(s) — investigate, then 'vast-teardown <id>' (it resolves the right account)." >&2
exit 0
