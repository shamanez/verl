#!/usr/bin/env bash
# Ephemeral supply watcher for EXP-16 (SUPPLY_BLOCKED).
# Polls the two sanctioned tiers every 90s. Exits 0 the instant either tier
# has >=1 qualifying offer (so the orchestrator wakes and dispatches the
# experiment-runner against live supply); exits 7 on timeout. Free (API only).
set -uo pipefail
source ~/.config/verl-research/secrets.env 2>/dev/null

TIER0='num_gpus=4 gpu_name=H200 gpu_ram>=140 reliability>=0.95 rentable=true verified=true'
TIER1='num_gpus=8 gpu_name=H100 gpu_ram>=80 reliability>=0.95 rentable=true verified=true'
FLAG=/Users/shamane/Documents/verl/research/.claude/state/.supply-result
MAX_POLLS=34   # ~51 min @ 90s
n=0
while (( n < MAX_POLLS )); do
  n=$((n+1))
  c0=$(vastai search offers "$TIER0" --raw 2>/dev/null | jq 'length' 2>/dev/null || echo 0)
  c1=$(vastai search offers "$TIER1" --raw 2>/dev/null | jq 'length' 2>/dev/null || echo 0)
  ts=$(date -Iseconds)
  echo "[$ts] poll $n/$MAX_POLLS  tier0(4xH200)=$c0  tier1(8xH100)=$c1"
  if [[ "${c0:-0}" -gt 0 || "${c1:-0}" -gt 0 ]]; then
    tier="tier1-8xH100"; [[ "${c0:-0}" -gt 0 ]] && tier="tier0-4xH200"
    echo "SUPPLY_AVAILABLE $tier tier0=$c0 tier1=$c1 at $ts" | tee "$FLAG"
    exit 0
  fi
  sleep 90
done
echo "SUPPLY_TIMEOUT after $MAX_POLLS polls (~51 min), both tiers still dry" | tee "$FLAG"
exit 7
