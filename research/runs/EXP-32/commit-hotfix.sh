#!/usr/bin/env bash
# Capture any edit under /workspace/verl as a git commit + format-patch.
# The patch is rsync'd back to the laptop's research/runs/EXP-32/hotfix-patches/
# by sync-metrics on the next 5-min tick. If $GH_PUSH_TOKEN is set in the
# container env, also pushes to origin/exp/32-<slug> on shamanez/verl right away.
#
# NOTE: EXP-32 is config-only (code_change=false) — there is NO exp/* branch and
# NO verl/ patch by design. This helper exists per the runner contract as a safety
# net ONLY: if a diagnosis forces an in-container verl/ edit, capture it here so the
# work survives the (operator-managed) box dying. Under normal operation it is a no-op.
#
# Usage:  bash /workspace/runs/EXP-32/commit-hotfix.sh "<short message>"
set -euo pipefail
MSG="${1:?usage: commit-hotfix.sh <message>}"

cd /workspace/verl
if git diff --quiet && git diff --staged --quiet; then
  echo "commit-hotfix: working tree clean — nothing to commit"
  exit 0
fi

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

git add -A
git commit -m "[EXP-32] in-container hotfix: $MSG"

mkdir -p /workspace/runs/EXP-32/hotfix-patches
N=$(ls /workspace/runs/EXP-32/hotfix-patches/*.patch 2>/dev/null | wc -l)
NEXT=$(printf "%03d" $((N + 1)))
git format-patch -1 --start-number "$NEXT" -o /workspace/runs/EXP-32/hotfix-patches/
echo "commit-hotfix: patch dropped in /workspace/runs/EXP-32/hotfix-patches/${NEXT}-*.patch"
echo "commit-hotfix: will rsync back to laptop within ~5 min (sync-metrics tick)."

if [[ -n "${GH_PUSH_TOKEN:-}" ]]; then
  REPO_URL="https://x-access-token:${GH_PUSH_TOKEN}@github.com/shamanez/verl.git"
  if git push "$REPO_URL" HEAD:"exp/32-signed-ema-validm"; then
    echo "commit-hotfix: also pushed to origin/exp/32-signed-ema-validm on shamanez/verl"
  else
    echo "commit-hotfix: push failed (auth?) — relying on rsync round-trip" >&2
  fi
else
  echo "commit-hotfix: no GH_PUSH_TOKEN in env — patch lives only in hotfix-patches/ until rsync"
fi
