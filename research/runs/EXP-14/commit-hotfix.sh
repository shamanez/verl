#!/usr/bin/env bash
# Capture any edit under /workspace/verl as a git commit + format-patch.
# The patch is rsync'd back to the laptop's research/runs/EXP-14/hotfix-patches/
# by sync-metrics on the next 5-min tick. If $GH_PUSH_TOKEN is set in the
# container env, also pushes to origin/exp/14-clean-cadence on shamanez/verl.
#
# Usage:  bash /workspace/runs/EXP-14/commit-hotfix.sh "<short message>"
set -euo pipefail
MSG="${1:?usage: commit-hotfix.sh <message>}"

cd /workspace/verl
if git diff --quiet && git diff --staged --quiet; then
  echo "commit-hotfix: working tree clean — nothing to commit"
  exit 0
fi

git add -A
git commit -m "[EXP-14] in-container hotfix: $MSG"

mkdir -p /workspace/runs/EXP-14/hotfix-patches
N=$(ls /workspace/runs/EXP-14/hotfix-patches/*.patch 2>/dev/null | wc -l)
NEXT=$(printf "%03d" $((N + 1)))
git format-patch -1 --start-number "$NEXT" -o /workspace/runs/EXP-14/hotfix-patches/
echo "commit-hotfix: patch dropped in /workspace/runs/EXP-14/hotfix-patches/${NEXT}-*.patch"
echo "commit-hotfix: will rsync back to laptop within ~5 min (sync-metrics tick)."

if [[ -n "${GH_PUSH_TOKEN:-}" ]]; then
  REPO_URL="https://x-access-token:${GH_PUSH_TOKEN}@github.com/shamanez/verl.git"
  if git push "$REPO_URL" HEAD:"exp/14-clean-cadence"; then
    echo "commit-hotfix: also pushed to origin/exp/14-clean-cadence on shamanez/verl"
  else
    echo "commit-hotfix: push failed (auth?) — relying on rsync round-trip" >&2
  fi
else
  echo "commit-hotfix: no GH_PUSH_TOKEN in env — patch lives only in hotfix-patches/ until rsync"
fi
