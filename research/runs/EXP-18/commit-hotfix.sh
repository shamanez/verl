#!/usr/bin/env bash
# Capture any edit under /workspace/verl as a git commit + format-patch.
# The patch is rsync'd back to the laptop's $PARENT/runs/EXP-18/hotfix-patches/
# by sync-metrics on the next 5-min tick.
#
# THIS dispatch runs code_change=FALSE on the vast-ai-workload branch (dense
# reference + spectral floor — no patch). Any in-container edit therefore lands
# on whatever branch /workspace/verl is on (vast-ai-workload by default). Later
# step-2 candidate dispatches check out exp/18-<slug>; this helper captures the
# diff regardless of branch, so the work survives the box dying.
#
# If $GH_PUSH_TOKEN is set in the container env, also pushes to origin right
# away (best case — instance can die immediately after). For THIS dispatch no
# token is shipped, so the patch lives in hotfix-patches/ until rsync.
#
# Usage:  bash /workspace/runs/EXP-18/commit-hotfix.sh "<short message>" [branch]
set -euo pipefail
MSG="${1:?usage: commit-hotfix.sh <message> [branch]}"
PUSH_BRANCH="${2:-$(cd /workspace/verl && git rev-parse --abbrev-ref HEAD)}"

cd /workspace/verl
if git diff --quiet && git diff --staged --quiet; then
  echo "commit-hotfix: working tree clean — nothing to commit"
  exit 0
fi

git add -A
git commit -m "[EXP-18] in-container hotfix: $MSG"

# Format-patch under the run dir so sync-metrics rsyncs it back.
mkdir -p /workspace/runs/EXP-18/hotfix-patches
N=$(ls /workspace/runs/EXP-18/hotfix-patches/*.patch 2>/dev/null | wc -l)
NEXT=$(printf "%03d" $((N + 1)))
git format-patch -1 --start-number "$NEXT" -o /workspace/runs/EXP-18/hotfix-patches/
echo "commit-hotfix: patch dropped in /workspace/runs/EXP-18/hotfix-patches/${NEXT}-*.patch"
echo "commit-hotfix: will rsync back to laptop within ~5 min (sync-metrics tick)."

# Best-effort in-container push, if a fine-scoped PAT was passed to the container.
if [[ -n "${GH_PUSH_TOKEN:-}" ]]; then
  REPO_URL="https://x-access-token:${GH_PUSH_TOKEN}@github.com/shamanez/verl.git"
  if git push "$REPO_URL" HEAD:"$PUSH_BRANCH"; then
    echo "commit-hotfix: also pushed to origin/$PUSH_BRANCH on shamanez/verl"
  else
    echo "commit-hotfix: push failed (auth?) — relying on rsync round-trip" >&2
  fi
else
  echo "commit-hotfix: no GH_PUSH_TOKEN in env — patch lives only in hotfix-patches/ until rsync"
fi
