#!/usr/bin/env bash
# commit-hotfix.sh — Vast-volatility safety: capture any in-container edit to
# /workspace/verl as a commit + format-patch BEFORE the box dies. The runner
# copies this template to runs/83-growing-fixed-base-anchor/commit-hotfix.sh (substituting 83-growing-fixed-base-anchor) and
# rsyncs it to the box. sync-metrics rsyncs hotfix-patches/ back to the laptop;
# log-writer lists them in the PR body.
# Usage on the box:  bash /workspace/runs/83-growing-fixed-base-anchor/commit-hotfix.sh "<short message>"
set -euo pipefail
MSG="${1:?usage: commit-hotfix.sh <message>}"
RUN_ID="83-growing-fixed-base-anchor"

cd /workspace/verl
if git diff --quiet && git diff --staged --quiet; then
  echo "commit-hotfix: working tree clean — nothing to commit"; exit 0
fi
git config user.email "harness@verl-research.local" 2>/dev/null || true
git config user.name  "verl-research-harness" 2>/dev/null || true
git add -A
git commit -m "[$RUN_ID] in-container hotfix: $MSG"

mkdir -p "/workspace/runs/$RUN_ID/hotfix-patches"
N=$(ls "/workspace/runs/$RUN_ID/hotfix-patches/"*.patch 2>/dev/null | wc -l)
git format-patch -1 --start-number "$(printf '%03d' $((N + 1)))" \
  -o "/workspace/runs/$RUN_ID/hotfix-patches/"
echo "commit-hotfix: patch dropped; rsyncs to the laptop on the next sync tick."

# Best-effort immediate push if a fine-scoped PAT is present (box may die any time).
if [[ -n "${GH_PUSH_TOKEN:-}" ]]; then
  git push "https://x-access-token:${GH_PUSH_TOKEN}@github.com/shamanez/verl.git" \
    HEAD:"exp/$RUN_ID" \
    && echo "commit-hotfix: pushed to origin/exp/$RUN_ID" \
    || echo "commit-hotfix: push failed — relying on rsync round-trip" >&2
fi
