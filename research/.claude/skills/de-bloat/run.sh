#!/usr/bin/env bash
# de-bloat — fold a COMPLETED experiment into runs/SUMMARY.md and remove its bulky
# artifacts (run dir incl. *.bundle, its plan file, its stale handle). See SKILL.md.
#
# Hard guards:
#   - NEVER touches the baseline (.claude/plans/baseline.md, id 3/EXP-3); the baseline is
#     comm-eff OFF and keeps no standalone run dir.
#   - Refuses an experiment that still has a live (RUNNING/PROVISIONED) ledger row.
#   - Refuses an undone experiment (run dir present but no verdict.md / LOG entry).
#   - Idempotent: re-running on an already-folded id is a no-op.
set -euo pipefail

PROG=de-bloat
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

LEDGER=".claude/state/runs.jsonl"
SUMMARY="runs/SUMMARY.md"
DRY=0
IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY=1; shift ;;
    -h|--help) sed -n '1,60p' "$(dirname "$0")/SKILL.md"; exit 0 ;;
    *) IDS+=("$1"); shift ;;
  esac
done
[[ ${#IDS[@]} -gt 0 ]] || { echo "$PROG: need at least one experiment id (e.g. EXP-5)"; exit 2; }

# Ensure SUMMARY exists with the canonical table header.
if [[ ! -f "$SUMMARY" ]]; then
  mkdir -p runs
  cat > "$SUMMARY" <<'HDR'
# Research runs — summary

Concise record of what has run on this harness. Full per-experiment artifacts are
pruned (folded here by the `de-bloat` skill); the durable record is here + git history + merged code.

| id | milestone | what | result | merged |
|---|---|---|---|---|
| **baseline** | M1 | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the control) | reference curve | — |
HDR
fi

run() { if [[ "$DRY" == 1 ]]; then echo "  [dry-run] $*"; else eval "$*"; fi; }

field() { grep -m1 -E "^- *$1:" "$2" 2>/dev/null | sed -E "s/^- *$1: *//" | sed 's/[[:space:]]*$//'; }

folded=0
for RAW in "${IDS[@]}"; do
  # --- baseline guard, by NAME, before any parsing (covers `baseline`, `EXP-3`, `3`) ---
  if [[ "$RAW" == "baseline" || "$RAW" == "EXP-3" || "$RAW" == "3" ]]; then
    echo "$PROG: refusing to de-bloat the baseline ($RAW) — it is the permanent control."; continue
  fi
  NUM="$(echo "$RAW" | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -z "$NUM" ]]; then
    echo "$PROG: can't parse an experiment number from '$RAW' — skipping."; continue
  fi
  if [[ "$NUM" == "3" ]]; then
    echo "$PROG: refusing to de-bloat the baseline (EXP-3) — it is the permanent control."; continue
  fi
  ID="EXP-$NUM"

  RUNDIR="runs/$ID"
  PLAN=".claude/plans/$NUM.md"

  # --- idempotency: already folded? ---
  if [[ ! -d "$RUNDIR" && ! -f "$PLAN" ]] && grep -q "| *$ID " "$SUMMARY"; then
    echo "$PROG: $ID already folded — skipping."; continue
  fi

  # --- live-instance guard ---
  if jq -e --arg id "$ID" 'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED"))' "$LEDGER" >/dev/null 2>&1; then
    echo "$PROG: $ID still has a live (RUNNING/PROVISIONED) ledger row — tear it down first. Refusing."; continue
  fi

  # --- done-check: refuse unless the issue is terminal (verdict written OR a LOG entry).
  #     This protects PLANNED-but-not-run backlog issues (plan file present, no run yet) —
  #     de-bloating one of those would delete pending work. ---
  VERDICT="$(grep -m1 -oE 'VERDICT:[[:space:]]*(PASS|REVISE|STOP)' "$RUNDIR/verdict.md" 2>/dev/null | grep -oE 'PASS|REVISE|STOP' || true)"
  if [[ -z "$VERDICT" ]] && ! grep -q "$ID" LOG.md 2>/dev/null; then
    echo "$PROG: $ID is not done (no verdict.md, no LOG.md entry) — refusing to fold/delete a pending issue."; continue
  fi
  [[ -z "$VERDICT" ]] && VERDICT="$(grep -m1 -oE "$ID .*(PASS|STOP|REVISE)" LOG.md 2>/dev/null | grep -oE 'PASS|STOP|REVISE' || echo 'done')"

  # --- extract a concise row (best-effort; falls back to placeholders) ---
  TITLE="$(field title "$PLAN")"; [[ -z "$TITLE" ]] && TITLE="$(grep -m1 "$ID" LOG.md 2>/dev/null | sed -E 's/^#+ *//' || echo "$ID")"
  MILE="$(field milestone "$PLAN")"; [[ -z "$MILE" ]] && MILE="?"
  # PR number scoped to lines that mention THIS id (avoids grabbing another issue's PR).
  PR="$(grep -hE "$ID" LOG.md PROGRESS.md 2>/dev/null | grep -oE 'pull/[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
  MERGED="—"; [[ -n "$PR" ]] && MERGED="PR #$PR → \`vast-ai-workload\`"
  ROW="| $ID | $MILE | ${TITLE:-$ID} | $VERDICT | $MERGED |"

  echo "$PROG: folding $ID → $SUMMARY"
  echo "    $ROW"

  # --- insert row after the table separator (newest first), unless already present ---
  if ! grep -q "| *$ID " "$SUMMARY"; then
    if [[ "$DRY" == 1 ]]; then echo "  [dry-run] insert row into $SUMMARY";
    else
      TMP="$(mktemp)"; awk -v row="$ROW" '
        { print }
        /^\|---/ && !done { print row; done=1 }
      ' "$SUMMARY" > "$TMP" && mv "$TMP" "$SUMMARY"
    fi
  fi

  # --- remove the bulky artifacts (git rm if tracked, else plain rm) ---
  if [[ -d "$RUNDIR" ]]; then run "git rm -r -q --ignore-unmatch '$RUNDIR' 2>/dev/null || rm -rf '$RUNDIR'"; fi
  if [[ -f "$PLAN"  ]]; then run "git rm -q --ignore-unmatch '$PLAN' 2>/dev/null || rm -f '$PLAN'"; fi
  folded=$((folded+1))
done

# --- light tidy: drop handle files for instances already TORN_DOWN (never touches live ones) ---
if [[ -d .claude/state/vast-handles && "$DRY" == 0 ]]; then
  for h in .claude/state/vast-handles/*.json; do
    [[ -e "$h" ]] || continue
    iid="$(basename "$h" .json)"
    if jq -e --arg i "$iid" 'select((.handles[]?.instance_id==$i or .instance_id==$i) and .status=="TORN_DOWN")' "$LEDGER" >/dev/null 2>&1; then
      git rm -q --ignore-unmatch "$h" 2>/dev/null || rm -f "$h"
    fi
  done
fi

echo "$PROG: DE_BLOATED count=$folded (baseline untouched). Review 'git status' / runs/SUMMARY.md, then commit."
