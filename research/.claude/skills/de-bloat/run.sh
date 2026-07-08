#!/usr/bin/env bash
# de-bloat — fold a COMPLETED experiment into runs/SUMMARY.md and remove its bulky
# artifacts (run dir, plan cache, stale handles). HUMAN-ONLY — see the gate below.
# This is the manual BATCH FALLBACK for leftover dirs: the normal path is
# /close's own cleanup sweep (scripts/close_cleanup.sh), which cleans each
# issue automatically once its record is durably published.
#
# Hard guards:
#   - OPERATOR GATE: refuses unless DEBLOAT_OPERATOR_ACK=1 is set. The autonomous
#     loop must NEVER set it; only /de-bloat typed by the human does (SKILL.md is
#     disable-model-invocation, so the model cannot auto-fire this skill).
#   - NEVER the baseline (ids baseline / 3 / EXP-3).
#   - Refuses a live (RUNNING/PROVISIONED/EXTERNAL) ledger row.
#   - Refuses an undone experiment (no verdict.md AND no SUMMARY row).
#   - Idempotent.
set -euo pipefail

PROG=de-bloat
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

if [[ "${DEBLOAT_OPERATOR_ACK:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
de-bloat: REFUSED — this skill deletes run dirs and plan files, so it is a
deliberate HUMAN action. If you are the operator and typed /de-bloat yourself:
    DEBLOAT_OPERATOR_ACK=1 bash .claude/skills/de-bloat/run.sh <id> [--dry-run]
Autonomous sessions: never set that variable; suggest the command instead.
EOF
  exit 5
fi

LEDGER=".claude/state/runs.jsonl"
SUMMARY="runs/SUMMARY.md"
PLAN_CACHE_DIR=".claude/state/plan-cache"
# PR base branch for the SUMMARY "merged" column — from project.yaml, never hardcoded
BASE_BRANCH="$(awk -F': ' '/^  code_pr_base_branch:/{sub(/#.*/,"",$2);gsub(/[ "]/,"",$2);print $2}' .claude/project.yaml 2>/dev/null || true)"
BASE_BRANCH="${BASE_BRANCH:-base}"
DRY=0
ALL_TERMINAL=0
IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY=1; shift ;;
    -a|--all-terminal) ALL_TERMINAL=1; shift ;;
    -h|--help) sed -n '1,70p' "$(dirname "$0")/SKILL.md"; exit 0 ;;
    *) IDS+=("$1"); shift ;;
  esac
done

# --all-terminal: enumerate every run dir and let the per-id guards below do
# the deciding — live rows, pending work, and the baseline are refused with a
# printed reason, so the batch folds EXACTLY the terminal runs. Still behind
# the same DEBLOAT_OPERATOR_ACK human gate as single-id invocations.
if [[ "$ALL_TERMINAL" == 1 ]]; then
  for d in runs/*/; do
    [[ -d "$d" ]] || continue
    IDS+=("$(basename "$d")")
  done
fi
[[ ${#IDS[@]} -gt 0 ]] || { echo "$PROG: need at least one run id (e.g. 61-math-ablation, EXP-44) or --all-terminal"; exit 2; }

if [[ ! -f "$SUMMARY" ]]; then
  mkdir -p runs
  cat > "$SUMMARY" <<'HDR'
# Runs summary — one row per issue

Concise durable index (the OFFLINE fallback). The verdict SSOT is each issue's
close comment + the published report page (project.yaml `reports:`).

| id | date | verdict | headline | issue | PR |
|---|---|---|---|---|---|
| baseline | — | reference | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the permanent control) | — | — |
HDR
fi

# grep pattern with word boundaries so EXP-4 never matches EXP-44 / 4-foo never 44-foo.
bounded() { printf '(^|[^0-9A-Za-z-])%s([^0-9A-Za-z-]|$)' "$(sed 's/[][\.*^$(){}?+|/]/\\&/g' <<<"$1")"; }

folded=0
for RAW in "${IDS[@]}"; do
  # --- baseline guard, by NAME, before any parsing ---
  if [[ "$RAW" == "baseline" || "$RAW" == "EXP-3" || "$RAW" == "3" ]]; then
    echo "$PROG: refusing to de-bloat the baseline ($RAW) — it is the permanent control."; continue
  fi

  # --- resolve the id VERBATIM first (<N>-<slug> and named dirs), then older EXP-<N> ids ---
  ID=""; ISSUE_NUM=""
  if [[ -d "runs/$RAW" ]]; then
    ID="$RAW"
  elif [[ "$RAW" =~ ^EXP-([0-9]+)$ || "$RAW" =~ ^([0-9]+)$ ]]; then
    ISSUE_NUM="${BASH_REMATCH[1]}"; ID="EXP-$ISSUE_NUM"
  else
    ID="$RAW"   # ledger-only id (dir may already be gone)
  fi
  [[ "$ID" =~ ^([0-9]+)- ]] && ISSUE_NUM="${BASH_REMATCH[1]}"
  # run.json knows its issue even when the id has no number prefix
  [[ -z "$ISSUE_NUM" && -f "runs/$ID/run.json" ]] \
    && ISSUE_NUM=$(jq -r '.issue // empty' "runs/$ID/run.json" 2>/dev/null)
  if [[ "$ISSUE_NUM" == "3" ]]; then
    echo "$PROG: refusing to de-bloat the baseline (issue 3)."; continue
  fi

  RUNDIR="runs/$ID"

  # --- idempotency ---
  if [[ ! -d "$RUNDIR" ]] && grep -qE "$(bounded "$ID")" "$SUMMARY"; then
    echo "$PROG: $ID already folded — skipping."; continue
  fi

  # --- live-instance guard (EXTERNAL counts as live: operator-managed box) ---
  if [[ -f "$LEDGER" ]] && jq -e --arg id "$ID" \
      'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"))' \
      "$LEDGER" >/dev/null 2>&1; then
    echo "$PROG: $ID still has a live ledger row — tear it down first. Refusing."; continue
  fi

  # --- done-check: verdict.md OR a word-bounded SUMMARY row (LOG.md is retired) ---
  VERDICT="$(grep -m1 -oE 'VERDICT:[[:space:]]*(PASS|REVISE|STOP)' "$RUNDIR/verdict.md" 2>/dev/null | grep -oE 'PASS|REVISE|STOP' || true)"
  if [[ -z "$VERDICT" ]] && ! grep -qE "$(bounded "$ID")" "$SUMMARY" 2>/dev/null; then
    echo "$PROG: $ID is not done (no verdict.md, no SUMMARY row) — refusing to delete pending work."; continue
  fi
  [[ -z "$VERDICT" ]] && VERDICT="$(grep -E "$(bounded "$ID")" "$SUMMARY" 2>/dev/null | grep -m1 -oE 'PASS|STOP|REVISE' || echo 'done')"

  # --- concise SUMMARY row (run.json is the only detail source) ---
  TITLE=""; MILE=""
  [[ -f "$RUNDIR/run.json" ]] && TITLE=$(jq -r '.title // empty' "$RUNDIR/run.json" 2>/dev/null)
  [[ -z "$TITLE" ]] && TITLE="$ID"
  [[ -f "$RUNDIR/run.json" ]] && MILE=$(jq -r '.milestone // empty' "$RUNDIR/run.json" 2>/dev/null)
  [[ -n "$MILE" && "$MILE" != "none" && "$MILE" != "?" ]] && TITLE="[$MILE] $TITLE"
  PR="$(grep -hE "$(bounded "$ID")" "$SUMMARY" PROGRESS.md 2>/dev/null | grep -oE 'pull/[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
  MERGED="—"; [[ -n "$PR" ]] && MERGED="PR #$PR → \`$BASE_BRANCH\`"
  ISSUE_COL="—"; [[ -n "$ISSUE_NUM" ]] && ISSUE_COL="#$ISSUE_NUM"
  ROW="| $ID | $(date +%F) | $VERDICT | $TITLE | $ISSUE_COL | $MERGED |"

  echo "$PROG: folding $ID → $SUMMARY"
  echo "    $ROW"

  if ! grep -qE "$(bounded "$ID")" "$SUMMARY"; then
    if [[ "$DRY" == 1 ]]; then echo "  [dry-run] insert row into $SUMMARY";
    else
      TMP="$(mktemp)"; awk -v row="$ROW" '
        { print }
        /^\|---/ && !done { print row; done=1 }
      ' "$SUMMARY" > "$TMP" && mv "$TMP" "$SUMMARY"
    fi
  fi

  # --- remove bulky artifacts (no eval; git rm if tracked, else rm) ---
  if [[ -d "$RUNDIR" ]]; then
    if [[ "$DRY" == 1 ]]; then echo "  [dry-run] rm -rf $RUNDIR"
    else git rm -r -q --ignore-unmatch "$RUNDIR" 2>/dev/null || true; rm -rf "$RUNDIR"; fi
  fi
  # plan CACHE is derived (SSOT = the issue body) — always safe to drop
  if [[ -n "$ISSUE_NUM" && -f "$PLAN_CACHE_DIR/$ISSUE_NUM.md" ]]; then
    if [[ "$DRY" == 1 ]]; then echo "  [dry-run] rm $PLAN_CACHE_DIR/$ISSUE_NUM.md"
    else rm -f "$PLAN_CACHE_DIR/$ISSUE_NUM.md"; fi
  fi
  folded=$((folded+1))
done

# --- light tidy: drop handle files for instances already TORN_DOWN ---
if [[ -d .claude/state/vast-handles && "$DRY" == 0 && -f "$LEDGER" ]]; then
  for h in .claude/state/vast-handles/*.json; do
    [[ -e "$h" ]] || continue
    iid="$(basename "$h" .json)"
    if jq -e --arg i "$iid" 'select((.handles[]?.instance_id==$i or .instance_id==$i) and .status=="TORN_DOWN")' "$LEDGER" >/dev/null 2>&1; then
      git rm -q --ignore-unmatch "$h" 2>/dev/null || true; rm -f "$h"
    fi
  done
fi

echo "$PROG: DE_BLOATED count=$folded (baseline untouched). Review 'git status' / runs/SUMMARY.md, then commit."
