#!/usr/bin/env bash
# de-bloat — fold a COMPLETED experiment into runs/SUMMARY.md and remove its bulky
# artifacts (run dir, plan file, stale handles). HUMAN-ONLY — see the gate below.
#
# Hard guards:
#   - OPERATOR GATE: refuses unless DEBLOAT_OPERATOR_ACK=1 is set. The autonomous
#     loop must NEVER set it; only /de-bloat typed by the human does (SKILL.md is
#     disable-model-invocation, so the model cannot auto-fire this skill).
#   - NEVER the baseline (ids baseline / 3 / EXP-3).
#   - Refuses a live (RUNNING/PROVISIONED/EXTERNAL) ledger row.
#   - Refuses an undone experiment (no verdict.md AND no LOG entry).
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
# Research runs — summary

Concise record of what has run on this harness. Full per-experiment artifacts are
pruned (folded here by the `de-bloat` skill); the durable record is here + git history + merged code.

| id | milestone | what | result | merged |
|---|---|---|---|---|
| **baseline** | M1 | Dense GRPO, Qwen2.5-1.5B-Instruct on GSM8K — verl unmodified (the control) | reference curve | — |
HDR
fi

# grep pattern with word boundaries so EXP-4 never matches EXP-44 / 4-foo never 44-foo.
bounded() { printf '(^|[^0-9A-Za-z-])%s([^0-9A-Za-z-]|$)' "$(sed 's/[][\.*^$(){}?+|/]/\\&/g' <<<"$1")"; }

field() { grep -m1 -E "^-? *$1:" "$2" 2>/dev/null | sed -E "s/^-? *$1:[[:space:]]*//" | sed 's/[[:space:]]*(#.*)?$//'; }

folded=0
for RAW in "${IDS[@]}"; do
  # --- baseline guard, by NAME, before any parsing ---
  if [[ "$RAW" == "baseline" || "$RAW" == "EXP-3" || "$RAW" == "3" ]]; then
    echo "$PROG: refusing to de-bloat the baseline ($RAW) — it is the permanent control."; continue
  fi

  # --- resolve the id VERBATIM first (new <N>-<slug> and legacy slug dirs), then legacy EXP-<N> ---
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
  PLAN=""; [[ -n "$ISSUE_NUM" ]] && PLAN=".claude/plans/$ISSUE_NUM.md"

  # --- idempotency ---
  if [[ ! -d "$RUNDIR" && ( -z "$PLAN" || ! -f "$PLAN" ) ]] && grep -qE "$(bounded "$ID")" "$SUMMARY"; then
    echo "$PROG: $ID already folded — skipping."; continue
  fi

  # --- live-instance guard (EXTERNAL counts as live: operator-managed box) ---
  if [[ -f "$LEDGER" ]] && jq -e --arg id "$ID" \
      'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"))' \
      "$LEDGER" >/dev/null 2>&1; then
    echo "$PROG: $ID still has a live ledger row — tear it down first. Refusing."; continue
  fi

  # --- done-check: verdict.md OR a word-bounded LOG entry ---
  VERDICT="$(grep -m1 -oE 'VERDICT:[[:space:]]*(PASS|REVISE|STOP)' "$RUNDIR/verdict.md" 2>/dev/null | grep -oE 'PASS|REVISE|STOP' || true)"
  if [[ -z "$VERDICT" ]] && ! grep -qE "$(bounded "$ID")" LOG.md 2>/dev/null; then
    echo "$PROG: $ID is not done (no verdict.md, no LOG.md entry) — refusing to delete pending work."; continue
  fi
  [[ -z "$VERDICT" ]] && VERDICT="$(grep -E "$(bounded "$ID")" LOG.md 2>/dev/null | grep -m1 -oE 'PASS|STOP|REVISE' || echo 'done')"

  # --- concise SUMMARY row (plan → run.json → LOG fallbacks) ---
  TITLE=""; MILE=""
  [[ -n "$PLAN" && -f "$PLAN" ]] && { TITLE="$(field title "$PLAN")"; MILE="$(field milestone "$PLAN")"; }
  [[ -z "$TITLE" && -f "$RUNDIR/run.json" ]] && TITLE=$(jq -r '.title // empty' "$RUNDIR/run.json" 2>/dev/null)
  [[ -z "$TITLE" ]] && TITLE="$(grep -E "$(bounded "$ID")" LOG.md 2>/dev/null | head -1 | sed -E 's/^#+ *//' || true)"
  [[ -z "$TITLE" ]] && TITLE="$ID"
  [[ -z "$MILE" && -f "$RUNDIR/run.json" ]] && MILE=$(jq -r '.milestone // empty' "$RUNDIR/run.json" 2>/dev/null)
  [[ -z "$MILE" ]] && MILE="?"
  PR="$(grep -hE "$(bounded "$ID")" LOG.md PROGRESS.md 2>/dev/null | grep -oE 'pull/[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
  MERGED="—"; [[ -n "$PR" ]] && MERGED="PR #$PR → \`$BASE_BRANCH\`"
  ROW="| $ID | $MILE | $TITLE | $VERDICT | $MERGED |"

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
  if [[ -n "$PLAN" && -f "$PLAN" ]]; then
    if [[ "$DRY" == 1 ]]; then echo "  [dry-run] rm $PLAN"
    else git rm -q --ignore-unmatch "$PLAN" 2>/dev/null || true; rm -f "$PLAN"; fi
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
