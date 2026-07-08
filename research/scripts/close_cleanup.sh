#!/usr/bin/env bash
# close_cleanup.sh <N> <id> [--dry-run] — /close's FINAL step: remove a finished
# issue's local footprint (operator directive 2026-07-08: nothing about a done
# run lives in the harness checkout — the durable record is the issue close
# comment + the published report page + runs/SUMMARY.md + git history).
#
# Deletes / compacts, in order:
#   1. runs/<id>/                      (evidence, not state — run_dir_invariant)
#   2. .claude/state/plan-cache/<N>.md (derived cache; plan SSOT = issue body)
#   3. .claude/state/vast-handles/*    for instances whose row is TORN_DOWN
#      (incl. *.reaped markers)
#   4. ledger rows for <id> compacted to the single terminal row
#   5. PROGRESS.md ticks mentioning #<N> / <id> (progress_sweep)
#
# HARD GUARDS (each refuses with a named reason, rc 3 — never a retry loop):
#   G1 issue label is status:done            (the record is on GitHub)
#   G2 NO row for <id> is live — ANY row, not just the last (a failed destroy
#      can leave an earlier RUNNING row behind a later TORN_DOWN one)
#   G3 runs/SUMMARY.md carries a row for <id> (offline fallback exists)
#   G4 the published report page exists in the reports repo (project.yaml
#      reports:) — the rich record must survive the deletion below
# gh unreachable ⇒ refuse (G1 cannot be proven) — re-run when online; this is
# a cleanup, deferring it costs nothing.
#
# Unlike de-bloat (human-only batch fallback for leftover dirs), this script is
# sanctioned for the autonomous loop BECAUSE of the guards: it only ever
# removes what /close has already durably published elsewhere.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"                       # research/
# shellcheck disable=SC1091
source "$HERE/.claude/skills/_lib.sh"

N="${1:-}"; ID="${2:-}"; DRY=0
[[ "${3:-}" == "--dry-run" ]] && DRY=1
[[ -n "$N" && -n "$ID" ]] || die "usage: close_cleanup.sh <issue-number> <run-id> [--dry-run]"

bounded() { printf '(^|[^0-9A-Za-z-])%s([^0-9A-Za-z-]|$)' "$(sed 's/[][\.*^$(){}?+|/]/\\&/g' <<<"$1")"; }

# G1 — status:done on the issue (this also implies the close comment exists).
st=$(issue_status "$N") || true
[[ "$st" == "done" ]] || die "#$N is status:${st:-unknown} (or gh unreachable) — cleanup only runs AFTER /close labeled it done"

# G2 — no live box: ANY row for the id (relaunch histories can hide a live
# row behind a later TORN_DOWN one; same all-rows pattern as de-bloat).
if [[ -f "$LEDGER" ]] && jq -e --arg id "$ID" \
    'select(.id==$id and (.status=="RUNNING" or .status=="PROVISIONED" or .status=="EXTERNAL"))' \
    "$LEDGER" >/dev/null 2>&1; then
  die "$ID still has a live ledger row — teardown first (check ALL rows: jq -c 'select(.id==\"$ID\")' $LEDGER)"
fi

# G3 — the offline fallback row exists.
grep -qE "$(bounded "$ID")" "$RUNS_DIR/SUMMARY.md" 2>/dev/null \
  || die "runs/SUMMARY.md has no row for $ID — run the log-writer (it writes the SUMMARY row) before cleanup"

# G4 — the published report page exists (project.yaml reports:). Deleting the
# run dir before the page exists would destroy the only rich copy.
REPORT_REPO=$(awk -F': ' '/^  repo_dir:/{sub(/#.*/,"",$2);gsub(/[ "]/,"",$2);print $2; exit}' "$RESEARCH_DIR/.claude/project.yaml" 2>/dev/null || true)
REPORT_RUNS=$(awk -F': ' '/^  runs_dir:/{sub(/#.*/,"",$2);gsub(/[ "]/,"",$2);print $2; exit}' "$RESEARCH_DIR/.claude/project.yaml" 2>/dev/null || true)
[[ -n "$REPORT_REPO" && -f "$REPORT_REPO/${REPORT_RUNS:-runs}/$ID.html" ]] \
  || die "no report page at ${REPORT_REPO:-<reports.repo_dir unset>}/${REPORT_RUNS:-runs}/$ID.html — publish first (python3 scripts/publish_run_report.py --issue $N --run-id $ID)"

run() { if [[ "$DRY" == 1 ]]; then echo "  [dry-run] $*"; else "$@"; fi; }

echo "close_cleanup: #$N / $ID (dry-run=$DRY)"

# 1. run dir
if [[ -d "$RUNS_DIR/$ID" ]]; then
  run git -C "$RESEARCH_DIR" rm -r -q --ignore-unmatch "runs/$ID" 2>/dev/null || true
  run rm -rf "$RUNS_DIR/$ID"
  echo "  removed runs/$ID/"
fi

# 2. plan cache (derived — SSOT is the issue body)
[[ -f "$PLAN_CACHE_DIR/$N.md" ]] && { run rm -f "$PLAN_CACHE_DIR/$N.md"; echo "  removed plan-cache/$N.md"; }

# 3. torn-down handles (+ reaped markers)
if [[ -d "$STATE_DIR/vast-handles" && -f "$LEDGER" ]]; then
  for h in "$STATE_DIR"/vast-handles/*.json "$STATE_DIR"/vast-handles/*.json.reaped; do
    [[ -e "$h" ]] || continue
    iid="$(basename "$h" | sed 's/\.json\(\.reaped\)*$//')"
    if jq -e --arg i "$iid" 'select((any(.handles[]?.instance_id // empty; (.|tostring)==$i)) and .status=="TORN_DOWN")' \
        "$LEDGER" >/dev/null 2>&1; then
      run rm -f "$h"; echo "  removed stale handle $(basename "$h")"
    fi
  done
fi

# 4. ledger compaction (refuses internally if the id is somehow live)
if [[ "$DRY" == 0 ]]; then
  ledger_compact "$ID" || echo "  ledger not compacted (live row?) — left as-is"
else
  echo "  [dry-run] ledger_compact $ID"
fi

# 5. PROGRESS sweep
if [[ "$DRY" == 0 ]]; then progress_sweep "$N" "$ID"; else echo "  [dry-run] progress_sweep $N $ID"; fi

echo "close_cleanup: DONE — durable record: issue #$N close comment · report page · runs/SUMMARY.md row"
