#!/usr/bin/env bash
# test_debloat_invariant.sh — hermetic proof of the P3 invariant:
#   "a terminal issue's preconditions never require its run dir."
# Exercises de-bloat's guards + batch mode and _lib.sh's graceful degradation
# in a throwaway sandbox (no git repo, no network — gh is stubbed to fail).
# Run from research/:  bash scripts/test_debloat_invariant.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # research/
DEBLOAT="$HERE/.claude/skills/de-bloat/run.sh"
LIB="$HERE/.claude/skills/_lib.sh"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT
FAILS=0
say()  { printf '  %s\n' "$*"; }
pass() { say "ok   - $1"; }
fail() { say "FAIL - $1"; FAILS=$((FAILS+1)); }
check() { local msg="$1"; shift; if "$@"; then pass "$msg"; else fail "$msg"; fi; }

# ---- sandbox: a minimal research/ with 4 runs in different lifecycle states ----
mkdir -p "$SANDBOX/runs" "$SANDBOX/.claude/state/plan-cache" "$SANDBOX/.claude/plans" "$SANDBOX/stub-bin"
cat > "$SANDBOX/stub-bin/gh" <<'EOF'
#!/usr/bin/env bash
exit 1   # hermetic: no network — plan_fetch must degrade, never hang
EOF
chmod +x "$SANDBOX/stub-bin/gh"
export PATH="$SANDBOX/stub-bin:$PATH"

echo "# Progress" > "$SANDBOX/PROGRESS.md"
echo "line2" >> "$SANDBOX/PROGRESS.md"
echo "line3" >> "$SANDBOX/PROGRESS.md"

# 90-terminal-run: verdict + TORN_DOWN row  → must fold + delete
mkdir -p "$SANDBOX/runs/90-terminal-run"
echo "VERDICT: PASS" > "$SANDBOX/runs/90-terminal-run/verdict.md"
echo '{"issue":90,"milestone":"M9","title":"synthetic terminal run"}' > "$SANDBOX/runs/90-terminal-run/run.json"
echo "cached plan" > "$SANDBOX/.claude/state/plan-cache/90.md"
# 91-live-run: RUNNING ledger row               → must be refused
mkdir -p "$SANDBOX/runs/91-live-run"
echo "VERDICT: PASS" > "$SANDBOX/runs/91-live-run/verdict.md"
# 92-pending-run: no verdict, no LOG entry      → must be refused
mkdir -p "$SANDBOX/runs/92-pending-run"
# baseline: permanent control                   → must be refused
mkdir -p "$SANDBOX/runs/baseline"
echo "VERDICT: PASS" > "$SANDBOX/runs/baseline/verdict.md"

cat > "$SANDBOX/.claude/state/runs.jsonl" <<'EOF'
{"id":"90-terminal-run","issue":90,"status":"TORN_DOWN","dph":2.0}
{"id":"91-live-run","issue":91,"status":"RUNNING","dph":2.0}
EOF

run_debloat() { DEBLOAT_OPERATOR_ACK=1 CLAUDE_PROJECT_DIR="$SANDBOX" bash "$DEBLOAT" "$@" 2>&1; }

echo "== gate =="
out=$(DEBLOAT_OPERATOR_ACK=0 CLAUDE_PROJECT_DIR="$SANDBOX" bash "$DEBLOAT" 90-terminal-run 2>&1) && rc=0 || rc=$?
check "refuses without operator ack (rc=5)" test "$rc" -eq 5

echo "== batch fold (--all-terminal) =="
out=$(run_debloat --all-terminal)
check "terminal run folded"            grep -q "folding 90-terminal-run" <<<"$out"
check "live run refused"               grep -q "91-live-run still has a live ledger row" <<<"$out"
check "pending run refused"            grep -q "92-pending-run is not done" <<<"$out"
check "baseline refused"               grep -q "refusing to de-bloat the baseline" <<<"$out"
check "terminal run dir deleted"       test ! -d "$SANDBOX/runs/90-terminal-run"
check "plan cache deleted"             test ! -f "$SANDBOX/.claude/state/plan-cache/90.md"
check "live run dir intact"            test -d "$SANDBOX/runs/91-live-run"
check "pending run dir intact"         test -d "$SANDBOX/runs/92-pending-run"
check "baseline dir intact"            test -d "$SANDBOX/runs/baseline"
check "SUMMARY row written"            grep -qE "\| 90-terminal-run \| [0-9-]+ \| PASS \| \[M9\] synthetic terminal run \| #90 \|" "$SANDBOX/runs/SUMMARY.md"

echo "== idempotency =="
out=$(run_debloat 90-terminal-run)
check "second fold is a no-op"         grep -q "already folded" <<<"$out"
check "exactly one SUMMARY row"        test "$(grep -c "90-terminal-run" "$SANDBOX/runs/SUMMARY.md")" -eq 1

echo "== the invariant: terminal issue needs NO run dir (lib degradation) =="
# shellcheck disable=SC1090
( RESEARCH_DIR="$SANDBOX" CLAUDE_PROJECT_DIR="$SANDBOX" source "$LIB"
  row=$(ledger_row_by_issue 90)
  [[ "$(jq -r .status <<<"$row")" == "TORN_DOWN" ]] || exit 1     # labels+ledger still resolve the issue
  [[ "$(snapshot_get 90-terminal-run .milestone none)" == "none" ]] || exit 1   # missing run.json → default, no error
  [[ "$(plan_field 90 kind experiment)" == "experiment" ]] || exit 1            # no plan anywhere + gh dead → default
  plan_exists 90 && exit 1                                        # and plan_exists says NO rather than hanging
  exit 0
) && pass "ledger resolves; snapshot_get/plan_field/plan_exists degrade cleanly" \
  || fail "lib degradation after run-dir deletion"

echo "== close_cleanup guards (gh dead ⇒ status:done unprovable ⇒ refuse) =="
mkdir -p "$SANDBOX/runs/93-closed-run"
out=$(CLAUDE_PROJECT_DIR="$SANDBOX" RESEARCH_DIR="$SANDBOX" bash "$HERE/scripts/close_cleanup.sh" 93 93-closed-run 2>&1) && rc=0 || rc=$?
check "close_cleanup refuses without provable status:done (rc=3)" test "$rc" -eq 3
check "close_cleanup left the run dir alone"                      test -d "$SANDBOX/runs/93-closed-run"

echo "== progress_sweep + ledger_compact =="
( RESEARCH_DIR="$SANDBOX" CLAUDE_PROJECT_DIR="$SANDBOX" source "$LIB"
  progress "[analyst #94] verdict=PASS"
  progress "teardown 94-sweep-me reason=verdict"
  progress "STANDING note that must survive"
  progress_sweep 94 94-sweep-me
  grep -q 'STANDING note that must survive' "$PROGRESS" || exit 1   # unrelated lines survive
  grep -qE '#94|94-sweep-me' "$PROGRESS" && exit 1                  # the issue's ticks are gone
  ledger_append '{"id":"95-compact-me","issue":95,"status":"TORN_DOWN","launch_attempts":1}'
  ledger_append '{"id":"95-compact-me","issue":95,"status":"TORN_DOWN","launch_attempts":2}'
  ledger_compact 95-compact-me
  [[ "$(jq -c 'select(.id=="95-compact-me")' "$LEDGER" | wc -l | tr -d ' ')" == "1" ]] || exit 1
  [[ "$(ledger_row 95-compact-me | jq -r .launch_attempts)" == "2" ]] || exit 1   # the LAST row survives
  ledger_compact 91-live-run && exit 1                              # refuses a live id
  # the review-confirmed blind spot: an EARLIER live row hidden behind a later
  # TORN_DOWN row (failed destroy of box A, successful relaunch on box B) must
  # ALSO block compaction — tail-1 liveness alone would delete the RUNNING row.
  ledger_append '{"id":"96-hidden-live","issue":96,"status":"RUNNING","handles":[{"instance_id":"111"}]}'
  ledger_append '{"id":"96-hidden-live","issue":96,"status":"TORN_DOWN","handles":[{"instance_id":"222"}]}'
  ledger_compact 96-hidden-live && exit 1                           # must refuse
  [[ "$(jq -c 'select(.id=="96-hidden-live")' "$LEDGER" | wc -l | tr -d ' ')" == "2" ]] || exit 1  # both rows intact
  exit 0
) && pass "progress_sweep keeps standing notes; ledger_compact keeps last row, refuses live ids (incl. hidden earlier live row)" \
  || fail "progress_sweep / ledger_compact"

echo "== progress cap =="
( RESEARCH_DIR="$SANDBOX" CLAUDE_PROJECT_DIR="$SANDBOX" source "$LIB"
  for i in $(seq 1 500); do progress "tick $i"; done
  n=$(wc -l < "$PROGRESS" | tr -d '[:space:]')
  [[ "$n" -le $((PROGRESS_CAP_LINES + 44)) ]] || exit 1   # capped: prune hysteresis tops out at CAP+40 + header
  head -1 "$PROGRESS" | grep -q '^# Progress' || exit 1   # header survives the cap
  tail -1 "$PROGRESS" | grep -q 'tick 500' || exit 1      # newest tick survives
  exit 0
) && pass "PROGRESS capped, header + newest ticks preserved" \
  || fail "progress cap"

echo
if [[ "$FAILS" -eq 0 ]]; then echo "ALL CHECKS PASSED"; else echo "$FAILS CHECK(S) FAILED"; exit 1; fi
