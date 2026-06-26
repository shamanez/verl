#!/usr/bin/env bash
# EXP-42 ON-BOX DRIVER — GPU busy from minute 1, zero idle gaps.
#
# One command runs the whole pipeline back-to-back inside a single tmux:
#   smoke (fast gate) -> run1 -> run2 -> run3
# The smoke (cadence=2/delay_K=2, reduced batch) reaches a PROJECTING fire in ~2-3
# min and exercises the +2-backward path BEFORE committing to the long runs. If the
# smoke GATE fails (no projecting probe fire, or a crash), the chain ABORTS and no
# cell launches — the only legitimate idle, and the signal to fix on the branch.
# Otherwise the 3 full runs (full batch, 50 steps, all 4 GPUs saturated) run
# strictly sequentially with no gaps between them. on_fail=continue per the plan
# (a collapse is expected DATA, not a stop — the per-fire grad_proj_gain is the
# headline and is captured even on an early collapse).
#
# Launch:  tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'
# Watch:   tail -f /workspace/runs/EXP-42/drive.status
set -uo pipefail
BASE=/workspace/runs/EXP-42
mkdir -p "$BASE"
STATUS="$BASE/drive.status"
: > "$STATUS"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

log "DRIVE START"

# ---------- Phase 0: smoke probe = the FSDP/backend hard gate ----------
log "PHASE smoke (fixed_linear) START"
bash "$BASE/smoke.sh" fixed_linear
SMOKE_RC=$?
SL="$BASE/smoke_fixed_linear/train_smoke_internal.log"
DL="$BASE/smoke_fixed_linear/driver.log"
PROBE_FIRES=$(grep -cE "\[grad-proj-probe\] .*lookahead_active=True" "$SL" 2>/dev/null || echo 0)
XDEV_BAD=$(grep -oE "cross_rank_max_rel_dev=[0-9.eE+-]+" "$SL" 2>/dev/null | awk -F= '$2+0>1e-3{n++} END{print n+0}')
CRASH=$(grep -cE "Traceback|CUDA out of memory|OutOfMemory|AssertionError|NaN|Killed" "$SL" "$DL" 2>/dev/null || echo 0)
log "smoke rc=$SMOKE_RC projecting_fires=$PROBE_FIRES cross_rank_bad=$XDEV_BAD crash_hits=$CRASH"
if [[ "$SMOKE_RC" -ne 0 || "$PROBE_FIRES" -lt 1 || "$CRASH" -gt 0 || "$XDEV_BAD" -gt 0 ]]; then
  log "GATE_FAIL — NOT launching cells. Inspect $SL / $DL, fix on the branch, re-drive."
  echo "EXP42_DRIVE_ABORT"
  exit 1
fi
log "GATE_PASS — smoke clean; launching the 3 runs"

# ---------- Phases 1-3: the 3 cells, STRICTLY sequential, no gaps ----------
for CELL in run1 run2 run3; do
  log "PHASE $CELL START"
  bash "$BASE/run_cell.sh" "$CELL"
  RC=$?
  IL="$BASE/$CELL/train_${CELL}_internal.log"
  FIRES=$(grep -cE "\[grad-proj-probe\]" "$IL" 2>/dev/null || echo 0)
  VAL=$(grep -oE "val-core[^ ]*acc/mean@1:[0-9.]+" "$IL" 2>/dev/null | tail -1 || echo "n/a")
  log "PHASE $CELL DONE rc=$RC grad_proj_fires=$FIRES last_val=$VAL"
done
log "ALL_DONE"
echo "EXP42_DRIVE_COMPLETE"
