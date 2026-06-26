#!/usr/bin/env bash
# EXP-42 ON-BOX DRIVER — GPU busy from minute 1, zero idle gaps.
#
# Runs the 3 training cells back-to-back inside a single tmux, STRICTLY sequential:
#   run1 (fixed_linear@0.50) -> run2 (learned@0.50) -> run3 (no-projection)
# Each: 100 steps @ 1024 ctx, anchor delay_K=cadence=10, full batch, DYNAMIC batching.
#
# NO smoke phase (operator 2026-06-26): the FSDP +2-backward path was already
# validated (a prior smoke GATE_PASS on this exact code + run1 ran healthy through
# step 7). We go straight to training. on_fail=continue per the plan — a collapse is
# expected DATA, not a stop (the per-fire grad_proj_gain is the headline and is
# captured even on an early collapse). The training-log-monitor subagent catches
# NaN/OOM/Traceback within the first few steps if the new (dynamic-batching) path
# misbehaves.
#
# Launch:  tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'
# Watch:   tail -f /workspace/runs/EXP-42/drive.status
set -uo pipefail
BASE=/workspace/runs/EXP-42
mkdir -p "$BASE"
STATUS="$BASE/drive.status"
: > "$STATUS"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

log "DRIVE START (no smoke — direct to training, dynamic batching)"

# ---------- The 3 cells, STRICTLY sequential, no gaps ----------
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
