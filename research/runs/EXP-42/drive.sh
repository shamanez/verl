#!/usr/bin/env bash
# EXP-42 ON-BOX DRIVER — weight-projection accuracy study (2-regime, single-GPU).
# Re-materialised 2026-06-29 for the 2-regime design. Runs the two regimes
# back-to-back inside ONE tmux, STRICTLY sequential:
#   regimeA (plain GRPO, COMM_EFF_ENABLED=false)
#   regimeB (PowerSGD r=77, codec only — anchor + spectral OFF)
# Each: 80 steps (=160 optimizer ticks) @ resp=1024, dynamic batching,
# probe.weight_traj.enabled=true (per-tick count-sketch + bounded exact calib).
#
# on_fail policy (per plan):
#   regimeA = STOP   (if plain GRPO can't run on 1xH200, the single-GPU premise
#                     is broken — fix before B; do NOT waste B's compute)
#   regimeB = continue (a collapse is ALLOWED data — the codec trajectory IS the
#                     measurement; the per-tick sketch is captured even on collapse)
#
# Launch:  tmux new -d -s exp42 'bash /workspace/runs/EXP-42/drive.sh'
# Watch:   tail -f /workspace/runs/EXP-42/drive.status
set -uo pipefail
BASE=/workspace/runs/EXP-42
mkdir -p "$BASE"
STATUS="$BASE/drive.status"
: > "$STATUS"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

log "DRIVE START — 2 regimes sequential (regimeA plain-GRPO -> regimeB powersgd-r77 codec-only)"

for REGIME in regimeA regimeB; do
  log "PHASE $REGIME START"
  bash "$BASE/run_cell.sh" "$REGIME"
  RC=$?
  OUT="$BASE/$REGIME"
  IL="$OUT/train_${REGIME}_internal.log"
  NPZ=$(ls "$OUT/weights/"sketch_tick_*.npz 2>/dev/null | wc -l | tr -d ' ')
  MAN=$(wc -l < "$OUT/weights/manifest.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  CAL=$(wc -l < "$OUT/weights/calib.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  VAL=$(grep -oE "val-core[^ ]*acc/mean@1:[0-9.]+" "$IL" 2>/dev/null | tail -1 || echo "n/a")
  log "PHASE $REGIME DONE rc=$RC sketch_npz=$NPZ manifest_rows=$MAN calib_rows=$CAL last_val=$VAL"
  # on_fail=stop gate for regimeA keys on the CAPTURED TRAJECTORY LENGTH, NOT rc.
  # The launcher routinely exits rc!=0 from BENIGN atexit teardown noise (a
  # DataLoader-worker SIGKILL during the final wandb flush + a stale
  # launcher-internal done.flag path) AFTER training + val + checkpoint + sync +
  # the FULL per-tick sketch capture all complete (observed regimeA 2026-06-29:
  # rc=1 but 80/80 steps + 160 sketches). Gating on rc falsely aborts the chain;
  # gate on the trajectory we actually paid for instead.
  EXPECT=$((2 * 80))   # 2 optimizer ticks/step at batch128/mini64
  if [[ "$REGIME" == "regimeA" ]] && (( NPZ < EXPECT - 20 )); then
    log "REGIME A INCOMPLETE (sketch_npz=$NPZ < ~$EXPECT) — single-GPU plain-GRPO premise broken; STOP before B (on_fail=stop)"
    break
  fi
  if [[ "$REGIME" == "regimeA" && "$RC" -ne 0 ]]; then
    log "REGIME A launcher rc=$RC but trajectory complete (sketch_npz=$NPZ ~ $EXPECT) — benign teardown rc, proceeding to regimeB"
  fi
done
log "ALL_DONE"
echo "EXP42_DRIVE_COMPLETE"
