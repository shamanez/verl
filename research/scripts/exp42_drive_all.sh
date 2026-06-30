#!/usr/bin/env bash
# EXP-42 WIDENED (select_all) ON-BOX DRIVER — completeness extension.
# Re-runs BOTH regimes with the weight-trajectory instrument widened to sketch
# EVERY 1-D/2-D param (decoder 196 + the projector-EXCLUDED token embeddings /
# RMSNorm gains / attention biases), so the offline sweep can measure whether
# linear weight projection would help/hurt on the excluded params too.
#
# Output goes to a SEPARATE run dir (EXP-42-all) so the narrow 196-matrix study
# (runs/EXP-42/{regimeA,regimeB}/weights) is preserved untouched. WandB run names
# get the -all suffix (exp42-regimeA-all / exp42-regimeB-all) so they don't
# collide with the narrow runs.
#
# Launch:  tmux new -d -s exp42all 'bash /workspace/runs/EXP-42/drive_all.sh'
# Watch:   tail -f /workspace/runs/EXP-42-all/drive.status
set -uo pipefail
export RUN_DIR=/workspace/runs/EXP-42-all
export WEIGHT_TRAJ_SELECT_ALL=true
export EXPN_SUFFIX=-all
BASE=/workspace/runs/EXP-42
ADIR=/workspace/runs/EXP-42-all
mkdir -p "$ADIR"
STATUS="$ADIR/drive.status"
: > "$STATUS"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

log "WIDENED DRIVE START — select_all=true (ALL matrices), RUN_DIR=$RUN_DIR (regimeA -> regimeB)"

for REGIME in regimeA regimeB; do
  log "PHASE $REGIME START (select_all=true)"
  bash "$BASE/run_cell.sh" "$REGIME"
  RC=$?
  OUT="$ADIR/$REGIME"
  NPT=$(ls "$OUT/weights/full/"step_*.pt 2>/dev/null | wc -l | tr -d ' ')
  MAN=$(wc -l < "$OUT/weights/full_manifest.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  NMAT=$(head -1 "$OUT/weights/full_manifest.jsonl" 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['n_matrices'])" 2>/dev/null || echo "?")
  VAL=$(grep -oE "val-core[^ ]*acc/mean@1:[0-9.]+" "$OUT/train_${REGIME}_internal.log" 2>/dev/null | tail -1 || echo "n/a")
  log "PHASE $REGIME DONE rc=$RC full_step_pt=$NPT manifest_rows=$MAN n_matrices=$NMAT last_val=$VAL"
  # Gate on captured-trajectory length, NOT rc (launcher exits rc=1 from benign atexit teardown noise).
  # Full weights dump once per TRAINING STEP, so EXPECT = total_training_steps (80), not ticks.
  EXPECT=80
  if [[ "$REGIME" == "regimeA" ]] && (( NPT < EXPECT - 10 )); then
    log "WIDENED REGIME A INCOMPLETE (full_step_pt=$NPT < ~$EXPECT) — STOP before B"
    break
  fi
done
log "ALL_DONE"
echo "EXP42_WIDENED_COMPLETE"
