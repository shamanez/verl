#!/usr/bin/env bash
# On-box GPU-occupancy chain for issue #93: when cell a5b ends, launch cell a6
# with no idle gap. Runs detached in tmux session chain-93 so it survives the
# laptop going to sleep or any ssh drop.
#
# Fires on ANY termination of a5b, clean or crashed. That is deliberate: a6 is a
# pre-registered cell, not a known-broken config, so occupying the GPU with it
# is correct either way and the a5b log stays on disk for scoring.
set -uo pipefail

PREV=a6-prf-exactk-tis-bnorm-200
NEXT=a7-frlr-r48k28-notis-200
CHAINLOG=/workspace/chain-93b.log

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[$(stamp)] $*" >> "$CHAINLOG"; }

say "chain armed: PREV=$PREV NEXT=$NEXT; waiting for tmux run-93 to end"

# ---------------------------------------------------------------------------
# 1. Wait for the cell to terminate. tmux session death is the only signal
#    used: the pane runs the launcher directly and remain-on-exit is off, so
#    the session disappears exactly when the driver exits. A log-quiet
#    heuristic is deliberately NOT used, because relaunching onto a live run
#    would put two trainers on one GPU.
# ---------------------------------------------------------------------------
while tmux has-session -t run-93 2>/dev/null; do sleep 60; done
LASTSTEP="$(grep -aoE 'global_step:[0-9]+' /workspace/runs/$PREV/train.log 2>/dev/null | tail -1)"
say "run-93 gone; $PREV terminated at ${LASTSTEP:-unknown}"

# ---------------------------------------------------------------------------
# 2. Snapshot the terminal metrics. WandB drops the final step to an atexit
#    race, so the on-box log is the only record of it.
# ---------------------------------------------------------------------------
mkdir -p "/workspace/runs/$PREV/final"
tail -600 "/workspace/runs/$PREV/train.log" > "/workspace/runs/$PREV/final/tail600.txt" 2>/dev/null || true
grep -aE 'global_step:(19[0-9]|200) ' "/workspace/runs/$PREV/train.log" \
  > "/workspace/runs/$PREV/final/final_steps.txt" 2>/dev/null || true
grep -acE 'Traceback|CUDA out of memory|FATAL' "/workspace/runs/$PREV/train.log" \
  > "/workspace/runs/$PREV/final/error_count.txt" 2>/dev/null || true
say "snapshotted terminal metrics for $PREV"

# ---------------------------------------------------------------------------
# 3. Wait for the GPU to actually free. Residual ray workers commonly hold
#    memory for a minute after the driver exits. Kill by process NAME only
#    (pkill -x); never pkill -f, which would match this script, and never a
#    bare `ray stop`.
# ---------------------------------------------------------------------------
for i in $(seq 1 24); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  used="${used:-999999}"
  if [[ "$used" -lt 4000 ]]; then
    say "gpu free (${used} MiB) after ${i} checks"
    break
  fi
  say "gpu holds ${used} MiB, waiting (check $i/24)"
  if [[ "$i" -ge 5 ]]; then
    for p in ray raylet gcs_server plasma_store pt_main_thread python3 python; do
      pkill -x "$p" 2>/dev/null || true
    done
    say "sent SIGTERM to residual workers by name"
  fi
  sleep 30
done

# ---------------------------------------------------------------------------
# 4. Re-point the reaper heartbeat BEFORE launching, and seed the target so
#    its mtime is fresh during bring-up. /workspace/train.log is the run.json
#    remote_log: if it goes stale for 30 min the reaper destroys the box.
# ---------------------------------------------------------------------------
mkdir -p "/workspace/runs/$NEXT"
echo "[$(stamp)] chain: bring-up starting for $NEXT" >> "/workspace/runs/$NEXT/train.log"
ln -sfn "/workspace/runs/$NEXT/train.log" /workspace/train.log
say "heartbeat symlink -> /workspace/runs/$NEXT/train.log"

# ---------------------------------------------------------------------------
# 5. Launch. The engine appends to $LOG itself (tee -a), so no second tee.
# ---------------------------------------------------------------------------
tmux new-session -d -s run-93 "bash /workspace/launch_a7.sh"
sleep 120
if tmux has-session -t run-93 2>/dev/null; then
  say "LAUNCHED $NEXT: tmux run-93 live, log /workspace/runs/$NEXT/train.log"
else
  say "LAUNCH FAILED for $NEXT: tmux run-93 not alive after 120s; inspect /workspace/runs/$NEXT/train.log"
fi

# ---------------------------------------------------------------------------
# 6. Keep watching a6 through bring-up so a fast crash is recorded rather than
#    leaving the GPU quietly idle until someone looks.
# ---------------------------------------------------------------------------
for i in $(seq 1 40); do
  sleep 60
  if ! tmux has-session -t run-93 2>/dev/null; then
    say "ALERT: $NEXT died during bring-up watch (check $i/40). GPU IS NOW IDLE."
    exit 1
  fi
  s="$(grep -aoE 'global_step:[0-9]+' "/workspace/runs/$NEXT/train.log" 2>/dev/null | tail -1)"
  if [[ -n "$s" ]]; then
    say "$NEXT training: $s"
    exit 0
  fi
done
say "WARN: $NEXT alive but no global_step after 40 min of bring-up watch"
exit 0
