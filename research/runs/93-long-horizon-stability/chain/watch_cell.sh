#!/usr/bin/env bash
# Generic per-cell watcher for issue #93. Takes the cell name as $1 so it can
# never be pointed at the wrong run by editing a constant.
#
# THREE watcher bugs preceded this file. Each was a variation of one mistake,
# using SHARED state to reason about a SPECIFIC cell:
#
#   1. watch93.sh compared the previous cell's step against the shared tmux
#      session name, so when the next cell took over it read "session live, old
#      cell not advancing" as a stall.
#   2. watch93.sh's exit condition grepped tail -1 of the chain log for
#      "LAUNCHED", but the chain appends a line after that, so it could never
#      fire.
#   3. watch_a6.sh keyed progress on a6's own step (fixing 1) but kept the
#      SHARED tmux session as its terminal condition, justified by "nothing is
#      chained after a6". a7 was then chained an hour later, breaking that
#      premise, so it ran forever re-reporting a finished cell's errors.
#
# The lesson: a cell-specific watcher must decide "am I done" from cell-specific
# evidence. This one uses three independent terminal conditions, any of which is
# sufficient, and none of which depends on another cell's absence:
#
#   a. the heartbeat symlink no longer points at THIS cell's log  -> succeeded
#   b. this cell reached its final step AND logged a val          -> complete
#   c. no tmux session at all                                     -> GPU idle
set -uo pipefail
CELL="${1:?usage: watch_cell.sh <cell-name> [total-steps]}"
TOTAL="${2:-200}"
LOG="/workspace/runs/$CELL/train.log"
SSHC=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes \
      -i "$HOME/.ssh/vast_ai" -p 8602 root@50.46.253.92)

fails=0; prev=-1; stale=0; milestone=0; reported_err=0

while true; do
  out="$("${SSHC[@]}" "s=\$(grep -aoE 'global_step:[0-9]+' $LOG 2>/dev/null | tail -1 | cut -d: -f2)
t=\$(tmux ls 2>/dev/null | wc -l | tr -d ' ')
h=\$(readlink /workspace/train.log 2>/dev/null)
v=\$(grep -ac 'val-core' $LOG 2>/dev/null || echo 0)
e=\$(grep -acE 'Traceback|CUDA out of memory|FATAL:' $LOG 2>/dev/null || echo 0)
g=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo \"step=\${s:-0} sessions=\$t val=\$v err=\$e gpu=\${g:-NA}\"
echo \"heartbeat=\$h\"" 2>/dev/null)"

  if [ -z "$out" ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 3 ]; then
      echo "$CELL: BOX UNREACHABLE 3 polls at $(date -u +%H:%MZ)"; fails=0
    fi
    sleep 180; continue
  fi
  fails=0
  line1="$(printf '%s\n' "$out" | sed -n 1p)"
  hb="$(printf '%s\n' "$out" | sed -n 2p)"
  step="$(printf '%s' "$line1" | grep -oE 'step=[0-9]+' | cut -d= -f2)"; step="${step:-0}"
  sess="$(printf '%s' "$line1" | grep -oE 'sessions=[0-9]+' | cut -d= -f2)"; sess="${sess:-0}"
  val="$(printf '%s' "$line1" | grep -oE 'val=[0-9]+' | cut -d= -f2)"; val="${val:-0}"
  err="$(printf '%s' "$line1" | grep -oE 'err=[0-9]+' | cut -d= -f2)"; err="${err:-0}"

  # (c) nothing running at all
  if [ "$sess" -eq 0 ]; then
    echo "$CELL: NO TMUX SESSIONS at step $step. GPU IS IDLE. ($line1)"; exit 1
  fi
  # (a) succeeded by another cell
  case "$hb" in
    *"$CELL"*) ;;
    *) echo "$CELL: TERMINAL at step $step, heartbeat moved to [$hb]. Score it, then read against the matrix."; exit 0 ;;
  esac
  # (b) reached the end and validated
  if [ "$step" -ge "$TOTAL" ] && [ "$val" -gt 0 ]; then
    echo "$CELL: COMPLETE at step $step/$TOTAL with val logged ($line1). Score against its PREREG."; exit 0
  fi

  # report errors once per new count, not every poll
  if [ "$err" -gt "$reported_err" ]; then
    echo "$CELL: $err error marker(s) at step $step -- classify (shutdown-path is benign, mid-run is not)"
    reported_err="$err"
  fi

  if [ "$step" -eq "$prev" ] && [ "$step" -gt 0 ]; then
    stale=$((stale + 1))
    [ "$stale" -eq 5 ] && { echo "$CELL STALL: step $step/$TOTAL unchanged 15 min ($line1)"; stale=0; }
  else
    stale=0
  fi
  prev="$step"

  m=$((step / 50))
  if [ "$m" -gt "$milestone" ]; then
    milestone="$m"; echo "$CELL PROGRESS: step $step/$TOTAL ($line1)"
  fi
  sleep 180
done
