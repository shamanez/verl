#!/usr/bin/env bash
# Watcher for issue #93 cell a6. Replaces watch93.sh, which had two bugs worth
# not repeating:
#
#   1. Its stall detector compared the PREVIOUS cell's step against the SHARED
#      tmux session name, so once a6 took over run-93 it read "session live,
#      a5b not advancing" as a stall. a finished cell can never advance. A
#      watcher must key on the step of the cell it is actually watching.
#   2. Its exit condition grepped `tail -1` of chain-93.log for "LAUNCHED", but
#      the chain appends a bring-up line after that, so the condition could
#      never fire and the watcher would have run forever.
#
# This one watches a6 and only a6, and its terminal condition is a6's own tmux
# session disappearing, which cannot be spoofed by a later cell because nothing
# is chained after a6.
CELL=a6-prf-exactk-tis-bnorm-200
LOG=/workspace/runs/$CELL/train.log
SSHC=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes \
      -i "$HOME/.ssh/vast_ai" -p 8602 root@50.46.253.92)

fails=0; prev=-1; stale=0; milestone=0

while true; do
  out="$("${SSHC[@]}" "s=\$(grep -aoE 'global_step:[0-9]+' $LOG 2>/dev/null | tail -1 | cut -d: -f2)
t=\$(tmux has-session -t run-93 2>/dev/null && echo live || echo gone)
e=\$(grep -acE 'Traceback|CUDA out of memory|FATAL:' $LOG 2>/dev/null || echo 0)
g=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo \"step=\${s:-0} tmux=\$t err=\$e gpu=\${g:-NA}\"" 2>/dev/null)"

  if [ -z "$out" ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 3 ]; then
      echo "a6: BOX UNREACHABLE 3 polls in a row at $(date -u +%H:%MZ)"
      fails=0
    fi
    sleep 180; continue
  fi
  fails=0

  step="$(printf '%s' "$out" | grep -oE 'step=[0-9]+' | cut -d= -f2)"
  tmx="$(printf '%s' "$out" | grep -oE 'tmux=[a-z]+' | cut -d= -f2)"
  err="$(printf '%s' "$out" | grep -oE 'err=[0-9]+' | cut -d= -f2)"
  step="${step:-0}"; err="${err:-0}"

  if [ "$tmx" = "gone" ]; then
    echo "a6 TERMINAL at step $step/200 ($out). Score against PREREG_a6.md, read the 2x2 against a5b, then close out. GPU IS NOW IDLE."
    exit 0
  fi

  [ "$err" -gt 0 ] && echo "a6: $err error marker(s) at step $step -- classify (shutdown-path tracebacks are benign, mid-run ones are not)"

  if [ "$step" -eq "$prev" ] && [ "$step" -gt 0 ]; then
    stale=$((stale + 1))
    if [ "$stale" -eq 5 ]; then
      echo "a6 STALL: step $step/200 unchanged for 15 min, nominal is about 120 s/step ($out)"
      stale=0
    fi
  else
    stale=0
  fi
  prev="$step"

  m=$((step / 50))
  if [ "$m" -gt "$milestone" ]; then
    milestone="$m"
    echo "a6 PROGRESS: step $step/200 ($out)"
  fi

  sleep 180
done
