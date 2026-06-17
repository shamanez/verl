#!/usr/bin/env bash
# EXP-34 lightweight status probe — run ON the box, one shot per poll.
# Greppable lines: TMUX: / AGG_DONE: / CELLDONE: / PROG: / VAL: / ERR: / GPU:
RUN=/workspace/runs/EXP-34
tmux has-session -t exp-34-104_202_252_41 2>/dev/null && echo TMUX:alive || echo TMUX:dead
[ -f "$RUN/done.flag" ] && echo AGG_DONE:yes
for c in signed_ema_b0p25 signed_ema_b0p50 signed_ema_b0p75; do
  [ -f "$RUN/done_$c.flag" ] && echo "CELLDONE:$c"
  L=/workspace/verl/runs/$c/train.log
  [ -f "$L" ] || continue
  GS=$(grep -oE 'global_step:[0-9]+' "$L" 2>/dev/null | tail -1)
  RW=$(grep -oE 'critic/score/mean:[0-9.]+' "$L" 2>/dev/null | tail -1)
  RL=$(grep -oE 'response_length/mean:[0-9.]+' "$L" 2>/dev/null | tail -1)
  TS=$(grep -oE 'perf/time_per_step:[0-9.]+' "$L" 2>/dev/null | tail -1)
  echo "PROG:$c: ${GS:-global_step:none} ${RW:-} ${RL:-} ${TS:-}"
  # val metrics (mean@1 appears only at val steps 25 & 50; greedy val)
  grep -hoE '[a-zA-Z0-9_/.-]*mean@1[:=][0-9.]+' "$L" 2>/dev/null | tail -3 | sed "s/^/VAL:$c: /"
  # corrupting-failure signatures (box's own EARLY_STOP watcher also writes a sentinel)
  grep -m1 -hE 'EARLY_STOP_SIGNAL|custom_all_reduce\.cuh:455|[Nn]a[Nn] detected|CUDA out of memory|^Traceback' "$L" 2>/dev/null | cut -c1-160 | sed "s/^/ERR:$c: /"
done
echo "GPU:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
