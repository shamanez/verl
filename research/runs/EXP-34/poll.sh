#!/usr/bin/env bash
# EXP-34 lightweight status probe — run ON the box, one shot per poll.
# Emits greppable lines: TMUX:, AGG_DONE:, CELLDONE:, STEP:, VAL:, ERR:, GPU:.
RUN=/workspace/runs/EXP-34
tmux has-session -t exp-34-104_202_252_41 2>/dev/null && echo TMUX:alive || echo TMUX:dead
[ -f "$RUN/done.flag" ] && echo AGG_DONE:yes
for c in signed_ema_b0p25 signed_ema_b0p50 signed_ema_b0p75; do
  [ -f "$RUN/done_$c.flag" ] && echo "CELLDONE:$c"
  L=/workspace/verl/runs/$c/train.log
  [ -f "$L" ] || continue
  echo "STEP:$c:$(grep -oE 'step:[0-9]+' "$L" 2>/dev/null | tail -1)"
  # val lines (appear at steps 25 & 50): print any line carrying mean@1, truncated
  grep -hE 'val-core.*mean@1|mean@1.*val-core' "$L" 2>/dev/null | tail -2 \
    | cut -c1-320 | sed "s/^/VAL:$c: /"
  # corrupting-failure signatures (the box's own EARLY_STOP watcher also writes a sentinel)
  grep -m1 -hE 'EARLY_STOP_SIGNAL|custom_all_reduce\.cuh:455|[Nn]a[Nn] detected|CUDA out of memory|^Traceback' "$L" 2>/dev/null \
    | cut -c1-200 | sed "s/^/ERR:$c: /"
done
echo "GPU:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
