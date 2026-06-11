#!/usr/bin/env bash
# EXP-27 long-haul watcher (laptop-side, cheap — no LLM tokens during the wait).
# Polls the box every 5 min; EXITS (and thereby re-invokes the orchestrator) the
# moment the run is DONE, the tmux dies, or an OOM/NaN/length-ignition fires.
# Tolerates transient SSH failures (an empty poll just waits). 8 h hard ceiling.
WLOG=/Users/shamane/Documents/verl/research/runs/EXP-27/watcher.log
RLOG=/workspace/runs/EXP-27/train_exp27_B_ef_damped.log
SSH=(ssh -i "$HOME/.ssh/vast_ai" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -p 40569 root@46.243.55.155)
REASON=""
echo "[$(date -Iseconds)] watcher armed (5-min cadence, 8h ceiling)" >> "$WLOG"
for i in $(seq 1 96); do
  SNAP=$("${SSH[@]}" 'L=/workspace/runs/EXP-27/train_exp27_B_ef_damped.log
    tmux has-session -t exp-27-46_243_55_155 2>/dev/null && echo TMUX=ALIVE || echo TMUX=DEAD
    [ -f /workspace/runs/EXP-27/done.flag ] && echo DONE=YES || echo DONE=NO
    echo "STEP=$(grep -oE "global_step:[0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+")"
    echo "OOM=$(grep -caE "CUDA out of memory|OutOfMemoryError" "$L" 2>/dev/null)"
    echo "NAN=$(grep -caE "NaN detected" "$L" 2>/dev/null)"
    echo "LENMAX=$(grep -oE "response_length/max:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "LENMEAN=$(grep -oE "response_length/mean:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "VAL=$(grep -oE "val-core/openai/gsm8k/acc/mean@1:[0-9.]+" "$L" 2>/dev/null | tail -1)"
  ' 2>/dev/null)
  TS=$(date -Iseconds)
  if [ -z "$SNAP" ]; then echo "[$TS] poll $i: SSH empty/failed (transient) — keep waiting" >> "$WLOG"; sleep 300; continue; fi
  echo "[$TS] poll $i: $(echo "$SNAP" | tr '\n' ' ')" >> "$WLOG"
  echo "$SNAP" | grep -q "DONE=YES"  && { REASON="DONE_FLAG"; break; }
  echo "$SNAP" | grep -q "TMUX=DEAD" && { REASON="TMUX_DEAD"; break; }
  echo "$SNAP" | grep -qE "OOM=[1-9]" && { REASON="OOM"; break; }
  echo "$SNAP" | grep -qE "NAN=[1-9]" && { REASON="NAN"; break; }
  LM=$(echo "$SNAP" | sed -n 's/^LENMAX=//p'); LME=$(echo "$SNAP" | sed -n 's/^LENMEAN=//p')
  awk "BEGIN{exit !(${LM:-0}>=16000)}"  && { REASON="IGNITION_MAX_PINNED"; break; }
  awk "BEGIN{exit !(${LME:-0}>509)}"    && { REASON="IGNITION_MEAN_2X"; break; }
  sleep 300
done
echo "[$(date -Iseconds)] WATCHER EXIT reason=${REASON:-MAX_ITERS_8H}" | tee -a "$WLOG"
