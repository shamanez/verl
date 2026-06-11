#!/usr/bin/env bash
# EXP-27 TIGHT ignition-confirm watcher. The run is showing an incipient
# length-explosion at step ~63 (max pinned 16384 x3, mean climbing, entropy →0.25).
# 90 s cadence; EXITS (re-invokes orchestrator) on a FIRM event:
#   - mean > 509 (plan's hard length alarm = 2x step-10 baseline)  -> FIRM_IGNITION
#   - a 3rd val line appears (= val@75 captured)                   -> VAL75
#   - score/mean < 0.55 (reward degradation)                       -> SCORE_DEGRADE
#   - done.flag / tmux dead / OOM / NaN
# 50-poll (~75 min) ceiling. Tolerates transient SSH drops.
WLOG=/Users/shamane/Documents/verl/research/runs/EXP-27/watcher.log
SSH=(ssh -i "$HOME/.ssh/vast_ai" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -p 40569 root@46.243.55.155)
REASON=""
echo "[$(date -Iseconds)] TIGHT ignition watcher armed (90s cadence)" >> "$WLOG"
for i in $(seq 1 50); do
  SNAP=$("${SSH[@]}" 'L=/workspace/runs/EXP-27/train_exp27_B_ef_damped.log
    tmux has-session -t exp-27-46_243_55_155 2>/dev/null && echo TMUX=ALIVE || echo TMUX=DEAD
    [ -f /workspace/runs/EXP-27/done.flag ] && echo DONE=YES || echo DONE=NO
    echo "STEP=$(grep -oE "global_step:[0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+")"
    echo "OOM=$(grep -caE "CUDA out of memory|OutOfMemoryError" "$L" 2>/dev/null)"
    echo "NAN=$(grep -caE "NaN detected" "$L" 2>/dev/null)"
    echo "LENMEAN=$(grep -oE "response_length/mean:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "LENMAX=$(grep -oE "response_length/max:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "ENT=$(grep -oE "actor/entropy:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "SCORE=$(grep -oE "critic/score/mean:[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")"
    echo "VALCOUNT=$(grep -cE "val-core/openai/gsm8k/acc/mean@1:" "$L" 2>/dev/null)"
    echo "VALLAST=$(grep -oE "val-core/openai/gsm8k/acc/mean@1:[0-9.]+" "$L" 2>/dev/null | tail -1)"
  ' 2>/dev/null)
  TS=$(date -Iseconds)
  if [ -z "$SNAP" ]; then echo "[$TS] poll $i: SSH empty (transient)" >> "$WLOG"; sleep 90; continue; fi
  echo "[$TS] poll $i: $(echo "$SNAP" | tr '\n' ' ')" >> "$WLOG"
  echo "$SNAP" | grep -q "DONE=YES"  && { REASON="DONE_FLAG"; break; }
  echo "$SNAP" | grep -q "TMUX=DEAD" && { REASON="TMUX_DEAD"; break; }
  echo "$SNAP" | grep -qE "OOM=[1-9]" && { REASON="OOM"; break; }
  echo "$SNAP" | grep -qE "NAN=[1-9]" && { REASON="NAN"; break; }
  LME=$(echo "$SNAP" | sed -n 's/^LENMEAN=//p'); SC=$(echo "$SNAP" | sed -n 's/^SCORE=//p'); VC=$(echo "$SNAP" | sed -n 's/^VALCOUNT=//p')
  awk "BEGIN{exit !(${LME:-0}>509)}" && { REASON="FIRM_IGNITION_MEAN"; break; }
  awk "BEGIN{exit !(${SC:-1}<0.55)}" && { REASON="SCORE_DEGRADE"; break; }
  [ "${VC:-0}" -ge 3 ] 2>/dev/null && { REASON="VAL75"; break; }
  sleep 90
done
echo "[$(date -Iseconds)] TIGHT WATCHER EXIT reason=${REASON:-MAX_ITERS}" | tee -a "$WLOG"
