#!/usr/bin/env bash
# EXP-20 r=77 monitor polling script
# Run: bash monitor-poll.sh >> monitor-detail.log 2>&1

SSH="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -p 40009 root@84.8.106.109"
LOG=/workspace/runs/EXP-20/ce_powersgd_r77_clean5_50s_gsm8k.log
DONE_FLAG=/workspace/runs/EXP-20/powersgd_r77_arm.done

POLL=0
LOW_GPU_COUNT=0
PREV_STEP=7

while true; do
  POLL=$((POLL + 1))
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  RESULT=$($SSH "
    tmux has-session -t exp-20-powersgd-r77 2>&1 && echo TMUX_ALIVE || echo TMUX_DEAD
    ls $DONE_FLAG 2>/dev/null && echo DONE_FOUND || echo DONE_ABSENT
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    grep -aoE 'step:[0-9]+ - global_seqlen' $LOG 2>/dev/null | tail -1 | grep -oE 'step:[0-9]+'
    grep -c 'Traceback' $LOG 2>/dev/null || echo 0
  " 2>/dev/null)

  TMUX=$(echo "$RESULT" | grep -E "TMUX_ALIVE|TMUX_DEAD" | head -1)
  DONE=$(echo "$RESULT" | grep -E "DONE_FOUND|DONE_ABSENT" | head -1)
  STEP=$(echo "$RESULT" | grep -oE "step:[0-9]+" | grep -oE "[0-9]+" | tail -1)
  ERRORS=$(echo "$RESULT" | grep -v "TMUX\|DONE\|step\|%" | tail -1)

  # GPU util: extract percentages
  GPU_UTILS=$(echo "$RESULT" | grep "%" | grep -oE "[0-9]+ %" | tr '\n' '|')
  ALL_LOW=true
  while IFS= read -r line; do
    PCT=$(echo "$line" | grep -oE "[0-9]+ %" | grep -oE "[0-9]+")
    [ -z "$PCT" ] && continue
    [ "$PCT" -gt 5 ] && ALL_LOW=false
  done < <(echo "$RESULT" | grep "%")

  $ALL_LOW && LOW_GPU_COUNT=$((LOW_GPU_COUNT + 1)) || LOW_GPU_COUNT=0

  echo "POLL-$POLL $TS | $TMUX | step:${STEP:-?}/${PREV_STEP} | $DONE | GPU:$GPU_UTILS | LowGPU:$LOW_GPU_COUNT | Err:$ERRORS"

  [ -n "$STEP" ] && PREV_STEP=$STEP

  # EXIT CONDITIONS
  if echo "$DONE" | grep -q "DONE_FOUND"; then
    echo "*** EXIT: DONE FLAG FOUND at step $STEP ***"
    exit 0
  fi

  if echo "$TMUX" | grep -q "TMUX_DEAD"; then
    echo "*** EXIT: TMUX DEAD at step $STEP ***"
    exit 1
  fi

  if [ "$LOW_GPU_COUNT" -ge 4 ]; then
    echo "*** WARNING: GPU STALL - all GPUs <=5% for $LOW_GPU_COUNT consecutive polls ***"
    # Continue monitoring per instructions
  fi

  # Wait ~30s via busy loop (no sleep allowed)
  TARGET=$(($(date +%s) + 30))
  until [ $(date +%s) -ge $TARGET ]; do :; done
done
