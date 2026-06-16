#!/bin/bash
# EXP-33 window-6 polling loop
# Runs 30s-cadence polls for up to 40 min, writes to monitor-detail.log
# Exits when done/stall/error exit condition fires

SSH="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40266 root@46.243.55.155"
LOG_LOCAL=/Users/shamane/Documents/verl/research/runs/EXP-33/monitor-detail.log
METRICS_FILE=/Users/shamane/Documents/verl/research/runs/EXP-33/metrics/incoming.log
EXP_DIR=/Users/shamane/Documents/verl/research/runs/EXP-33
B0P75_DIR=$EXP_DIR/b0p75
B1P00_DIR=$EXP_DIR/b1p00

START_TS=$(date +%s)
POLL=1
GPU_STALL_COUNT=0
SSH_FAIL_COUNT=0
C3_DONE=0
C4_STARTED=0
C4_PASSTHROUGH_CHECKED=0
C4_DONE=0

poll() {
  local POLL_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local WALL=$(( $(date +%s) - START_TS ))

  # touch heartbeat
  touch "$METRICS_FILE" && echo "heartbeat $POLL_TS" >> "$METRICS_FILE"

  RESULT=$($SSH '
OUT=/tmp/mw6.txt; > "$OUT"
LOG_C3=/workspace/verl/runs/b0p75/train.log
LOG_C4=/workspace/verl/runs/b1p00/train.log
echo "TMUX=$(tmux has-session -t exp-33-46_243_55_155 2>/dev/null && echo ALIVE || echo DEAD)" >> "$OUT"
echo "FLAG_B0P75=$(ls /workspace/runs/EXP-33/done_b0p75.flag 2>/dev/null && echo YES || echo NO)" >> "$OUT"
echo "FLAG_B1P00=$(ls /workspace/runs/EXP-33/done_b1p00.flag 2>/dev/null && echo YES || echo NO)" >> "$OUT"
echo "FLAG_AGG=$(ls /workspace/runs/EXP-33/done.flag 2>/dev/null && echo YES || echo NO)" >> "$OUT"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null | tr "\n" "|" >> "$OUT"
echo "" >> "$OUT"
if [ -f "$LOG_C3" ]; then
  echo "C3_SIZE=$(stat -c%s $LOG_C3)" >> "$OUT"
  STEP=$(grep -a "TaskRunner.*step:[0-9]" "$LOG_C3" | grep "actor/grad_norm" | tail -1 | grep -oE "step:[0-9]+" | head -1)
  echo "C3_STEP=$STEP" >> "$OUT"
  CLIPFRAC=$(grep -a "TaskRunner.*actor/pg_clipfrac:[0-9]" "$LOG_C3" | tail -5 | grep -oE "actor/pg_clipfrac:[0-9.]+" | paste -sd,)
  echo "C3_CLIPFRAC=$CLIPFRAC" >> "$OUT"
  RESP=$(grep -a "TaskRunner.*response_length/mean:[0-9]" "$LOG_C3" | tail -5 | grep -oE "response_length/mean:[0-9.]+" | paste -sd,)
  echo "C3_RESP=$RESP" >> "$OUT"
  BYTES=$(grep -a "TaskRunner.*bytes_ratio:[0-9]" "$LOG_C3" | tail -1 | grep -oE "bytes_ratio:[0-9.]+" | head -1)
  echo "C3_BYTES=$BYTES" >> "$OUT"
  VAL=$(grep -a "val-core/openai/gsm8k/acc/mean@1" "$LOG_C3" | tail -3 | grep -oE "mean@1:[0-9.]+" | paste -sd,)
  echo "C3_VAL=$VAL" >> "$OUT"
  SCORE=$(grep -a "TaskRunner.*critic/score/mean:[0-9]" "$LOG_C3" | tail -1 | grep -oE "critic/score/mean:[0-9.]+" | head -1)
  echo "C3_SCORE=$SCORE" >> "$OUT"
  ERRS=$(grep -aE "Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|NaN detected" "$LOG_C3" | grep -v "DataLoader\|worker\|Killing\|signal Killed\|signal: killed" | wc -l)
  echo "C3_ERRORS=$ERRS" >> "$OUT"
fi
if [ -f "$LOG_C4" ]; then
  echo "C4_EXISTS=YES" >> "$OUT"
  echo "C4_SIZE=$(stat -c%s $LOG_C4)" >> "$OUT"
  STEP4=$(grep -a "TaskRunner.*step:[0-9]" "$LOG_C4" | grep "actor/grad_norm" | tail -1 | grep -oE "step:[0-9]+" | head -1)
  echo "C4_STEP=$STEP4" >> "$OUT"
  VAL4=$(grep -a "val-core/openai/gsm8k/acc/mean@1" "$LOG_C4" | tail -3 | grep -oE "mean@1:[0-9.]+" | paste -sd,)
  echo "C4_VAL=$VAL4" >> "$OUT"
  ERRS4=$(grep -aE "Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|NaN detected" "$LOG_C4" | grep -v "DataLoader\|worker\|Killing\|signal Killed\|signal: killed" | wc -l)
  echo "C4_ERRORS=$ERRS4" >> "$OUT"
  # passthrough check - search for beta_anc=1.0 in set -x trace
  BETA_CHECK=$(grep -a "beta_anc" "$LOG_C4" | head -5 | grep -oE "beta_anc=[0-9.]+" | head -1)
  echo "C4_BETA_CHECK=$BETA_CHECK" >> "$OUT"
  WANDB_CHECK=$(grep -a "wandb.*b1p00" "$LOG_C4" | head -3 | tr "\n" "|")
  echo "C4_WANDB=$WANDB_CHECK" >> "$OUT"
  VALBEFORE=$(grep -a "val_before_train" "$LOG_C4" | head -3 | grep -oE "val_before_train=[^ ]+" | head -1)
  echo "C4_VALBEFORE=$VALBEFORE" >> "$OUT"
else
  echo "C4_EXISTS=NO" >> "$OUT"
fi
cat "$OUT"
' 2>&1)

  local EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ] || echo "$RESULT" | grep -q "Permission denied\|Connection refused\|ssh: connect"; then
    SSH_FAIL_COUNT=$((SSH_FAIL_COUNT + 1))
    echo "=== POLL $POLL === $POLL_TS wall=${WALL}s SSH_FAIL #$SSH_FAIL_COUNT" >> "$LOG_LOCAL"
    if [ $SSH_FAIL_COUNT -ge 4 ]; then
      echo "ENV_FAILURE: SSH unreachable for >2 min" >> "$LOG_LOCAL"
      echo "EXIT_STATE=ENV_FAILURE" > $EXP_DIR/w6_exit.txt
      exit 1
    fi
    return
  fi
  SSH_FAIL_COUNT=0

  local TMUX=$(echo "$RESULT" | grep "^TMUX=" | cut -d= -f2)
  local FLAG_B0P75=$(echo "$RESULT" | grep "^FLAG_B0P75=" | cut -d= -f2)
  local FLAG_B1P00=$(echo "$RESULT" | grep "^FLAG_B1P00=" | cut -d= -f2)
  local FLAG_AGG=$(echo "$RESULT" | grep "^FLAG_AGG=" | cut -d= -f2)
  local GPU_LINE=$(echo "$RESULT" | grep -E "^[0-9]+,")
  local C3_STEP=$(echo "$RESULT" | grep "^C3_STEP=" | cut -d= -f2)
  local C3_CLIPFRAC=$(echo "$RESULT" | grep "^C3_CLIPFRAC=" | cut -d= -f2)
  local C3_RESP=$(echo "$RESULT" | grep "^C3_RESP=" | cut -d= -f2)
  local C3_BYTES=$(echo "$RESULT" | grep "^C3_BYTES=" | cut -d= -f2)
  local C3_VAL=$(echo "$RESULT" | grep "^C3_VAL=" | cut -d= -f2)
  local C3_SCORE=$(echo "$RESULT" | grep "^C3_SCORE=" | cut -d= -f2)
  local C3_ERRORS=$(echo "$RESULT" | grep "^C3_ERRORS=" | cut -d= -f2)
  local C4_EXISTS=$(echo "$RESULT" | grep "^C4_EXISTS=" | cut -d= -f2)
  local C4_STEP=$(echo "$RESULT" | grep "^C4_STEP=" | cut -d= -f2)
  local C4_VAL=$(echo "$RESULT" | grep "^C4_VAL=" | cut -d= -f2)
  local C4_ERRORS=$(echo "$RESULT" | grep "^C4_ERRORS=" | cut -d= -f2)
  local C4_BETA=$(echo "$RESULT" | grep "^C4_BETA_CHECK=" | cut -d= -f2)
  local C4_WANDB=$(echo "$RESULT" | grep "^C4_WANDB=" | cut -d= -f2)
  local C4_VALBEFORE=$(echo "$RESULT" | grep "^C4_VALBEFORE=" | cut -d= -f2)

  # GPU stall check
  local ALL_LOW=1
  while IFS= read -r line; do
    UTIL=$(echo "$line" | awk -F',' '{print $2}' | tr -d ' %')
    if [ -n "$UTIL" ] && [ "$UTIL" -gt 5 ] 2>/dev/null; then
      ALL_LOW=0
      break
    fi
  done <<< "$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null || echo "$GPU_LINE")"
  # Re-parse GPU util from RESULT
  ALL_LOW=1
  while IFS= read -r line; do
    if echo "$line" | grep -qE "^[0-9]+, [0-9]+ %"; then
      UTIL=$(echo "$line" | grep -oE "[0-9]+ %" | head -1 | tr -d ' %')
      if [ "$UTIL" -gt 5 ] 2>/dev/null; then ALL_LOW=0; break; fi
    fi
  done <<< "$(echo "$RESULT")"

  if [ "$ALL_LOW" -eq 1 ] && [ "$TMUX" = "ALIVE" ] && [ "$FLAG_AGG" = "NO" ]; then
    GPU_STALL_COUNT=$((GPU_STALL_COUNT + 1))
  else
    GPU_STALL_COUNT=0
  fi

  # Write poll entry
  {
    echo ""
    echo "=== POLL $POLL === $POLL_TS wall=${WALL}s"
    echo "TMUX=$TMUX FLAG_B0P75=$FLAG_B0P75 FLAG_B1P00=$FLAG_B1P00 FLAG_AGG=$FLAG_AGG GPU_STALL_CONSEC=$GPU_STALL_COUNT"
    echo "GPU: $GPU_LINE"
    echo "C3: step=$C3_STEP val=$C3_VAL score=$C3_SCORE bytes=$C3_BYTES errors=$C3_ERRORS"
    echo "C3_CLIPFRAC_RECENT: $C3_CLIPFRAC"
    echo "C3_RESP_RECENT: $C3_RESP"
    echo "C4: exists=$C4_EXISTS step=$C4_STEP val=$C4_VAL errors=$C4_ERRORS"
    if [ "$C4_EXISTS" = "YES" ]; then
      echo "C4_BETA=$C4_BETA C4_VALBEFORE=$C4_VALBEFORE"
      echo "C4_WANDB=$C4_WANDB"
    fi
  } >> "$LOG_LOCAL"

  # Handle done flags
  if [ "$FLAG_AGG" = "YES" ]; then
    echo "EXIT: DONE_AGGREGATE at poll $POLL" >> "$LOG_LOCAL"
    echo "EXIT_STATE=DONE_AGGREGATE" > $EXP_DIR/w6_exit.txt
    return 99
  fi

  if [ "$FLAG_B0P75" = "YES" ] && [ "$C3_DONE" -eq 0 ]; then
    C3_DONE=1
    echo "C3_DONE_FLAG_DETECTED at poll $POLL step=$C3_STEP val=$C3_VAL" >> "$LOG_LOCAL"
    # rsync C3 artifacts
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/verl/runs/b0p75/train.log" \
      "$B0P75_DIR/" 2>/dev/null
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/runs/EXP-33/done_b0p75.flag" \
      "$EXP_DIR/" 2>/dev/null
    echo "C3_RSYNC_DONE" >> "$LOG_LOCAL"
  fi

  if [ "$C4_EXISTS" = "YES" ] && [ "$C4_STARTED" -eq 0 ]; then
    C4_STARTED=1
    echo "C4_STARTED at poll $POLL" >> "$LOG_LOCAL"
  fi

  # C4 passthrough check (run once when C4 log appears and has enough content)
  if [ "$C4_EXISTS" = "YES" ] && [ "$C4_PASSTHROUGH_CHECKED" -eq 0 ] && [ -n "$C4_BETA" ]; then
    C4_PASSTHROUGH_CHECKED=1
    echo "C4_PASSTHROUGH_CHECK: beta_anc=$C4_BETA valbefore=$C4_VALBEFORE wandb=$C4_WANDB" >> "$LOG_LOCAL"
    if echo "$C4_BETA" | grep -qE "1\.0|1\.00"; then
      echo "C4_PASSTHROUGH: beta_anc CONFIRMED 1.0" >> "$LOG_LOCAL"
    else
      echo "C4_PASSTHROUGH_FAIL: beta_anc=$C4_BETA (expected 1.0) -- RELAUNCH NEEDED" >> "$LOG_LOCAL"
      echo "EXIT_STATE=C4_PASSTHROUGH_FAIL beta_anc=$C4_BETA" > $EXP_DIR/w6_exit.txt
      return 98
    fi
  fi

  if [ "$FLAG_B1P00" = "YES" ] && [ "$C4_DONE" -eq 0 ]; then
    C4_DONE=1
    echo "C4_DONE_FLAG_DETECTED at poll $POLL step=$C4_STEP val=$C4_VAL" >> "$LOG_LOCAL"
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/verl/runs/b1p00/train.log" \
      "$B1P00_DIR/" 2>/dev/null
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/runs/EXP-33/done_b1p00.flag" \
      "$EXP_DIR/" 2>/dev/null
    echo "C4_RSYNC_DONE" >> "$LOG_LOCAL"
  fi

  if [ "$FLAG_B0P75" = "YES" ] && [ "$FLAG_B1P00" = "YES" ] && [ "$TMUX" = "DEAD" ]; then
    echo "EXIT: DONE_3FLAGS at poll $POLL" >> "$LOG_LOCAL"
    echo "EXIT_STATE=DONE_ALL_RESUME_CELLS" > $EXP_DIR/w6_exit.txt
    return 99
  fi

  if [ "$TMUX" = "DEAD" ] && [ "$FLAG_B0P75" = "NO" ]; then
    echo "EXIT: TMUX_DEAD_PREMATURE no b0p75 flag poll $POLL" >> "$LOG_LOCAL"
    echo "EXIT_STATE=TMUX_DEAD_PREMATURE" > $EXP_DIR/w6_exit.txt
    return 97
  fi

  if [ "$GPU_STALL_COUNT" -ge 4 ] && [ "$TMUX" = "ALIVE" ]; then
    echo "EXIT: GPU_STALL >4 consecutive polls" >> "$LOG_LOCAL"
    echo "EXIT_STATE=GPU_STALL" > $EXP_DIR/w6_exit.txt
    return 96
  fi

  return 0
}

echo "=== MONITOR WINDOW 6 START === $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_LOCAL"

while true; do
  WALL=$(( $(date +%s) - START_TS ))
  if [ $WALL -ge 2400 ]; then  # 40 min
    echo "EXIT: TIMEOUT 40min" >> "$LOG_LOCAL"
    echo "EXIT_STATE=TIMEOUT" > $EXP_DIR/w6_exit.txt
    break
  fi

  poll
  EXIT=$?
  if [ $EXIT -eq 99 ] || [ $EXIT -eq 98 ] || [ $EXIT -eq 97 ] || [ $EXIT -eq 96 ]; then
    # Final rsync
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/verl/runs/b0p75/train.log" \
      "$B0P75_DIR/" 2>/dev/null
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/verl/runs/b1p00/train.log" \
      "$B1P00_DIR/" 2>/dev/null
    rsync -avz -e "ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266" \
      "root@46.243.55.155:/workspace/runs/EXP-33/" \
      "$EXP_DIR/" --include="done_*.flag" --include="done.flag" --exclude="*" 2>/dev/null
    echo "FINAL_RSYNC_DONE" >> "$LOG_LOCAL"
    break
  fi

  POLL=$((POLL + 1))
  sleep 30
done

echo "MONITOR_LOOP_END $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_LOCAL"
