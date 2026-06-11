#!/bin/bash
# EXP-27 monitor poll runner
# Runs polls 3-80 (up to 40min), writes results to poll_results.log
# Each poll outputs a sentinel line so Monitor can track progress

RESULTS=/Users/shamane/Documents/verl/research/runs/EXP-27/poll_results.log
LOG=/Users/shamane/Documents/verl/research/runs/EXP-27/monitor-detail.log
CELL=exp27_B_ef_damped
SESSION=exp-27-46_243_55_155

source ~/.config/verl-research/secrets.env 2>/dev/null

do_poll() {
  local PNUM=$1
  local PTIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  RESULT=$(ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40569 root@46.243.55.155 "
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    tmux has-session -t $SESSION 2>&1 && echo TMUX_ALIVE || echo TMUX_DEAD
    ls /workspace/runs/EXP-27/done*.flag 2>/dev/null || echo NO_FLAGS
    ls /workspace/runs/EXP-27/${CELL}.done.flag 2>/dev/null && echo CELL_FLAG_EXISTS || echo CELL_FLAG_ABSENT
    ls /workspace/runs/EXP-27/done.flag 2>/dev/null && echo AGG_FLAG_EXISTS || echo AGG_FLAG_ABSENT
    stat -c 'LOG_STAT %s %Y' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 'LOG_STAT NOT_FOUND 0'
    echo '=== TAIL ==='
    tail -25 /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null
    echo '=== ERRORS ==='
    echo TB=\$(grep -aEc 'Traceback \(most recent call last\)' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 0)
    echo RT=\$(grep -aEc 'RuntimeError:' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 0)
    echo OOM=\$(grep -aEc 'CUDA out of memory' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 0)
    echo NAN=\$(grep -aEc 'NaN detected' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 0)
    echo FATAL=\$(grep -aEc 'FATAL' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null || echo 0)
    echo '=== LAST_STEP ==='
    grep -a 'global_step' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | tail -2 || echo NO_STEP_YET
    echo '=== LAST_COMM_EFF ==='
    grep -a '\[comm_eff\]\[EXP-27\]' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | tail -2 || echo NO_COMM_EFF
    echo '=== VAL_SCORES ==='
    grep -a 'val\|test_score\|reward_score\|critic/score' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | grep -v 'FutureWarning\|deprecat\|please use\|validate_config\|val_type\|n_val\|val_before' | tail -5 || echo NO_VAL_YET
    echo '=== RESP_LEN ==='
    grep -a 'response_length/mean\|resp_len/mean\|response_length/max\|resp_len/max' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | tail -3 || echo NO_RESP_LEN_YET
    echo '=== ENTROPY ==='
    grep -a 'entropy' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | grep -v 'entropy_checkpointing\|entropy_from_logits\|FutureWarning\|config\|param' | tail -3 || echo NO_ENTROPY_YET
    echo '=== CORRECTION ==='
    grep -a 'correction_mode=ef_powersgd rel_change=\|anchor_q_updates\|powersgd_basis_updates\|comm/bytes_ratio\|ef_clip\|ef_decay' /workspace/runs/EXP-27/train_${CELL}.log 2>/dev/null | tail -3 || echo NO_CORRECTION_YET
    echo '=== GPU ==='
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  " 2>&1)

  {
    echo ""
    echo "=== POLL $PNUM === $PTIME ==="
    echo "$RESULT"
  } >> $LOG

  # Extract key state for results file
  TMUX_ST=$(echo "$RESULT" | grep -E "TMUX_ALIVE|TMUX_DEAD" | head -1 | tr -d '\r')
  AGG_FLAG=$(echo "$RESULT" | grep -E "AGG_FLAG_EXISTS|AGG_FLAG_ABSENT" | head -1 | tr -d '\r')
  CELL_FLAG=$(echo "$RESULT" | grep -E "CELL_FLAG_EXISTS|CELL_FLAG_ABSENT" | head -1 | tr -d '\r')
  ERRORS=$(echo "$RESULT" | grep -E "^TB=|^RT=|^OOM=|^NAN=|^FATAL=" | tr '\n' ' ')
  LAST_STEP=$(echo "$RESULT" | grep -E "global_step" | grep -v "LAST_STEP" | tail -1 | tr -d '\r')
  GPU_LINE=$(echo "$RESULT" | grep -E "^[0-9]+, [0-9]+ %" | head -4 | tr '\n' '|')
  LOG_STAT=$(echo "$RESULT" | grep "LOG_STAT" | head -1 | tr -d '\r')
  COMM_EFF=$(echo "$RESULT" | grep -E "correction_mode=ef_powersgd rel_change=|anchor_q_updates|comm/bytes_ratio" | tail -1 | tr -d '\r')
  VAL_LINE=$(echo "$RESULT" | grep -E "test_score|reward_score|critic/score" | grep -v "validate_config\|val_type\|n_val\|FutureWarning" | tail -1 | tr -d '\r')
  RESP_LEN=$(echo "$RESULT" | grep -E "response_length/mean|resp_len/mean" | tail -1 | tr -d '\r')

  # Determine GPU stall
  GPU_UTILS=$(echo "$RESULT" | grep -E "^[0-9]+, [0-9]+ %" | awk -F', ' '{gsub(/ %/,"",$2); print $2}')
  ALL_LOW=true
  for U in $GPU_UTILS; do
    if [ "$U" -gt 5 ] 2>/dev/null; then
      ALL_LOW=false
      break
    fi
  done

  # Write summary line to results file
  echo "POLL_RESULT poll=$PNUM time=$PTIME tmux=$TMUX_ST agg=$AGG_FLAG cell=$CELL_FLAG errors=[$ERRORS] step=[$LAST_STEP] gpu=[$GPU_LINE] val=[$VAL_LINE] resp=[$RESP_LEN] comm=[$COMM_EFF] log=[$LOG_STAT] all_low=$ALL_LOW" >> $RESULTS
  echo "POLL_DONE poll=$PNUM time=$PTIME tmux=$TMUX_ST agg=$AGG_FLAG cell=$CELL_FLAG all_low=$ALL_LOW step=[$LAST_STEP]"
}

do_wandb_poll() {
  local PNUM=$1
  WANDB_RESP=$(curl -s -m 15 -H "Authorization: Bearer $WANDB_API_KEY" \
    "https://api.wandb.ai/graphql" \
    -H "Content-Type: application/json" \
    -d '{"query":"{ project(entityName: \"shamanework-pl\", name: \"verl_compression_research\") { runs(filters: {\"display_name\": {\"$eq\": \"exp27_B_ef_damped\"}}, first: 1) { edges { node { name displayName state historyLineCount summaryMetrics } } } } }"}' 2>&1)
  echo "" >> $LOG
  echo "=== WANDB_POLL at poll $PNUM ===" >> $LOG
  echo "$WANDB_RESP" >> $LOG
  echo "WANDB_RESULT poll=$PNUM resp=$WANDB_RESP" >> $RESULTS
  echo "WANDB_DONE poll=$PNUM resp_len=${#WANDB_RESP}"
}

STALL_COUNT=0
PREV_ALL_LOW=false
START=$(date +%s)

for POLL_NUM in $(seq 3 80); do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  # Each poll iteration takes ~30s (SSH time) + we sleep 30s between polls
  # Total budget: 40min = 2400s from start of THIS script
  # Note: polls 1-2 already done, so start timer from now with remaining budget
  # We add 60s for the two already-done polls
  if [ $ELAPSED -gt 2340 ]; then  # 39 min from start of this script
    echo "TIMEOUT_REACHED elapsed=$ELAPSED" >> $RESULTS
    echo "POLL_DONE poll=$POLL_NUM time=$(date -u +"%Y-%m-%dT%H:%M:%SZ") EXIT=TIMEOUT"
    break
  fi

  POLL_OUTPUT=$(do_poll $POLL_NUM)
  echo "$POLL_OUTPUT"

  # Parse the output for exit conditions
  TMUX_VAL=$(echo "$POLL_OUTPUT" | grep "tmux=" | sed 's/.*tmux=//;s/ .*//')
  AGG_VAL=$(echo "$POLL_OUTPUT" | grep "agg=" | sed 's/.*agg=//;s/ .*//')
  CELL_VAL=$(echo "$POLL_OUTPUT" | grep "cell=" | sed 's/.*cell=//;s/ .*//')
  ALL_LOW_VAL=$(echo "$POLL_OUTPUT" | grep "all_low=" | sed 's/.*all_low=//;s/ .*//' | tr -d '\n')

  if echo "$AGG_VAL" | grep -q "AGG_FLAG_EXISTS"; then
    echo "EXIT_CONDITION=DONE_AGGREGATE poll=$POLL_NUM" >> $RESULTS
    echo "POLL_DONE poll=$POLL_NUM EXIT=DONE_AGGREGATE"
    break
  fi

  if echo "$CELL_VAL" | grep -q "CELL_FLAG_EXISTS" && echo "$TMUX_VAL" | grep -q "TMUX_DEAD"; then
    echo "EXIT_CONDITION=DONE_CELL_TMUX_DEAD poll=$POLL_NUM" >> $RESULTS
    echo "POLL_DONE poll=$POLL_NUM EXIT=DONE_CELL_TMUX_DEAD"
    break
  fi

  if echo "$TMUX_VAL" | grep -q "TMUX_DEAD" && ! echo "$CELL_VAL" | grep -q "CELL_FLAG_EXISTS"; then
    echo "EXIT_CONDITION=TMUX_DEAD_PREMATURE poll=$POLL_NUM" >> $RESULTS
    echo "POLL_DONE poll=$POLL_NUM EXIT=TMUX_DEAD_PREMATURE"
    break
  fi

  if [ "$ALL_LOW_VAL" = "true" ]; then
    STALL_COUNT=$((STALL_COUNT + 1))
    echo "STALL stall_count=$STALL_COUNT poll=$POLL_NUM" >> $RESULTS
  else
    STALL_COUNT=0
  fi

  if [ $STALL_COUNT -ge 4 ]; then
    echo "EXIT_CONDITION=GPU_STALL stall_count=$STALL_COUNT poll=$POLL_NUM" >> $RESULTS
    echo "POLL_DONE poll=$POLL_NUM EXIT=GPU_STALL"
    break
  fi

  # WandB every 3rd poll
  if [ $((POLL_NUM % 3)) -eq 0 ]; then
    do_wandb_poll $POLL_NUM
  fi

  sleep 30
done

echo "LOOP_END total_polls=$POLL_NUM"
