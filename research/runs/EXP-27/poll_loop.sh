#!/bin/bash
# EXP-27 monitoring poll loop
# Runs 40min max, polls every 30s, appends to monitor-detail.log

LOG=/Users/shamane/Documents/verl/research/runs/EXP-27/monitor-detail.log
SSH_CMD="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40569 root@46.243.55.155"
EXP_DIR=/workspace/runs/EXP-27
CELL=exp27_B_ef_damped
SESSION=exp-27-46_243_55_155

POLL=1
START_TIME=$(date +%s)
MAX_WALL=2400  # 40 min
STALL_COUNT=0
WANDB_POLL_MOD=3  # poll WandB every 3rd iteration
PREV_LOG_MTIME=0
DONE_SEEN=0

source ~/.config/verl-research/secrets.env 2>/dev/null

get_wandb_scalars() {
  if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_KEY_MISSING"
    return
  fi
  # Query WandB for run state and latest scalars
  RESP=$(curl -s -m 15 -H "Authorization: Bearer $WANDB_API_KEY" \
    "https://api.wandb.ai/graphql" \
    -H "Content-Type: application/json" \
    -d '{"query":"{ project(entityName: \"shamanework-pl\", name: \"verl_compression_research\") { runs(filters: {\"display_name\": {\"$eq\": \"exp27_B_ef_damped\"}}, first: 1) { edges { node { name displayName state historyLineCount summaryMetrics } } } } }"}' 2>&1)
  echo "WANDB_RESP: $RESP"
}

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))

  if [ $ELAPSED -ge $MAX_WALL ]; then
    echo "=== TIMEOUT === $(date -u +"%Y-%m-%dT%H:%M:%SZ") elapsed=${ELAPSED}s ===" >> $LOG
    echo "EXIT_STATE: TIMEOUT" >> $LOG
    break
  fi

  POLL_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "" >> $LOG
  echo "=== POLL $POLL === $POLL_TIME === elapsed=${ELAPSED}s ===" >> $LOG

  # Main poll in one SSH call
  RESULT=$($SSH_CMD "
    echo TMUX_CHECK && tmux has-session -t $SESSION 2>&1 && echo TMUX_ALIVE || echo TMUX_DEAD
    echo FLAGS_CHECK && ls $EXP_DIR/done*.flag 2>/dev/null && echo FLAGS_FOUND || echo NO_FLAGS
    echo CELL_FLAG_CHECK && ls $EXP_DIR/${CELL}.done.flag 2>/dev/null && echo CELL_FLAG_EXISTS || echo CELL_FLAG_ABSENT
    echo AGGREGATE_FLAG_CHECK && ls $EXP_DIR/done.flag 2>/dev/null && echo AGG_FLAG_EXISTS || echo AGG_FLAG_ABSENT
    echo LOG_STAT && stat -c '%s %Y' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 'LOG_NOT_FOUND 0'
    echo LOG_TAIL && tail -25 $EXP_DIR/train_${CELL}.log 2>/dev/null
    echo ERRORS_CHECK
    echo TRACEBACK_COUNT=$(grep -aEc 'Traceback \(most recent call last\)' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 0)
    echo RUNTIME_ERR_COUNT=$(grep -aEc 'RuntimeError:' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 0)
    echo CUDA_OOM_COUNT=$(grep -aEc 'CUDA out of memory' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 0)
    echo NAN_COUNT=$(grep -aEc 'NaN detected' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 0)
    echo FATAL_COUNT=$(grep -aEc 'FATAL' $EXP_DIR/train_${CELL}.log 2>/dev/null || echo 0)
    echo LAST_STEP=$(grep -a 'global_step' $EXP_DIR/train_${CELL}.log 2>/dev/null | tail -1)
    echo LAST_COMM_EFF=$(grep -a '\[comm_eff\]\[EXP-27\]' $EXP_DIR/train_${CELL}.log 2>/dev/null | tail -1)
    echo LAST_VAL=$(grep -a 'val/test_score\|val/reward_score\|test/score' $EXP_DIR/train_${CELL}.log 2>/dev/null | tail -3)
    echo RESP_LEN=$(grep -a 'response_length/mean\|resp_len' $EXP_DIR/train_${CELL}.log 2>/dev/null | grep -v 'max_response_length\|response_length.*16384\|response_length.*2048' | tail -3)
    echo ENTROPY=$(grep -a 'entropy' $EXP_DIR/train_${CELL}.log 2>/dev/null | grep -v 'entropy_checkpointing\|entropy_from_logits\|FutureWarning' | tail -3)
    echo CORRECTION=$(grep -a 'correction_mode=ef_powersgd rel_change=\|spectral_corrections\|anchor_q_updates\|powersgd_basis_updates\|comm/bytes_ratio\|ef_clip\|ef_decay' $EXP_DIR/train_${CELL}.log 2>/dev/null | tail -5)
    echo GPU_UTIL && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  " 2>&1)

  echo "$RESULT" >> $LOG

  # Parse key values for exit condition logic
  TMUX_STATUS=$(echo "$RESULT" | grep -E "^TMUX_ALIVE|^TMUX_DEAD" | head -1)
  AGG_FLAG=$(echo "$RESULT" | grep -E "AGG_FLAG_EXISTS|AGG_FLAG_ABSENT" | head -1)
  CELL_FLAG=$(echo "$RESULT" | grep -E "CELL_FLAG_EXISTS|CELL_FLAG_ABSENT" | head -1)
  LOG_MTIME=$(echo "$RESULT" | grep -E "^[0-9]+ [0-9]+" | head -1 | awk '{print $2}')

  # GPU utilization - check for stall
  GPU_UTILS=$(echo "$RESULT" | grep -E "^[0-9]+, [0-9]+ %" | awk -F', ' '{print $2}' | tr -d ' %')
  ALL_LOW=true
  for UTIL in $GPU_UTILS; do
    if [ -n "$UTIL" ] && [ "$UTIL" -gt 5 ] 2>/dev/null; then
      ALL_LOW=false
      break
    fi
  done

  if $ALL_LOW && [ -n "$GPU_UTILS" ]; then
    STALL_COUNT=$((STALL_COUNT + 1))
    echo "STALL_COUNT: $STALL_COUNT/4" >> $LOG
  else
    STALL_COUNT=0
    echo "STALL_COUNT: reset (GPUs active)" >> $LOG
  fi

  # Check exit conditions
  if echo "$AGG_FLAG" | grep -q "AGG_FLAG_EXISTS"; then
    echo "EXIT_CONDITION: DONE_AGGREGATE" >> $LOG
    echo "EXIT_STATE: DONE_AGGREGATE" >> $LOG
    DONE_SEEN=1
    break
  fi

  if echo "$CELL_FLAG" | grep -q "CELL_FLAG_EXISTS" && echo "$TMUX_STATUS" | grep -q "TMUX_DEAD"; then
    echo "EXIT_CONDITION: DONE_1FLAG_TMUX_DEAD (single cell run)" >> $LOG
    echo "EXIT_STATE: DONE_3FLAGS" >> $LOG
    DONE_SEEN=1
    break
  fi

  if echo "$TMUX_STATUS" | grep -q "TMUX_DEAD" && ! echo "$CELL_FLAG" | grep -q "CELL_FLAG_EXISTS"; then
    echo "EXIT_CONDITION: TMUX_DEAD_PREMATURE" >> $LOG
    echo "EXIT_STATE: TMUX_DEAD_PREMATURE" >> $LOG
    DONE_SEEN=2
    break
  fi

  if [ $STALL_COUNT -ge 4 ]; then
    echo "EXIT_CONDITION: GPU_STALL (4 consecutive polls all GPUs <=5%)" >> $LOG
    echo "EXIT_STATE: GPU_STALL" >> $LOG
    DONE_SEEN=3
    break
  fi

  # WandB poll every 3rd iteration
  if [ $((POLL % WANDB_POLL_MOD)) -eq 0 ]; then
    echo "--- WANDB POLL ---" >> $LOG
    get_wandb_scalars >> $LOG 2>&1
  fi

  POLL=$((POLL + 1))

  # Wait 30s before next poll
  sleep 30
done

echo "POLL_LOOP_END: polls=$POLL elapsed=${ELAPSED}s done_state=$DONE_SEEN" >> $LOG
echo "LOOP_COMPLETE"
