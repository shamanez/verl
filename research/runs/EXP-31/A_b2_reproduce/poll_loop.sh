#!/bin/bash
# EXP-31 Cell A monitoring poll loop
# Polls box every 30s, writes to monitor-detail.log, exits on done/stall/error/timeout
# DO NOT run directly with sleep -- this script uses its own loop with sleep 30

LOGFILE="/Users/shamane/Documents/verl/research/runs/EXP-31/A_b2_reproduce/monitor-detail.log"
ARTDIR="/Users/shamane/Documents/verl/research/runs/EXP-31/A_b2_reproduce"
SSH_OPTS="-i ~/.ssh/vast_ai_name -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
SSH="ssh $SSH_OPTS -p 40276 root@46.243.55.155"
REMOTE_BASE="/workspace/verl/runs/A_b2_reproduce"
REMOTE_LOG="$REMOTE_BASE/train.log"
REMOTE_DONE="$REMOTE_BASE/done.flag"
TMUX_SESS="exp-31-46_243_55_155"

MAX_POLLS=80
STALL_THRESHOLD=4
stall_count=0
poll_num=0
START_TIME=$(date +%s)
WANDB_POLL_INTERVAL=3  # poll WandB every 3rd poll (~90s)

source ~/.config/verl-research/secrets.env 2>/dev/null || true

log() { echo "$1" >> "$LOGFILE"; }

while [ $poll_num -lt $MAX_POLLS ]; do
    POLL_START=$(date +%s)
    NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ELAPSED=$(( POLL_START - START_TIME ))
    poll_num=$((poll_num + 1))

    log ""
    log "=== POLL $poll_num @ $NOW (elapsed ${ELAPSED}s) ==="

    # One SSH call to capture all state
    RESULT=$($SSH '
echo "POLL_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "TMUX_CELL=$(tmux has-session -t exp-31-46_243_55_155 2>&1 && echo ALIVE || echo DEAD)"
echo "TMUX_CHAIN=$(tmux has-session -t exp-31-chain 2>&1 && echo ALIVE || echo DEAD)"
echo "DONE_FLAG=$(ls /workspace/verl/runs/A_b2_reproduce/done.flag 2>/dev/null && echo PRESENT || echo ABSENT)"
echo "LOG_SIZE_MTIME=$(stat -c "%s %Y" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null || echo "0 0")"
echo "MAX_GLOBAL_STEP=$(grep -oa "step:[0-9]\+" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | grep -v "steps\|max_step\|save_step\|test_freq\|total_step\|global_steps_in_epoch" | awk -F: "{print \$2}" | sort -n | tail -1)"
echo "LAST_VAL_LINE=$(grep -a "val-core/openai/gsm8k/acc/mean@1" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1 | grep -oE "step:[0-9]+.*val-core/openai/gsm8k/acc/mean@1:[0-9.]+")"
echo "RESP_MEAN=$(grep -oa "response_length/mean:[0-9.]*" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1)"
echo "RESP_MAX=$(grep -oa "response_length/max:[0-9.]*" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1)"
echo "GRAD_NORM=$(grep -oa "actor/grad_norm:[0-9.]*" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1)"
echo "BYTES_RATIO=$(grep -oa "bytes_ratio:[0-9.]*" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1)"
echo "ERROR_COUNT=$(grep -acE "Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|NaN detected|FATAL" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null || echo 0)"
echo "COMM_EFF_LAST=$(grep -a "\[comm_eff\]\[EXP-31\]" /workspace/verl/runs/A_b2_reproduce/train.log 2>/dev/null | tail -1)"
echo "CHAIN_LAST=$(tail -3 /workspace/chain.log 2>/dev/null | tr "\n" "|")"
echo "=SMI_START="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=SMI_END="
' 2>/dev/null || echo "SSH_FAILED")

    if echo "$RESULT" | grep -q "^SSH_FAILED"; then
        log "SSH_ERROR at poll $poll_num"
        if [ $ELAPSED -gt 120 ]; then
            log "EXIT: ENV_FAILURE (SSH unreachable >2min)"
            echo "ENV_FAILURE" > "$ARTDIR/exit_state.txt"
            exit 2
        fi
        sleep 30
        continue
    fi

    TMUX_CELL=$(echo "$RESULT" | grep "^TMUX_CELL=" | cut -d= -f2)
    TMUX_CHAIN=$(echo "$RESULT" | grep "^TMUX_CHAIN=" | cut -d= -f2)
    DONE_FLAG=$(echo "$RESULT" | grep "^DONE_FLAG=" | cut -d= -f2)
    LOG_INFO=$(echo "$RESULT" | grep "^LOG_SIZE_MTIME=" | cut -d= -f2-)
    MAX_STEP=$(echo "$RESULT" | grep "^MAX_GLOBAL_STEP=" | cut -d= -f2)
    LAST_VAL=$(echo "$RESULT" | grep "^LAST_VAL_LINE=" | cut -d= -f2-)
    RESP_MEAN=$(echo "$RESULT" | grep "^RESP_MEAN=" | cut -d= -f2-)
    RESP_MAX=$(echo "$RESULT" | grep "^RESP_MAX=" | cut -d= -f2-)
    GRAD_NORM=$(echo "$RESULT" | grep "^GRAD_NORM=" | cut -d= -f2-)
    BYTES_RATIO=$(echo "$RESULT" | grep "^BYTES_RATIO=" | cut -d= -f2-)
    ERROR_COUNT=$(echo "$RESULT" | grep "^ERROR_COUNT=" | cut -d= -f2)
    COMM_EFF=$(echo "$RESULT" | grep "^COMM_EFF_LAST=" | cut -d= -f2-)
    CHAIN_LAST=$(echo "$RESULT" | grep "^CHAIN_LAST=" | cut -d= -f2-)
    SMI_BLOCK=$(echo "$RESULT" | awk '/^=SMI_START=$/,/^=SMI_END=/' | grep -v "=SMI_START=\|=SMI_END=")

    log "tmux_cell=$TMUX_CELL tmux_chain=$TMUX_CHAIN done_flag=$DONE_FLAG"
    log "log_info=$LOG_INFO max_step=$MAX_STEP"
    log "last_val=$LAST_VAL"
    log "resp_mean=$RESP_MEAN resp_max=$RESP_MAX grad_norm=$GRAD_NORM bytes_ratio=$BYTES_RATIO"
    log "error_count=$ERROR_COUNT"
    log "comm_eff_last=$COMM_EFF"
    log "chain_last=$CHAIN_LAST"
    log "gpu_util:"
    echo "$SMI_BLOCK" | while IFS= read -r line; do [ -n "$line" ] && log "  $line"; done

    # GPU stall detection (all GPUs <=5%)
    ALL_IDLE=true
    while IFS= read -r line; do
        if [ -z "$line" ]; then continue; fi
        UTIL=$(echo "$line" | awk -F', ' '{print $2}' | tr -d ' %')
        if echo "$UTIL" | grep -qE '^[0-9]+$' && [ "$UTIL" -gt 5 ] 2>/dev/null; then
            ALL_IDLE=false
        fi
    done <<< "$SMI_BLOCK"

    if $ALL_IDLE && [ "$TMUX_CELL" = "ALIVE" ] && [ "$DONE_FLAG" != "PRESENT" ]; then
        stall_count=$((stall_count + 1))
        log "GPU_STALL_STREAK=$stall_count (all GPUs <=5%)"
        if [ $stall_count -ge $STALL_THRESHOLD ]; then
            log "EXIT: GPU_STALL after $stall_count consecutive idle polls"
            rsync -avz -e "ssh $SSH_OPTS -p 40276" "root@46.243.55.155:$REMOTE_BASE/" "$ARTDIR/" >> "$LOGFILE" 2>&1 || true
            echo "GPU_STALL" > "$ARTDIR/exit_state.txt"
            exit 3
        fi
    else
        stall_count=0
    fi

    # Exit on done flag
    if [ "$DONE_FLAG" = "PRESENT" ]; then
        log "DONE_FLAG PRESENT - Cell A completed"
        rsync -avz -e "ssh $SSH_OPTS -p 40276" "root@46.243.55.155:$REMOTE_BASE/" "$ARTDIR/" >> "$LOGFILE" 2>&1 || true
        log "EXIT: DONE_AGGREGATE"
        echo "DONE_AGGREGATE" > "$ARTDIR/exit_state.txt"
        exit 0
    fi

    # Exit on tmux death without done flag
    if [ "$TMUX_CELL" = "DEAD" ]; then
        log "TMUX_DEAD without done flag - PREMATURE TERMINATION"
        rsync -avz -e "ssh $SSH_OPTS -p 40276" "root@46.243.55.155:$REMOTE_BASE/" "$ARTDIR/" >> "$LOGFILE" 2>&1 || true
        log "EXIT: TMUX_DEAD_PREMATURE"
        echo "TMUX_DEAD_PREMATURE" > "$ARTDIR/exit_state.txt"
        exit 1
    fi

    # Error pattern: keep polling (per plan non-negotiable), just log
    if [ -n "$ERROR_COUNT" ] && [ "$ERROR_COUNT" -gt 0 ] 2>/dev/null; then
        log "WARNING: ERROR_PATTERN_DETECTED count=$ERROR_COUNT (keeping poll per plan §non-negotiables)"
    fi

    # Timeout (40 min = 2400s)
    if [ $ELAPSED -gt 2400 ]; then
        log "EXIT: TIMEOUT at elapsed=${ELAPSED}s"
        rsync -avz -e "ssh $SSH_OPTS -p 40276" "root@46.243.55.155:$REMOTE_BASE/" "$ARTDIR/" >> "$LOGFILE" 2>&1 || true
        echo "TIMEOUT" > "$ARTDIR/exit_state.txt"
        exit 4
    fi

    sleep 30
done

log "EXIT: MAX_POLLS=$MAX_POLLS reached"
echo "TIMEOUT" > "$ARTDIR/exit_state.txt"
exit 4
