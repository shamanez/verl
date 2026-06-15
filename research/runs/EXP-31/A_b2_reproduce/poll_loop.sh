#!/bin/bash
# EXP-31 Cell A polling loop
# Polls box every 60s (two 30s chunks), writes to monitor-detail.log, exits on done/stall/error

LOGFILE="/Users/shamane/Documents/verl/research/runs/EXP-31/A_b2_reproduce/monitor-detail.log"
ARTDIR="/Users/shamane/Documents/verl/research/runs/EXP-31/A_b2_reproduce"
SSH_OPTS="-i ~/.ssh/vast_ai_name -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
SSH="ssh $SSH_OPTS -p 40276 root@46.243.55.155"
REMOTE_LOG="/workspace/verl/runs/A_b2_reproduce/train.log"
REMOTE_DONE="/workspace/verl/runs/A_b2_reproduce/done.flag"

MAX_POLLS=80
STALL_THRESHOLD=4
stall_count=0
prev_step=-1
poll_num=0
START_TIME=$(date +%s)
val25_captured=""

log() {
    echo "$1" >> "$LOGFILE"
}

poll_box() {
    $SSH "
LOGF=$REMOTE_LOG
echo \"TMUX:\$(tmux has-session -t exp-31-46_243_55_155 2>&1 && echo ALIVE || echo DEAD)\"
echo \"DONE:\$(ls $REMOTE_DONE 2>/dev/null && echo EXISTS || echo NONE)\"
echo \"LOG_SIZE:\$(stat -c %s \$LOGF 2>/dev/null)\"
echo \"MAX_GLOBAL_STEP:\$(grep -oa 'training/global_step:[0-9]*' \$LOGF 2>/dev/null | sed 's/training\/global_step://' | sort -n | tail -1)\"
echo \"LAST_VAL_STEP:\$(grep -a 'val-core/openai/gsm8k/acc/mean' \$LOGF 2>/dev/null | grep -oP 'step:\K[0-9]+' | tail -1)\"
echo \"LAST_VAL_ACC:\$(grep -a 'val-core/openai/gsm8k/acc/mean' \$LOGF 2>/dev/null | grep -oP 'val-core/openai/gsm8k/acc/mean@1:\K[0-9.]+' | tail -1)\"
echo \"RESP_MEAN:\$(grep -oa 'response_length/mean:[0-9.]*' \$LOGF 2>/dev/null | tail -1)\"
echo \"RESP_MAX:\$(grep -oa 'response_length/max:[0-9.]*' \$LOGF 2>/dev/null | tail -1)\"
echo \"GRAD_NORM:\$(grep -oa 'actor/grad_norm:[0-9.]*' \$LOGF 2>/dev/null | tail -1)\"
echo \"BYTES_RATIO:\$(grep -oa 'bytes_ratio:[0-9.]*' \$LOGF 2>/dev/null | tail -1)\"
echo \"DELTA_RATIO_ALL:\$(grep -oa 'delta_ratio_median=[0-9.]*' \$LOGF 2>/dev/null | tr '\n' ',' )\"
echo \"ERR:\$(grep -ac 'Traceback\|RuntimeError:\|CUDA out of memory\|NaN detected\|FATAL' \$LOGF 2>/dev/null || echo 0)\"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
" 2>&1 | grep -v "^Welcome\|^Have fun"
}

while [ $poll_num -lt $MAX_POLLS ]; do
    NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ELAPSED=$(( $(date +%s) - START_TIME ))
    poll_num=$((poll_num + 1))

    log ""
    log "=== POLL $poll_num @ $NOW (elapsed ${ELAPSED}s) ==="

    # Poll the box
    RESULT=$(poll_box 2>&1)

    if [ $? -ne 0 ] || echo "$RESULT" | grep -q "^ssh:"; then
        log "SSH_ERROR: $RESULT"
        # If SSH fails for > 120s, it's an env failure
        if [ $ELAPSED -gt 120 ]; then
            log "EXIT: SSH unreachable >2min after start - ENV_FAILURE"
            echo "ENV_FAILURE" > "$ARTDIR/exit_state.txt"
            exit 2
        fi
        continue
    fi

    TMUX=$(echo "$RESULT" | grep "^TMUX:" | cut -d: -f2)
    DONE=$(echo "$RESULT" | grep "^DONE:" | cut -d: -f2)
    LOG_SIZE=$(echo "$RESULT" | grep "^LOG_SIZE:" | cut -d: -f2)
    MAX_STEP=$(echo "$RESULT" | grep "^MAX_GLOBAL_STEP:" | cut -d: -f2)
    VAL_STEP=$(echo "$RESULT" | grep "^LAST_VAL_STEP:" | cut -d: -f2)
    VAL_ACC=$(echo "$RESULT" | grep "^LAST_VAL_ACC:" | cut -d: -f2)
    RESP_MEAN=$(echo "$RESULT" | grep "^RESP_MEAN:" | cut -d: -f2-)
    RESP_MAX=$(echo "$RESULT" | grep "^RESP_MAX:" | cut -d: -f2-)
    GRAD_NORM=$(echo "$RESULT" | grep "^GRAD_NORM:" | cut -d: -f2-)
    BYTES_RATIO=$(echo "$RESULT" | grep "^BYTES_RATIO:" | cut -d: -f2-)
    DELTA_RATIO=$(echo "$RESULT" | grep "^DELTA_RATIO_ALL:" | cut -d: -f2-)
    ERR=$(echo "$RESULT" | grep "^ERR:" | cut -d: -f2)
    GPU_UTIL=$(echo "$RESULT" | grep -E "^[0-9]+, ")

    log "tmux=$TMUX done=$DONE log_size=$LOG_SIZE global_step=$MAX_STEP val_step=$VAL_STEP val_acc=$VAL_ACC"
    log "resp_mean=$RESP_MEAN resp_max=$RESP_MAX grad_norm=$GRAD_NORM bytes_ratio=$BYTES_RATIO err=$ERR"
    log "delta_ratio_all=$DELTA_RATIO"
    log "GPU: $GPU_UTIL"

    # Check for val@25
    if [ -n "$VAL_STEP" ] && [ "$VAL_STEP" = "25" ] && [ -z "$val25_captured" ]; then
        val25_captured="$VAL_ACC"
        log "*** VAL@25 CAPTURED: step=25 acc=$VAL_ACC ***"
        echo "VAL_AT_25=$VAL_ACC" >> "$ARTDIR/val25_result.txt"
    fi

    # Check error condition
    if [ -n "$ERR" ] && [ "$ERR" -gt "0" ] 2>/dev/null; then
        log "WARNING: Error pattern detected count=$ERR"
    fi

    # Check done flag
    if [ "$DONE" = "EXISTS" ]; then
        log "DONE FLAG EXISTS - cell completed"
        # rsync artifacts
        rsync -avz -e "ssh $SSH_OPTS -p 40276" \
            "root@46.243.55.155:/workspace/verl/runs/A_b2_reproduce/" \
            "$ARTDIR/" >> "$LOGFILE" 2>&1
        log "EXIT: DONE_FLAG"
        echo "DONE_AGGREGATE" > "$ARTDIR/exit_state.txt"
        exit 0
    fi

    # Check tmux dead + no done flag
    if [ "$TMUX" = "DEAD" ]; then
        log "TMUX DEAD without done flag - PREMATURE TERMINATION"
        rsync -avz -e "ssh $SSH_OPTS -p 40276" \
            "root@46.243.55.155:/workspace/verl/runs/A_b2_reproduce/" \
            "$ARTDIR/" >> "$LOGFILE" 2>&1
        log "EXIT: TMUX_DEAD_PREMATURE"
        echo "TMUX_DEAD_PREMATURE" > "$ARTDIR/exit_state.txt"
        exit 1
    fi

    # GPU stall detection
    ALL_IDLE=true
    while IFS= read -r line; do
        util=$(echo "$line" | awk -F', ' '{print $2}' | tr -d ' %')
        if [ -n "$util" ] && [ "$util" -gt 5 ] 2>/dev/null; then
            ALL_IDLE=false
        fi
    done <<< "$GPU_UTIL"

    if $ALL_IDLE && [ "$TMUX" = "ALIVE" ] && [ "$DONE" != "EXISTS" ]; then
        stall_count=$((stall_count + 1))
        log "GPU_STALL_COUNT=$stall_count (all GPUs <=5%)"
        if [ $stall_count -ge $STALL_THRESHOLD ]; then
            log "EXIT: GPU_STALL - 4 consecutive all-idle polls"
            echo "GPU_STALL" > "$ARTDIR/exit_state.txt"
            exit 3
        fi
    else
        stall_count=0
    fi

    # Timeout check (40 min = 2400s)
    if [ $ELAPSED -gt 2400 ]; then
        log "EXIT: TIMEOUT at ${ELAPSED}s"
        rsync -avz -e "ssh $SSH_OPTS -p 40276" \
            "root@46.243.55.155:/workspace/verl/runs/A_b2_reproduce/" \
            "$ARTDIR/" >> "$LOGFILE" 2>&1
        echo "TIMEOUT" > "$ARTDIR/exit_state.txt"
        exit 4
    fi

    sleep 60
done

log "EXIT: MAX_POLLS reached"
echo "TIMEOUT" > "$ARTDIR/exit_state.txt"
exit 4
