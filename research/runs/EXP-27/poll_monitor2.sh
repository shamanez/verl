#!/bin/bash
# EXP-27 Monitor Session 2 - polling loop
# Polls every 90s (3 polls per WandB check cadence), up to 40 min
# Writes to monitor-detail.log

LOG=/Users/shamane/Documents/verl/research/runs/EXP-27/monitor-detail.log
RESULTS=/Users/shamane/Documents/verl/research/runs/EXP-27/monitor2_results.log
SSH_KEY=~/.ssh/vast_ai_name
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
SSH_TARGET="root@46.243.55.155"
SSH_PORT=40569
REMOTE_LOG="/workspace/runs/EXP-27/train_exp27_B_ef_damped.log"
TMUX_SESSION="exp-27-46_243_55_155"
MAX_POLLS=27  # ~40 min at 90s cadence
STALL_COUNT=0
POLL_NUM=49  # continuing from prior session's last poll

# Load secrets for WandB
source ~/.config/verl-research/secrets.env 2>/dev/null

STEP_10_BASELINE=509  # from dispatch, ~step-10 response_length mean
IGNITION_THRESHOLD=$(echo "$STEP_10_BASELINE * 2" | bc)

echo "=== MONITOR2 STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) IGNITION_THRESHOLD=${IGNITION_THRESHOLD} ===" | tee -a "$LOG" "$RESULTS"

for i in $(seq 1 $MAX_POLLS); do
    POLL_NUM=$((POLL_NUM + 1))
    POLL_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    echo "--- STARTING POLL $POLL_NUM at $POLL_TIME ---" >> "$RESULTS"

    # Run single SSH call
    SSH_RESULT=$(ssh $SSH_OPTS -p $SSH_PORT $SSH_TARGET '
RLOG="/workspace/runs/EXP-27/train_exp27_B_ef_damped.log"
echo "TMUX:$(tmux has-session -t exp-27-46_243_55_155 2>/dev/null && echo ALIVE || echo DEAD)"
echo "AGG:$(ls /workspace/runs/EXP-27/done.flag 2>/dev/null && echo DONE || echo NO)"
echo "CELL:$(ls /workspace/runs/EXP-27/exp27_B_ef_damped.done.flag 2>/dev/null && echo DONE || echo NO)"
stat -c "SIZE:%s MTIME:%Y" "$RLOG" 2>/dev/null || echo "SIZE:0 MTIME:0"
echo "STEP:$(grep -aoE "training/global_step:[0-9]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "RESP_MEAN:$(grep -aoE "response_length/mean:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "RESP_MAX:$(grep -aoE "response_length/max:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "ENTROPY:$(grep -aoE "actor/entropy:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "SCORE:$(grep -aoE "critic/score/mean:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "GRAD_NORM:$(grep -aoE "actor/grad_norm:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "MEM_GB:$(grep -aoE "actor/perf/max_memory_allocated_gb:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "BYTES_RATIO:$(grep -aoE "actor/comm/bytes_ratio:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "REL_CHANGE:$(grep -aoE "spectral/rel_change_mean:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "CLIP_RATIO:$(grep -aoE "response_length/clip_ratio:[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | cut -d: -f2)"
echo "ERRORS:$(grep -acE "Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|NaN detected|FATAL" "$RLOG" 2>/dev/null)"
# Val scores - scan for val-core lines
echo "VAL_LINES:$(grep -aE "val.*(score|test_score)" "$RLOG" 2>/dev/null | tail -3 | tr "\n" "|")"
# GPU util
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed "s/^/GPU:/"
' 2>&1)

    SSH_EXIT=$?

    if [ $SSH_EXIT -ne 0 ]; then
        echo "=== POLL $POLL_NUM $POLL_TIME SSH_FAIL exit=$SSH_EXIT ===" | tee -a "$LOG" "$RESULTS"
        echo "SSH_OUTPUT: $SSH_RESULT" >> "$RESULTS"
        sleep 90
        continue
    fi

    # Parse key values
    TMUX=$(echo "$SSH_RESULT" | grep "^TMUX:" | cut -d: -f2)
    AGG=$(echo "$SSH_RESULT" | grep "^AGG:" | cut -d: -f2)
    CELL=$(echo "$SSH_RESULT" | grep "^CELL:" | cut -d: -f2)
    STEP=$(echo "$SSH_RESULT" | grep "^STEP:" | cut -d: -f2)
    RESP_MEAN=$(echo "$SSH_RESULT" | grep "^RESP_MEAN:" | cut -d: -f2)
    RESP_MAX=$(echo "$SSH_RESULT" | grep "^RESP_MAX:" | cut -d: -f2)
    ENTROPY=$(echo "$SSH_RESULT" | grep "^ENTROPY:" | cut -d: -f2)
    SCORE=$(echo "$SSH_RESULT" | grep "^SCORE:" | cut -d: -f2)
    GRAD_NORM=$(echo "$SSH_RESULT" | grep "^GRAD_NORM:" | cut -d: -f2)
    MEM_GB=$(echo "$SSH_RESULT" | grep "^MEM_GB:" | cut -d: -f2)
    BYTES_RATIO=$(echo "$SSH_RESULT" | grep "^BYTES_RATIO:" | cut -d: -f2)
    REL_CHANGE=$(echo "$SSH_RESULT" | grep "^REL_CHANGE:" | cut -d: -f2)
    CLIP_RATIO=$(echo "$SSH_RESULT" | grep "^CLIP_RATIO:" | cut -d: -f2)
    ERRORS=$(echo "$SSH_RESULT" | grep "^ERRORS:" | cut -d: -f2)
    GPU_LINES=$(echo "$SSH_RESULT" | grep "^GPU:")

    # Check ignition: resp_mean > 2x baseline
    IGNITION_ALARM=""
    if [ -n "$RESP_MEAN" ] && [ "$RESP_MEAN" != "" ]; then
        RESP_MEAN_INT=$(printf "%.0f" "$RESP_MEAN" 2>/dev/null || echo "0")
        if [ "$RESP_MEAN_INT" -gt "$IGNITION_THRESHOLD" ] 2>/dev/null; then
            IGNITION_ALARM="ALARM_LENGTH_EXPLOSION: resp_mean=${RESP_MEAN} > threshold=${IGNITION_THRESHOLD}"
        fi
    fi
    # Check max pinned at 16384
    if [ "$RESP_MAX" = "16384.0" ]; then
        IGNITION_ALARM="${IGNITION_ALARM} ALARM_MAX_PINNED_16384"
    fi

    # Check GPU stall
    GPU_UTILS=$(echo "$GPU_LINES" | grep -oE "[0-9]+ %" | grep -oE "[0-9]+")
    ALL_STALLED=true
    for u in $GPU_UTILS; do
        if [ "$u" -gt 5 ] 2>/dev/null; then
            ALL_STALLED=false
            break
        fi
    done
    if $ALL_STALLED && [ "$TMUX" = "ALIVE" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
    else
        STALL_COUNT=0
    fi

    # Write to logs
    {
    echo ""
    echo "=== POLL $POLL_NUM $POLL_TIME ==="
    echo "TMUX: $TMUX"
    echo "DONE_FLAGS: AGG=$AGG CELL=$CELL"
    echo "STEP: $STEP / 100"
    echo "RESP_LEN: mean=$RESP_MEAN max=$RESP_MAX clip_ratio=$CLIP_RATIO"
    echo "ENTROPY: $ENTROPY"
    echo "SCORE: $SCORE"
    echo "GRAD_NORM: $GRAD_NORM"
    echo "MEM_GB: $MEM_GB"
    echo "BYTES_RATIO: $BYTES_RATIO"
    echo "REL_CHANGE: $REL_CHANGE"
    echo "ERRORS: $ERRORS"
    echo "GPU_UTIL: $GPU_LINES"
    echo "STALL_COUNT: $STALL_COUNT"
    [ -n "$IGNITION_ALARM" ] && echo "*** $IGNITION_ALARM ***"
    [ "$ERRORS" -gt 0 ] 2>/dev/null && echo "*** ERROR_DETECTED errors=$ERRORS ***"
    } | tee -a "$LOG" "$RESULTS"

    # Exit conditions
    if [ "$AGG" = "DONE" ]; then
        echo "=== EXIT: DONE_AGGREGATE step=$STEP ===" | tee -a "$LOG" "$RESULTS"
        # rsync artifacts
        rsync -avz -e "ssh $SSH_OPTS -p $SSH_PORT" \
            "root@46.243.55.155:/workspace/runs/EXP-27/" \
            "/Users/shamane/Documents/verl/research/runs/EXP-27/" \
            --include="*.log" --include="*.flag" --include="metrics/" --include="metrics/**" \
            --exclude="*" 2>&1 | tee -a "$RESULTS"
        echo "RSYNC_DONE" >> "$RESULTS"
        break
    fi

    if [ "$TMUX" = "DEAD" ]; then
        echo "=== EXIT: TMUX_DEAD step=$STEP ===" | tee -a "$LOG" "$RESULTS"
        break
    fi

    if [ -n "$IGNITION_ALARM" ]; then
        echo "=== IGNITION_ALARM FIRED - EARLY REPORT step=$STEP resp_mean=$RESP_MEAN ===" | tee -a "$LOG" "$RESULTS"
        # Don't break - keep watching per instructions
    fi

    if [ "$STALL_COUNT" -ge 4 ]; then
        echo "=== GPU_STALL DETECTED stall_count=$STALL_COUNT ===" | tee -a "$LOG" "$RESULTS"
        break
    fi

    sleep 90
done

echo "=== MONITOR2 LOOP COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS"
