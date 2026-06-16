#!/bin/bash
# EXP-33 Monitor Window 2 — polling loop
# Polls every 30s, max 40 min (80 polls), exits on done/failure conditions
# Writes to runs/EXP-33/monitor-detail.log

SSH="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40266 root@46.243.55.155"
LOG="/Users/shamane/Documents/verl/research/runs/EXP-33/monitor-detail.log"
RESULT="/Users/shamane/Documents/verl/research/runs/EXP-33/monitor-poll2-result.json"
RSYNC_DEST="/Users/shamane/Documents/verl/research/runs/EXP-33"
RSYNC_SSH="ssh -i ~/.ssh/vast_ai_name -o StrictHostKeyChecking=accept-new -p 40266"

MAX_POLLS=80
STALL_THRESHOLD=4
GPU_STALL_COUNT=0
PREV_LOG_SIZE=0
PREV_LOG_MTIME=0
DONE_B0P50=false
DONE_B0P75=false
DONE_B1P00=false
ACTIVE_CELL="b0p50"
POLL_COUNT=0
START_TS=$(date +%s)

# Track length trajectory for P1/P2/P3/E1
declare -a LEN_MEAN_HISTORY=()
declare -a CLIP_RATIO_HISTORY=()
CONSECUTIVE_CAP_PINS=0
EARLY_EXIT_REASON=""
LAST_VAL_B0P50=""
LAST_VAL_B0P75=""
LAST_VAL_B1P00=""

log() {
    echo "$*" >> "$LOG"
}

rsync_cell() {
    local cell=$1
    local remote_log="/workspace/verl/runs/${cell}/train.log"
    local remote_flag="/workspace/runs/EXP-33/done_${cell}.flag"
    local local_dir="${RSYNC_DEST}/${cell}"
    mkdir -p "$local_dir"
    rsync -az -e "$RSYNC_SSH" \
        "root@46.243.55.155:${remote_log}" \
        "$local_dir/" 2>/dev/null || true
    rsync -az -e "$RSYNC_SSH" \
        "root@46.243.55.155:${remote_flag}" \
        "${RSYNC_DEST}/" 2>/dev/null || true
}

# Update heartbeat for teardown hook
touch_heartbeat() {
    touch /Users/shamane/Documents/verl/research/runs/EXP-33/metrics/incoming.log 2>/dev/null || true
}

while [ $POLL_COUNT -lt $MAX_POLLS ]; do
    POLL_COUNT=$((POLL_COUNT + 1))
    NOW=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    ELAPSED=$(( $(date +%s) - START_TS ))

    # Check 40-min wall limit
    if [ $ELAPSED -ge 2400 ]; then
        log ""
        log "=== POLL $POLL_COUNT [$NOW] TIMEOUT (${ELAPSED}s elapsed) ==="
        echo '{"exit_state":"TIMEOUT","elapsed_s":'$ELAPSED',"active_cell":"'$ACTIVE_CELL'","done_b0p50":'$DONE_B0P50',"done_b0p75":'$DONE_B0P75',"done_b1p00":'$DONE_B1P00'}' > "$RESULT"
        break
    fi

    # SSH poll — all in one call
    RAW=$(${SSH} '
tmux has-session -t exp-33-46_243_55_155 2>/dev/null && echo "TMUX:ALIVE" || echo "TMUX:DEAD"
echo "FLAGS:$(ls /workspace/runs/EXP-33/done_*.flag 2>/dev/null | tr "\n" "|")"
ls /workspace/runs/EXP-33/done.flag 2>/dev/null && echo "AGG:yes" || echo "AGG:no"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed "s/^/GPU:/"
echo "B0P50_STAT:$(stat -c "%s %Y" /workspace/verl/runs/b0p50/train.log 2>/dev/null || echo "0 0")"
echo "B0P50_STEP:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "step:[0-9]+" | grep -v "max_step\|min_step\|total_steps\|num_steps" | tail -1)"
echo "B0P50_MEAN:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "response_length/mean:[0-9.]+" | tail -1)"
echo "B0P50_MAX:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "response_length/max:[0-9.]+" | tail -1)"
echo "B0P50_CLIP:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "response_length/clip_ratio:[0-9.]+" | tail -1)"
echo "B0P50_VAL:$(tail -c 500000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "val-core/openai/gsm8k/acc/mean@1:[0-9.]+" | tail -1)"
echo "B0P50_REW:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "critic/reward/mean:[0-9.-]+" | tail -1)"
echo "B0P50_BYTES:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "bytes_ratio:[0-9.e+-]+" | tail -1)"
echo "B0P50_RECON:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "powersgd_reconstruction_rel_error:[0-9.e+-]+" | tail -1)"
echo "B0P50_ANCH:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "anchor_backwards:[0-9.]+" | tail -1)"
echo "B0P50_COLD:$(tail -c 100000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -oE "merger_coldM_fallbacks:[0-9.]+" | tail -1)"
echo "B0P50_ERRS:$(tail -c 500000 /workspace/verl/runs/b0p50/train.log 2>/dev/null | grep -c "Traceback\|CUDA out of memory\|NaN detected\|RuntimeError:" 2>/dev/null || echo 0)"
echo "B0P75_STAT:$(stat -c "%s %Y" /workspace/verl/runs/b0p75/train.log 2>/dev/null || echo "0 0")"
echo "B0P75_VAL:$(tail -c 200000 /workspace/verl/runs/b0p75/train.log 2>/dev/null | grep -oE "val-core/openai/gsm8k/acc/mean@1:[0-9.]+" | tail -1)"
echo "B0P75_STEP:$(tail -c 100000 /workspace/verl/runs/b0p75/train.log 2>/dev/null | grep -oE "step:[0-9]+" | grep -v "max_step\|min_step\|total_steps\|num_steps" | tail -1)"
echo "B1P00_STAT:$(stat -c "%s %Y" /workspace/verl/runs/b1p00/train.log 2>/dev/null || echo "0 0")"
echo "B1P00_VAL:$(tail -c 200000 /workspace/verl/runs/b1p00/train.log 2>/dev/null | grep -oE "val-core/openai/gsm8k/acc/mean@1:[0-9.]+" | tail -1)"
echo "B1P00_STEP:$(tail -c 100000 /workspace/verl/runs/b1p00/train.log 2>/dev/null | grep -oE "step:[0-9]+" | grep -v "max_step\|min_step\|total_steps\|num_steps" | tail -1)"
' 2>/dev/null)

    SSH_EXIT=$?

    # Parse key fields
    TMUX_STATE=$(echo "$RAW" | grep "^TMUX:" | cut -d: -f2)
    AGG=$(echo "$RAW" | grep "^AGG:" | cut -d: -f2)
    FLAGS=$(echo "$RAW" | grep "^FLAGS:" | cut -d: -f2-)
    GPU_UTIL=$(echo "$RAW" | grep "^GPU:" | sed 's/GPU://')
    B0P50_STAT=$(echo "$RAW" | grep "^B0P50_STAT:" | cut -d: -f2-)
    B0P50_SIZE=$(echo "$B0P50_STAT" | awk '{print $1}')
    B0P50_MTIME=$(echo "$B0P50_STAT" | awk '{print $2}')
    B0P50_STEP=$(echo "$RAW" | grep "^B0P50_STEP:" | cut -d: -f2-)
    B0P50_MEAN=$(echo "$RAW" | grep "^B0P50_MEAN:" | cut -d: -f2-)
    B0P50_MAX=$(echo "$RAW" | grep "^B0P50_MAX:" | cut -d: -f2-)
    B0P50_CLIP=$(echo "$RAW" | grep "^B0P50_CLIP:" | cut -d: -f2-)
    B0P50_VAL=$(echo "$RAW" | grep "^B0P50_VAL:" | cut -d: -f2-)
    B0P50_REW=$(echo "$RAW" | grep "^B0P50_REW:" | cut -d: -f2-)
    B0P50_BYTES=$(echo "$RAW" | grep "^B0P50_BYTES:" | cut -d: -f2-)
    B0P50_RECON=$(echo "$RAW" | grep "^B0P50_RECON:" | cut -d: -f2-)
    B0P50_ANCH=$(echo "$RAW" | grep "^B0P50_ANCH:" | cut -d: -f2-)
    B0P50_COLD=$(echo "$RAW" | grep "^B0P50_COLD:" | cut -d: -f2-)
    B0P50_ERRS=$(echo "$RAW" | grep "^B0P50_ERRS:" | cut -d: -f2-)
    B0P75_STAT=$(echo "$RAW" | grep "^B0P75_STAT:" | cut -d: -f2-)
    B0P75_SIZE=$(echo "$B0P75_STAT" | awk '{print $1}')
    B0P75_VAL=$(echo "$RAW" | grep "^B0P75_VAL:" | cut -d: -f2-)
    B0P75_STEP=$(echo "$RAW" | grep "^B0P75_STEP:" | cut -d: -f2-)
    B1P00_STAT=$(echo "$RAW" | grep "^B1P00_STAT:" | cut -d: -f2-)
    B1P00_SIZE=$(echo "$B1P00_STAT" | awk '{print $1}')
    B1P00_VAL=$(echo "$RAW" | grep "^B1P00_VAL:" | cut -d: -f2-)
    B1P00_STEP=$(echo "$RAW" | grep "^B1P00_STEP:" | cut -d: -f2-)

    # GPU stall detection
    ALL_GPU_UTIL=$(echo "$RAW" | grep "^GPU:" | awk -F'[,%]' '{sum+=$2; count++} END {if(count>0) print sum/count; else print 0}')
    MAX_GPU_UTIL=$(echo "$RAW" | grep "^GPU:" | awk -F'[,%]' 'BEGIN{max=0} {if($2>max) max=$2} END{print max}')

    # Track cap pins (clip_ratio > 0)
    CLIP_VAL="${B0P50_CLIP##*:}"
    if [ -n "$CLIP_VAL" ] && [ "$CLIP_VAL" != "0.0" ] && [ -n "$CLIP_VAL" ]; then
        CONSECUTIVE_CAP_PINS=$((CONSECUTIVE_CAP_PINS + 1))
    else
        CONSECUTIVE_CAP_PINS=0
    fi

    # Track response length for P2/P3
    MEAN_VAL="${B0P50_MEAN##*:}"
    if [ -n "$MEAN_VAL" ]; then
        LEN_MEAN_HISTORY+=("$MEAN_VAL")
    fi

    log ""
    log "=== POLL $POLL_COUNT [$NOW] elapsed=${ELAPSED}s active=$ACTIVE_CELL ==="
    log "TMUX: $TMUX_STATE  AGG: $AGG  FLAGS: $FLAGS"
    log "GPU_UTIL: $(echo "$RAW" | grep "^GPU:" | tr '\n' ' ')"
    log "B0P50: size=$B0P50_SIZE mtime=$B0P50_MTIME step=$B0P50_STEP mean=$B0P50_MEAN max=$B0P50_MAX clip=$B0P50_CLIP val=$B0P50_VAL rew=$B0P50_REW"
    log "  substrate: bytes=$B0P50_BYTES recon=$B0P50_RECON anch=$B0P50_ANCH cold=$B0P50_COLD errs=$B0P50_ERRS"
    log "B0P75: size=$B0P75_SIZE step=$B0P75_STEP val=$B0P75_VAL"
    log "B1P00: size=$B1P00_SIZE step=$B1P00_STEP val=$B1P00_VAL"
    log "  consecutive_cap_pins=$CONSECUTIVE_CAP_PINS  len_history=[${LEN_MEAN_HISTORY[*]}]"

    # Update heartbeat
    touch_heartbeat

    # --- EXIT CONDITIONS ---

    # SSH failure
    if [ $SSH_EXIT -ne 0 ] && [ -z "$TMUX_STATE" ]; then
        log "SSH UNREACHABLE (exit $SSH_EXIT)"
        if [ $POLL_COUNT -gt 4 ]; then
            echo '{"exit_state":"ENV_FAILURE","reason":"ssh_unreachable","elapsed_s":'$ELAPSED'}' > "$RESULT"
            break
        fi
    fi

    # Aggregate done
    if [ "$AGG" = "yes" ]; then
        log "AGGREGATE DONE FLAG DETECTED — exiting"
        # rsync all cells
        for cell in b0p50 b0p75 b1p00; do
            rsync_cell "$cell"
        done
        rsync -az -e "$RSYNC_SSH" "root@46.243.55.155:/workspace/runs/EXP-33/done.flag" "$RSYNC_DEST/" 2>/dev/null || true
        echo '{"exit_state":"DONE_AGGREGATE","elapsed_s":'$ELAPSED',"b0p50_val":"'$B0P50_VAL'","b0p75_val":"'$B0P75_VAL'","b1p00_val":"'$B1P00_VAL'"}' > "$RESULT"
        break
    fi

    # Done flags tracking
    FLAG_COUNT=$(echo "$FLAGS" | tr '|' '\n' | grep -c "flag" 2>/dev/null || echo 0)
    if echo "$FLAGS" | grep -q "done_b0p50"; then
        if [ "$DONE_B0P50" = "false" ]; then
            log "b0p50 DONE FLAG appeared — rsyncing"
            rsync_cell "b0p50"
            DONE_B0P50=true
            ACTIVE_CELL="b0p75"
            [ -n "$B0P50_VAL" ] && LAST_VAL_B0P50="$B0P50_VAL"
        fi
    fi
    if echo "$FLAGS" | grep -q "done_b0p75"; then
        if [ "$DONE_B0P75" = "false" ]; then
            log "b0p75 DONE FLAG appeared — rsyncing"
            rsync_cell "b0p75"
            DONE_B0P75=true
            ACTIVE_CELL="b1p00"
            [ -n "$B0P75_VAL" ] && LAST_VAL_B0P75="$B0P75_VAL"
        fi
    fi
    if echo "$FLAGS" | grep -q "done_b1p00"; then
        if [ "$DONE_B1P00" = "false" ]; then
            log "b1p00 DONE FLAG appeared — rsyncing"
            rsync_cell "b1p00"
            DONE_B1P00=true
            [ -n "$B1P00_VAL" ] && LAST_VAL_B1P00="$B1P00_VAL"
        fi
    fi

    # 3 done flags + tmux dead
    if [ "$DONE_B0P50" = "true" ] && [ "$DONE_B0P75" = "true" ] && [ "$DONE_B1P00" = "true" ]; then
        if [ "$TMUX_STATE" = "DEAD" ]; then
            log "3 DONE FLAGS + TMUX DEAD — all cells complete"
            echo '{"exit_state":"DONE_3FLAGS","elapsed_s":'$ELAPSED',"b0p50_val":"'$LAST_VAL_B0P50'","b0p75_val":"'$LAST_VAL_B0P75'","b1p00_val":"'$LAST_VAL_B1P00'"}' > "$RESULT"
            break
        fi
    fi

    # Tmux dead premature
    if [ "$TMUX_STATE" = "DEAD" ] && [ "$FLAG_COUNT" -lt 3 ]; then
        log "TMUX DEAD PREMATURE — only $FLAG_COUNT done flags (b0p50=$DONE_B0P50 b0p75=$DONE_B0P75 b1p00=$DONE_B1P00)"
        for cell in b0p50 b0p75 b1p00; do
            rsync_cell "$cell"
        done
        echo '{"exit_state":"TMUX_DEAD_PREMATURE","elapsed_s":'$ELAPSED',"done_b0p50":'$DONE_B0P50',"done_b0p75":'$DONE_B0P75',"done_b1p00":'$DONE_B1P00',"unexpected_termination":true}' > "$RESULT"
        break
    fi

    # Error detection (Traceback/OOM/NaN)
    if [ -n "$B0P50_ERRS" ] && [ "$B0P50_ERRS" -gt 0 ]; then
        log "ERRORS DETECTED in b0p50: count=$B0P50_ERRS — keeping polling per EXPERIMENT_FAILURE policy"
        # Don't exit — cell will exit naturally
    fi

    # Length-hack ignition detection
    # P1: >=2 consecutive cap_pins
    if [ "$CONSECUTIVE_CAP_PINS" -ge 2 ]; then
        log "IGNITION P1: consecutive_cap_pins=$CONSECUTIVE_CAP_PINS — EARLY EXIT"
        EARLY_EXIT_REASON="P1_cap_pins"
        echo '{"exit_state":"EXPERIMENT_FAILURE","reason":"ignition_P1_cap_pins","cell":"b0p50","consecutive_cap_pins":'$CONSECUTIVE_CAP_PINS',"elapsed_s":'$ELAPSED'}' > "$RESULT"
        break
    fi

    # Val@25 report (immediate)
    if [ -n "$B0P50_VAL" ] && [ "$B0P50_VAL" != "$LAST_VAL_B0P50" ]; then
        VAL_NUM="${B0P50_VAL##*:}"
        log "VAL READ b0p50: $VAL_NUM (C0 ref val@25=0.71418)"
        LAST_VAL_B0P50="$B0P50_VAL"
        # Check if clearly below C0@25 and falling — would recommend early kill
        # We report but don't auto-exit; orchestrator decides
    fi
    if [ -n "$B0P75_VAL" ] && [ "$B0P75_VAL" != "$LAST_VAL_B0P75" ]; then
        VAL_NUM="${B0P75_VAL##*:}"
        log "VAL READ b0p75: $VAL_NUM"
        LAST_VAL_B0P75="$B0P75_VAL"
    fi
    if [ -n "$B1P00_VAL" ] && [ "$B1P00_VAL" != "$LAST_VAL_B1P00" ]; then
        VAL_NUM="${B1P00_VAL##*:}"
        log "VAL READ b1p00: $VAL_NUM (expected floor ~0.63)"
        LAST_VAL_B1P00="$B1P00_VAL"
    fi

    # GPU stall detection (4 consecutive polls all <=5%)
    if [ "$(echo "$MAX_GPU_UTIL <= 5" | bc 2>/dev/null)" = "1" ]; then
        GPU_STALL_COUNT=$((GPU_STALL_COUNT + 1))
        log "GPU LOW UTIL ($GPU_STALL_COUNT/4 consecutive): max=$MAX_GPU_UTIL%"
        if [ $GPU_STALL_COUNT -ge 4 ] && [ "$TMUX_STATE" = "ALIVE" ] && [ "$AGG" != "yes" ]; then
            log "GPU STALL DETECTED — 4 consecutive polls all GPUs <=5% while tmux ALIVE"
            echo '{"exit_state":"GPU_STALL","elapsed_s":'$ELAPSED',"recommendation":"teardown_only"}' > "$RESULT"
            break
        fi
    else
        GPU_STALL_COUNT=0
    fi

    # Rsync b0p50 periodically (every 5 polls) for heartbeat and freshness
    if [ $((POLL_COUNT % 5)) -eq 0 ]; then
        rsync -az -e "$RSYNC_SSH" \
            "root@46.243.55.155:/workspace/verl/runs/b0p50/train.log" \
            "$RSYNC_DEST/b0p50/" 2>/dev/null || true
        touch_heartbeat
    fi

    sleep 30
done

log ""
log "=== MONITORING LOOP ENDED: poll_count=$POLL_COUNT elapsed=$(( $(date +%s) - START_TS ))s ==="
log "LAST VALS: b0p50=$LAST_VAL_B0P50 b0p75=$LAST_VAL_B0P75 b1p00=$LAST_VAL_B1P00"

# Final rsync
for cell in b0p50 b0p75 b1p00; do
    rsync -az -e "$RSYNC_SSH" \
        "root@46.243.55.155:/workspace/verl/runs/${cell}/train.log" \
        "$RSYNC_DEST/${cell}/" 2>/dev/null || true
done
rsync -az -e "$RSYNC_SSH" \
    "root@46.243.55.155:/workspace/runs/EXP-33/" \
    "$RSYNC_DEST/" --include="done*.flag" --include="driver_resume.log" --exclude="*" 2>/dev/null || true
