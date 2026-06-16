#!/bin/bash
# EXP-32 monitor poll loop
# Runs on laptop, polls remote box every 30s, appends to monitor-detail.log

LOGFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/monitor-detail.log"
STATUSFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/poll_status.txt"
SSH="ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p 40154 root@46.243.55.134"
RDIR="/workspace/verl/runs/exp32_signed_ema_a0p5_validM"
LOG="$RDIR/train.log"

# Trip-wire state
P1_CONSECUTIVE=0
LAST_CLIP=0
PREV_RLEN_MEAN=""
STALL_COUNT=0
POLL=3
MAX_POLLS=80  # 40 min at 30s each
WANDB_POLL_EVERY=3

# Load secrets for WandB
source ~/.config/verl-research/secrets.env 2>/dev/null

echo "=== POLL LOOP START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOGFILE"

while [ $POLL -le $MAX_POLLS ]; do
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # SSH probe
    PROBE=$($SSH '
LOG=/workspace/verl/runs/exp32_signed_ema_a0p5_validM/train.log
echo "TMUX:$(tmux has-session -t exp-32-46_243_55_134 2>/dev/null && echo ALIVE || echo DEAD)"
echo "DONE:$(ls /workspace/verl/runs/exp32_signed_ema_a0p5_validM/done*.flag 2>/dev/null | wc -l)"
echo "AGG:$(test -f /workspace/verl/runs/exp32_signed_ema_a0p5_validM/done.flag && echo YES || echo NO)"
echo "SIZE:$(stat -c%s $LOG 2>/dev/null)"
echo "TB:$(grep -ac "Traceback (most recent call last)" $LOG 2>/dev/null || echo 0)"
echo "OOM:$(grep -ac "CUDA out of memory" $LOG 2>/dev/null || echo 0)"
echo "NAN:$(grep -ac "NaN detected\|non-finite" $LOG 2>/dev/null || echo 0)"
LASTLINE=$(grep -a "training/global_step:" $LOG 2>/dev/null | tail -1)
echo "STEP:$(echo "$LASTLINE" | grep -oP "training/global_step:\K[0-9]+" | head -1)"
echo "REWARD:$(echo "$LASTLINE" | grep -oP "critic/score/mean:\K[0-9.]+" | head -1)"
echo "GRADNORM:$(echo "$LASTLINE" | grep -oP "actor/grad_norm:\K[0-9.]+" | head -1)"
echo "RLEN_MEAN:$(echo "$LASTLINE" | grep -oP "response_length/mean:\K[0-9.]+" | head -1)"
echo "RLEN_MAX:$(echo "$LASTLINE" | grep -oP "response_length/max:\K[0-9.]+" | head -1)"
echo "RLEN_CLIP:$(echo "$LASTLINE" | grep -oP "response_length/clip_ratio:\K[0-9.]+" | head -1)"
echo "BYTES_RATIO:$(echo "$LASTLINE" | grep -oP "actor/comm/bytes_ratio:\K[0-9.]+" | head -1)"
echo "RECON_ERR:$(echo "$LASTLINE" | grep -oP "powersgd_reconstruction_rel_error:\K[0-9.]+" | head -1)"
echo "ANCHOR_BACK:$(echo "$LASTLINE" | grep -oP "anchor_backwards:\K[0-9.]+" | head -1)"
# Val result if present
grep -a "val-core/openai/gsm8k/acc/mean@1" $LOG 2>/dev/null | grep -v "\[comm_eff\]" | tail -1 | grep -oP "val-core/openai/gsm8k/acc/mean@1:\K[0-9.]+" | xargs -I{} echo "VAL:{}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
' 2>/dev/null | grep -v "Welcome to vast\|Have fun")

    TMUX=$(echo "$PROBE" | grep "^TMUX:" | cut -d: -f2)
    DONE=$(echo "$PROBE" | grep "^DONE:" | cut -d: -f2)
    AGG=$(echo "$PROBE" | grep "^AGG:" | cut -d: -f2)
    SIZE=$(echo "$PROBE" | grep "^SIZE:" | cut -d: -f2)
    TB=$(echo "$PROBE" | grep "^TB:" | cut -d: -f2 | tr -d ' ')
    OOM=$(echo "$PROBE" | grep "^OOM:" | cut -d: -f2 | tr -d ' ')
    NAN=$(echo "$PROBE" | grep "^NAN:" | cut -d: -f2 | tr -d ' ')
    STEP=$(echo "$PROBE" | grep "^STEP:" | cut -d: -f2)
    REWARD=$(echo "$PROBE" | grep "^REWARD:" | cut -d: -f2)
    GRADNORM=$(echo "$PROBE" | grep "^GRADNORM:" | cut -d: -f2)
    RLEN_MEAN=$(echo "$PROBE" | grep "^RLEN_MEAN:" | cut -d: -f2)
    RLEN_MAX=$(echo "$PROBE" | grep "^RLEN_MAX:" | cut -d: -f2)
    RLEN_CLIP=$(echo "$PROBE" | grep "^RLEN_CLIP:" | cut -d: -f2)
    BYTES_RATIO=$(echo "$PROBE" | grep "^BYTES_RATIO:" | cut -d: -f2)
    RECON_ERR=$(echo "$PROBE" | grep "^RECON_ERR:" | cut -d: -f2)
    ANCHOR_BACK=$(echo "$PROBE" | grep "^ANCHOR_BACK:" | cut -d: -f2)
    VAL=$(echo "$PROBE" | grep "^VAL:" | cut -d: -f2)
    GPU_LINES=$(echo "$PROBE" | grep "^[0-9], ")

    # Check GPU stall (all GPUs <= 5%)
    ALL_UTIL=$(echo "$GPU_LINES" | awk -F', ' '{print $2}' | sed 's/ %//')
    STALL=1
    while IFS= read -r u; do
        [ -z "$u" ] && continue
        if [ "$u" -gt 5 ] 2>/dev/null; then
            STALL=0
            break
        fi
    done <<< "$ALL_UTIL"

    if [ "$STALL" = "1" ] && [ "$TMUX" = "ALIVE" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
    else
        STALL_COUNT=0
    fi

    # P1 trip-wire: consecutive clip_ratio > 0
    CLIP_GT0=0
    [ -n "$RLEN_CLIP" ] && python3 -c "exit(0 if float('${RLEN_CLIP:-0}') > 0 else 1)" 2>/dev/null && CLIP_GT0=1
    if [ "$CLIP_GT0" = "1" ]; then
        P1_CONSECUTIVE=$((P1_CONSECUTIVE + 1))
    else
        P1_CONSECUTIVE=0
    fi

    # P3: len-mean > 2x early (use 294 as early mean from step 1-3)
    EARLY_MEAN=294
    P3_FIRE=0
    if [ -n "$RLEN_MEAN" ]; then
        P3_FIRE=$(python3 -c "exit(0 if float('${RLEN_MEAN}') > 2*${EARLY_MEAN} else 1)" 2>/dev/null && echo 1 || echo 0)
    fi

    # E1: len/max > 4k at steps 10-30
    E1_FIRE=0
    if [ -n "$STEP" ] && [ "$STEP" -ge 10 ] && [ "$STEP" -le 30 ] && [ -n "$RLEN_MAX" ]; then
        E1_FIRE=$(python3 -c "exit(0 if float('${RLEN_MAX}') > 4000 else 1)" 2>/dev/null && echo 1 || echo 0)
    fi

    TRIP_STATUS="P1=${P1_CONSECUTIVE}/2 P3=${P3_FIRE} E1=${E1_FIRE} STALL=${STALL_COUNT}/4"

    # Log this poll
    {
        echo ""
        echo "=== POLL ${POLL} === ${TIMESTAMP} ==="
        echo "tmux: ${TMUX} | done_flags: ${DONE} | agg: ${AGG}"
        echo "log_size: ${SIZE} bytes | step: ${STEP}/50"
        echo "errors: TB=${TB} OOM=${OOM} NaN=${NAN}"
        echo "bytes_ratio: ${BYTES_RATIO} | recon_err: ${RECON_ERR}"
        echo "grad_norm: ${GRADNORM} | reward: ${REWARD}"
        echo "rlen_mean: ${RLEN_MEAN} | rlen_max: ${RLEN_MAX} | rlen_clip: ${RLEN_CLIP}"
        echo "anchor_backwards: ${ANCHOR_BACK}"
        [ -n "$VAL" ] && echo "*** VAL RESULT: val-core/gsm8k/acc/mean@1=${VAL} ***"
        echo "gpu_util: $GPU_LINES"
        echo "TRIP_WIRES: ${TRIP_STATUS}"
    } >> "$LOGFILE"

    # Write status file for quick check
    echo "POLL=${POLL} STEP=${STEP} TMUX=${TMUX} DONE=${DONE} AGG=${AGG} REWARD=${REWARD} VAL=${VAL} RLEN_MEAN=${RLEN_MEAN} P1=${P1_CONSECUTIVE} P3=${P3_FIRE} E1=${E1_FIRE} STALL=${STALL_COUNT} TIME=${TIMESTAMP}" > "$STATUSFILE"

    # --- EXIT CONDITIONS ---

    # DONE_AGGREGATE
    if [ "$AGG" = "YES" ]; then
        echo "EXIT: DONE_AGGREGATE at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:DONE_AGGREGATE STEP:${STEP}" > "$STATUSFILE"
        exit 0
    fi

    # DONE_3FLAGS + TMUX_DEAD
    if [ "$DONE" -ge 1 ] && [ "$TMUX" = "DEAD" ]; then
        echo "EXIT: DONE_FLAGS=${DONE} + TMUX_DEAD" >> "$LOGFILE"
        echo "EXIT:DONE_FLAGS_DEAD STEP:${STEP}" > "$STATUSFILE"
        exit 0
    fi

    # TMUX_DEAD_PREMATURE
    if [ "$TMUX" = "DEAD" ] && [ "$DONE" -lt 1 ]; then
        echo "EXIT: TMUX_DEAD_PREMATURE done_flags=${DONE}" >> "$LOGFILE"
        echo "EXIT:TMUX_DEAD_PREMATURE STEP:${STEP}" > "$STATUSFILE"
        exit 1
    fi

    # GPU_STALL
    if [ "$STALL_COUNT" -ge 4 ]; then
        echo "EXIT: GPU_STALL 4 consecutive polls with all GPUs <=5%" >> "$LOGFILE"
        echo "EXIT:GPU_STALL STEP:${STEP}" > "$STATUSFILE"
        exit 2
    fi

    # IGNITION trip-wire: P1 >= 2 consecutive
    if [ "$P1_CONSECUTIVE" -ge 2 ]; then
        echo "EXIT: IGNITION P1 fired - ${P1_CONSECUTIVE} consecutive clip polls at step ${STEP} rlen_mean=${RLEN_MEAN}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P1 STEP:${STEP} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    # P3 ignition
    if [ "$P3_FIRE" = "1" ]; then
        echo "EXIT: IGNITION P3 fired - rlen_mean=${RLEN_MEAN} > 2x early ${EARLY_MEAN} at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P3 STEP:${STEP} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    # E1 ignition
    if [ "$E1_FIRE" = "1" ]; then
        echo "EXIT: IGNITION E1 fired - rlen_max=${RLEN_MAX} > 4000 at step ${STEP} (in steps 10-30)" >> "$LOGFILE"
        echo "EXIT:IGNITION_E1 STEP:${STEP} RLEN_MAX:${RLEN_MAX}" > "$STATUSFILE"
        exit 3
    fi

    # Hard error
    if [ "${TB:-0}" -gt 0 ] || [ "${OOM:-0}" -gt 0 ]; then
        echo "ALERT: HARD ERROR TB=${TB} OOM=${OOM} at step ${STEP}" >> "$LOGFILE"
        # Keep polling per plan - experiment failures are the data
    fi

    POLL=$((POLL + 1))
    sleep 30
done

echo "EXIT: TIMEOUT after ${MAX_POLLS} polls" >> "$LOGFILE"
echo "EXIT:TIMEOUT STEP:${STEP}" > "$STATUSFILE"
exit 4
