#!/bin/bash
# EXP-32 monitor poll loop v2
# Tracks trip-wires at STEP level, not poll level (avoids false positives from same-step being polled twice)
# Appends to monitor-detail.log

LOGFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/monitor-detail.log"
STATUSFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/poll_status.txt"
SSH="ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p 40154 root@46.243.55.134"
LOG="/workspace/verl/runs/exp32_signed_ema_a0p5_validM/train.log"

# Trip-wire state (tracked at STEP level, not poll level)
LAST_SEEN_STEP=-1
P1_CONSECUTIVE_STEPS=0   # consecutive STEPS (not polls) with clip>0
EARLY_RLEN_MEAN=""        # set once steps 1-5 are seen
STALL_COUNT=0
POLL=7  # starting at poll 7 (polls 1-6 done above)
MAX_POLLS=80
WANDB_POLL_EVERY=3
LAST_WANDB_POLL=0

# Load secrets for WandB
source ~/.config/verl-research/secrets.env 2>/dev/null

echo "=== POLL LOOP V2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOGFILE"
echo "NOTE: v2 tracks trip-wires at STEP level to avoid false positives from same-step multi-poll" >> "$LOGFILE"
echo "NOTE: Restart from poll 7, step 7 already confirmed (clip=0, rlen_max=815)" >> "$LOGFILE"
echo "NOTE: P1 FALSE POSITIVE from v1 - steps 4 and 6 had isolated clips (non-consecutive steps)" >> "$LOGFILE"

# State from previous analysis:
# Step 7: rlen_mean=271, rlen_max=815, clip=0.0 - CLEAR
# Early mean (steps 1-5): ~280 average
EARLY_RLEN_MEAN=280
LAST_SEEN_STEP=7

while [ $POLL -le $MAX_POLLS ]; do
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # SSH probe
    PROBE=$($SSH "
LOG=$LOG
echo \"TMUX:\$(tmux has-session -t exp-32-46_243_55_134 2>/dev/null && echo ALIVE || echo DEAD)\"
echo \"DONE:\$(ls /workspace/verl/runs/exp32_signed_ema_a0p5_validM/done*.flag 2>/dev/null | wc -l)\"
echo \"AGG:\$(test -f /workspace/verl/runs/exp32_signed_ema_a0p5_validM/done.flag && echo YES || echo NO)\"
echo \"SIZE:\$(stat -c%s \$LOG 2>/dev/null)\"
echo \"TB:\$(grep -ac 'Traceback (most recent call last)' \$LOG 2>/dev/null || echo 0)\"
echo \"OOM:\$(grep -ac 'CUDA out of memory' \$LOG 2>/dev/null || echo 0)\"
echo \"NAN:\$(grep -ac 'NaN detected' \$LOG 2>/dev/null || echo 0)\"
LASTLINE=\$(grep -a 'training/global_step:' \$LOG 2>/dev/null | tail -1)
echo \"STEP:\$(echo \"\$LASTLINE\" | grep -oP 'training/global_step:\\K[0-9]+' | head -1)\"
echo \"REWARD:\$(echo \"\$LASTLINE\" | grep -oP 'critic/score/mean:\\K[0-9.]+' | head -1)\"
echo \"GRADNORM:\$(echo \"\$LASTLINE\" | grep -oP 'actor/grad_norm:\\K[0-9.]+' | head -1)\"
echo \"RLEN_MEAN:\$(echo \"\$LASTLINE\" | grep -oP 'response_length/mean:\\K[0-9.]+' | head -1)\"
echo \"RLEN_MAX:\$(echo \"\$LASTLINE\" | grep -oP 'response_length/max:\\K[0-9.]+' | head -1)\"
echo \"RLEN_CLIP:\$(echo \"\$LASTLINE\" | grep -oP 'response_length/clip_ratio:\\K[0-9.]+' | head -1)\"
echo \"BYTES_RATIO:\$(echo \"\$LASTLINE\" | grep -oP 'actor/comm/bytes_ratio:\\K[0-9.]+' | head -1)\"
echo \"RECON_ERR:\$(echo \"\$LASTLINE\" | grep -oP 'powersgd_reconstruction_rel_error:\\K[0-9.]+' | head -1)\"
echo \"ANCHOR_BACK:\$(echo \"\$LASTLINE\" | grep -oP 'anchor_backwards:\\K[0-9.]+' | head -1)\"
grep -a 'val-core/openai/gsm8k/acc/mean@1' \$LOG 2>/dev/null | grep -v '\[comm_eff\]' | tail -1 | grep -oP 'mean@1:\\K[0-9.]+' | xargs -I{} echo \"VAL:{}\"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
" 2>/dev/null | grep -v "Welcome to vast\|Have fun")

    TMUX=$(echo "$PROBE" | grep "^TMUX:" | cut -d: -f2)
    DONE=$(echo "$PROBE" | grep "^DONE:" | cut -d: -f2)
    AGG=$(echo "$PROBE" | grep "^AGG:" | cut -d: -f2)
    SIZE=$(echo "$PROBE" | grep "^SIZE:" | cut -d: -f2)
    TB=$(echo "$PROBE" | grep "^TB:" | cut -d: -f2 | tr -d ' ' | head -1)
    OOM=$(echo "$PROBE" | grep "^OOM:" | cut -d: -f2 | tr -d ' ' | head -1)
    NAN=$(echo "$PROBE" | grep "^NAN:" | cut -d: -f2 | tr -d ' ' | head -1)
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

    # Only update step-level trip-wires if step advanced
    NEW_STEP=0
    if [ -n "$STEP" ] && [ "$STEP" != "$LAST_SEEN_STEP" ]; then
        NEW_STEP=1
        LAST_SEEN_STEP="$STEP"

        # P1: step-level clip_ratio > 0 check
        CLIP_GT0=0
        if [ -n "$RLEN_CLIP" ]; then
            python3 -c "import sys; sys.exit(0 if float('${RLEN_CLIP}') > 0 else 1)" 2>/dev/null && CLIP_GT0=1
        fi
        if [ "$CLIP_GT0" = "1" ]; then
            P1_CONSECUTIVE_STEPS=$((P1_CONSECUTIVE_STEPS + 1))
        else
            P1_CONSECUTIVE_STEPS=0
        fi
    fi

    # GPU stall check (every poll)
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

    # P3: rlen_mean > 2x early (early ~280)
    P3_FIRE=0
    if [ -n "$RLEN_MEAN" ]; then
        python3 -c "import sys; sys.exit(0 if float('${RLEN_MEAN}') > 2*${EARLY_RLEN_MEAN} else 1)" 2>/dev/null && P3_FIRE=1
    fi

    # E1: rlen_max > 4k at steps 10-30
    E1_FIRE=0
    if [ -n "$STEP" ] && [ "$STEP" -ge 10 ] && [ "$STEP" -le 30 ] && [ -n "$RLEN_MAX" ]; then
        python3 -c "import sys; sys.exit(0 if float('${RLEN_MAX}') > 4000 else 1)" 2>/dev/null && E1_FIRE=1
    fi

    TRIP_STATUS="P1=${P1_CONSECUTIVE_STEPS}/2(step-level) P3=${P3_FIRE} E1=${E1_FIRE} STALL=${STALL_COUNT}/4"

    # WandB check every 3rd poll
    WANDB_INFO=""
    if [ $(( (POLL - LAST_WANDB_POLL) )) -ge $WANDB_POLL_EVERY ]; then
        LAST_WANDB_POLL=$POLL
        WB=$(curl -s -X POST "https://api.wandb.ai/graphql" \
          -H "Authorization: Bearer $WANDB_API_KEY" \
          -H "Content-Type: application/json" \
          -d '{"query":"{ project(entityName: \"shamanework-pl\", name: \"verl_compression_research\") { run(name: \"cwz1hu5p\") { name state historyLineCount summaryMetrics } } }"}' 2>/dev/null)
        WB_STATE=$(echo "$WB" | python3 -c "import json,sys; d=json.load(sys.stdin); r=d.get('data',{}).get('project',{}).get('run',{}); print(f'state={r.get(\"state\")} histLines={r.get(\"historyLineCount\")}')" 2>/dev/null)
        WANDB_INFO="wandb: $WB_STATE"
    fi

    # Log this poll
    {
        echo ""
        echo "=== POLL ${POLL} === ${TIMESTAMP} ==="
        echo "tmux: ${TMUX} | done_flags: ${DONE} | agg: ${AGG} | new_step: ${NEW_STEP}"
        echo "log_size: ${SIZE} bytes | step: ${STEP}/50"
        echo "errors: TB=${TB} OOM=${OOM} NaN=${NAN}"
        echo "bytes_ratio: ${BYTES_RATIO} | recon_err: ${RECON_ERR}"
        echo "grad_norm: ${GRADNORM} | reward: ${REWARD}"
        echo "rlen_mean: ${RLEN_MEAN} | rlen_max: ${RLEN_MAX} | rlen_clip: ${RLEN_CLIP}"
        echo "anchor_backwards: ${ANCHOR_BACK}"
        [ -n "$VAL" ] && echo "*** VAL RESULT: val-core/gsm8k/acc/mean@1=${VAL} ***"
        [ -n "$WANDB_INFO" ] && echo "$WANDB_INFO"
        echo "gpu_util:"
        echo "$GPU_LINES"
        echo "TRIP_WIRES: ${TRIP_STATUS}"
    } >> "$LOGFILE"

    # Write status file
    echo "POLL=${POLL} STEP=${STEP} TMUX=${TMUX} DONE=${DONE} AGG=${AGG} REWARD=${REWARD} VAL=${VAL} RLEN_MEAN=${RLEN_MEAN} RLEN_MAX=${RLEN_MAX} P1_STEPS=${P1_CONSECUTIVE_STEPS} P3=${P3_FIRE} E1=${E1_FIRE} STALL=${STALL_COUNT} TIME=${TIMESTAMP}" > "$STATUSFILE"

    # --- EXIT CONDITIONS ---

    if [ "$AGG" = "YES" ]; then
        echo "EXIT: DONE_AGGREGATE at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:DONE_AGGREGATE STEP:${STEP}" > "$STATUSFILE"
        exit 0
    fi

    if [ "${DONE:-0}" -ge 1 ] && [ "$TMUX" = "DEAD" ]; then
        echo "EXIT: DONE_FLAGS=${DONE} + TMUX_DEAD" >> "$LOGFILE"
        echo "EXIT:DONE_FLAGS_DEAD STEP:${STEP}" > "$STATUSFILE"
        exit 0
    fi

    if [ "$TMUX" = "DEAD" ] && [ "${DONE:-0}" -lt 1 ]; then
        echo "EXIT: TMUX_DEAD_PREMATURE done_flags=${DONE}" >> "$LOGFILE"
        echo "EXIT:TMUX_DEAD_PREMATURE STEP:${STEP}" > "$STATUSFILE"
        exit 1
    fi

    if [ "$STALL_COUNT" -ge 4 ]; then
        echo "EXIT: GPU_STALL 4 consecutive polls" >> "$LOGFILE"
        echo "EXIT:GPU_STALL STEP:${STEP}" > "$STATUSFILE"
        exit 2
    fi

    # IGNITION checks (step-level)
    if [ "$P1_CONSECUTIVE_STEPS" -ge 2 ]; then
        echo "EXIT: IGNITION P1 fired - ${P1_CONSECUTIVE_STEPS} consecutive STEPS with clip>0 at step ${STEP} rlen_mean=${RLEN_MEAN}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P1_STEPS STEP:${STEP} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    if [ "$P3_FIRE" = "1" ]; then
        echo "EXIT: IGNITION P3 - rlen_mean=${RLEN_MEAN} > 2x early ${EARLY_RLEN_MEAN} at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P3 STEP:${STEP} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    if [ "$E1_FIRE" = "1" ]; then
        echo "EXIT: IGNITION E1 - rlen_max=${RLEN_MAX} > 4000 at step ${STEP} (steps 10-30)" >> "$LOGFILE"
        echo "EXIT:IGNITION_E1 STEP:${STEP} RLEN_MAX:${RLEN_MAX}" > "$STATUSFILE"
        exit 3
    fi

    POLL=$((POLL + 1))
    sleep 30
done

echo "EXIT: TIMEOUT after ${MAX_POLLS} polls" >> "$LOGFILE"
echo "EXIT:TIMEOUT STEP:${STEP}" > "$STATUSFILE"
exit 4
