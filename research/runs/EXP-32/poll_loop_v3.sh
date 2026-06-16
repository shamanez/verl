#!/bin/bash
# EXP-32 monitor poll loop v3
# - Tracks trip-wires at STEP level
# - E1: fires only if rlen_max > 4k for >=2 consecutive NEW steps (not isolated outliers)
# - P1: step-level clip_ratio > 0 for >=2 consecutive new steps
# - Starts from step 13 (already confirmed clean)

LOGFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/monitor-detail.log"
STATUSFILE="/Users/shamane/Documents/verl/research/runs/EXP-32/poll_status.txt"
SSH="ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p 40154 root@46.243.55.134"
LOG="/workspace/verl/runs/exp32_signed_ema_a0p5_validM/train.log"

# Step-level state
LAST_SEEN_STEP=13
P1_CONSECUTIVE_STEPS=0   # consecutive STEPS with clip>0
E1_CONSECUTIVE_STEPS=0   # consecutive STEPS with max>4000
EARLY_RLEN_MEAN=280       # from steps 1-9
STALL_COUNT=0
POLL=16
MAX_POLLS=80
WANDB_POLL_EVERY=3
LAST_WANDB_POLL=0

source ~/.config/verl-research/secrets.env 2>/dev/null

echo "=== POLL LOOP V3 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOGFILE"
echo "NOTE: v3 requires 2 consecutive steps with max>4000 for E1 (removes isolated-outlier false positive)" >> "$LOGFILE"
echo "NOTE: Starting from step 13 (rlen_max=1028, clip=0.0, reward=0.570 - ALL CLEAR)" >> "$LOGFILE"

while [ $POLL -le $MAX_POLLS ]; do
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

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
grep -a 'val-core/openai/gsm8k/acc/mean@1' \$LOG 2>/dev/null | grep -v '\[comm_eff\]' | tail -3 | grep -oP 'mean@1:\\K[0-9.]+' | tail -1 | xargs -I{} echo \"VAL:{}\"
# Also check for any val lines specifically for step 25
VAL25LINE=\$(grep -a 'step:25 - val-core' \$LOG 2>/dev/null | tail -1)
[ -n \"\$VAL25LINE\" ] && echo \"VAL25_STEP25:\$(echo \"\$VAL25LINE\" | grep -oP 'mean@1:\\K[0-9.]+' | head -1)\"
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
    VAL25=$(echo "$PROBE" | grep "^VAL25_STEP25:" | cut -d: -f2)
    GPU_LINES=$(echo "$PROBE" | grep "^[0-9], ")

    # Step-level updates only when step advances
    NEW_STEP=0
    if [ -n "$STEP" ] && [ "$STEP" != "$LAST_SEEN_STEP" ]; then
        NEW_STEP=1
        LAST_SEEN_STEP="$STEP"

        # P1: consecutive steps with clip>0
        CLIP_GT0=0
        if [ -n "$RLEN_CLIP" ]; then
            python3 -c "import sys; sys.exit(0 if float('${RLEN_CLIP}') > 0 else 1)" 2>/dev/null && CLIP_GT0=1
        fi
        [ "$CLIP_GT0" = "1" ] && P1_CONSECUTIVE_STEPS=$((P1_CONSECUTIVE_STEPS + 1)) || P1_CONSECUTIVE_STEPS=0

        # E1: consecutive steps with max>4000 (in steps 10-30)
        MAX_GT4K=0
        if [ -n "$RLEN_MAX" ] && [ "$STEP" -ge 10 ] && [ "$STEP" -le 30 ]; then
            python3 -c "import sys; sys.exit(0 if float('${RLEN_MAX}') > 4000 else 1)" 2>/dev/null && MAX_GT4K=1
        fi
        [ "$MAX_GT4K" = "1" ] && E1_CONSECUTIVE_STEPS=$((E1_CONSECUTIVE_STEPS + 1)) || E1_CONSECUTIVE_STEPS=0
    fi

    # GPU stall (all GPUs <=5% for 4 consecutive polls)
    ALL_UTIL=$(echo "$GPU_LINES" | awk -F', ' '{print $2}' | sed 's/ %//')
    STALL=1
    while IFS= read -r u; do
        [ -z "$u" ] && continue
        [ "$u" -gt 5 ] 2>/dev/null && { STALL=0; break; }
    done <<< "$ALL_UTIL"
    [ "$STALL" = "1" ] && [ "$TMUX" = "ALIVE" ] && STALL_COUNT=$((STALL_COUNT + 1)) || STALL_COUNT=0

    # P3: mean > 2x early
    P3_FIRE=0
    [ -n "$RLEN_MEAN" ] && python3 -c "import sys; sys.exit(0 if float('${RLEN_MEAN}') > 2*${EARLY_RLEN_MEAN} else 1)" 2>/dev/null && P3_FIRE=1

    TRIP_STATUS="P1=${P1_CONSECUTIVE_STEPS}/2steps P3=${P3_FIRE} E1=${E1_CONSECUTIVE_STEPS}/2steps STALL=${STALL_COUNT}/4polls"

    # WandB check
    WANDB_INFO=""
    if [ $(( POLL - LAST_WANDB_POLL )) -ge $WANDB_POLL_EVERY ]; then
        LAST_WANDB_POLL=$POLL
        WB=$(curl -s -X POST "https://api.wandb.ai/graphql" \
          -H "Authorization: Bearer $WANDB_API_KEY" \
          -H "Content-Type: application/json" \
          -d '{"query":"{ project(entityName: \"shamanework-pl\", name: \"verl_compression_research\") { run(name: \"cwz1hu5p\") { name state historyLineCount summaryMetrics } } }"}' 2>/dev/null)
        WB_STATE=$(echo "$WB" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('data',{}).get('project',{}).get('run',{})
s=r.get('summaryMetrics','{}')
sm=json.loads(s) if s else {}
val=sm.get('val-core/openai/gsm8k/acc/mean@1','')
print(f'state={r.get(\"state\")} histLines={r.get(\"historyLineCount\")} val={val} step={sm.get(\"training/global_step\",\"\")} reward={sm.get(\"critic/score/mean\",\"\")}')
" 2>/dev/null)
        WANDB_INFO="wandb: $WB_STATE"
    fi

    # Log poll
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
        [ -n "$VAL" ] && echo "*** VAL: val-core/gsm8k/acc/mean@1=${VAL} ***"
        [ -n "$VAL25" ] && echo "*** VAL@25 RESULT: ${VAL25} ***"
        [ -n "$WANDB_INFO" ] && echo "$WANDB_INFO"
        echo "gpu_util:"
        echo "$GPU_LINES"
        echo "TRIP_WIRES: ${TRIP_STATUS}"
    } >> "$LOGFILE"

    echo "POLL=${POLL} STEP=${STEP} TMUX=${TMUX} DONE=${DONE} AGG=${AGG} REWARD=${REWARD} VAL=${VAL} RLEN_MEAN=${RLEN_MEAN} RLEN_MAX=${RLEN_MAX} P1_STEPS=${P1_CONSECUTIVE_STEPS} P3=${P3_FIRE} E1_STEPS=${E1_CONSECUTIVE_STEPS} STALL=${STALL_COUNT} TIME=${TIMESTAMP}" > "$STATUSFILE"

    # EXIT CONDITIONS
    if [ "$AGG" = "YES" ]; then
        echo "EXIT: DONE_AGGREGATE at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:DONE_AGGREGATE STEP:${STEP} VAL:${VAL}" > "$STATUSFILE"
        exit 0
    fi

    if [ "${DONE:-0}" -ge 1 ] && [ "$TMUX" = "DEAD" ]; then
        echo "EXIT: DONE_FLAGS+TMUX_DEAD at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:DONE_FLAGS_DEAD STEP:${STEP} VAL:${VAL}" > "$STATUSFILE"
        exit 0
    fi

    if [ "$TMUX" = "DEAD" ] && [ "${DONE:-0}" -lt 1 ]; then
        echo "EXIT: TMUX_DEAD_PREMATURE done_flags=${DONE}" >> "$LOGFILE"
        echo "EXIT:TMUX_DEAD_PREMATURE STEP:${STEP}" > "$STATUSFILE"
        exit 1
    fi

    if [ "$STALL_COUNT" -ge 4 ]; then
        echo "EXIT: GPU_STALL" >> "$LOGFILE"
        echo "EXIT:GPU_STALL STEP:${STEP}" > "$STATUSFILE"
        exit 2
    fi

    # IGNITION: P1 >=2 consecutive steps
    if [ "$P1_CONSECUTIVE_STEPS" -ge 2 ]; then
        echo "EXIT: IGNITION P1 - ${P1_CONSECUTIVE_STEPS} consecutive steps clip>0 at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P1 STEP:${STEP} RLEN_MEAN:${RLEN_MEAN} RLEN_MAX:${RLEN_MAX}" > "$STATUSFILE"
        exit 3
    fi

    # IGNITION: P3
    if [ "$P3_FIRE" = "1" ]; then
        echo "EXIT: IGNITION P3 - rlen_mean=${RLEN_MEAN} > 2x early=${EARLY_RLEN_MEAN} at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:IGNITION_P3 STEP:${STEP} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    # IGNITION: E1 >=2 consecutive steps in steps 10-30
    if [ "$E1_CONSECUTIVE_STEPS" -ge 2 ]; then
        echo "EXIT: IGNITION E1 - ${E1_CONSECUTIVE_STEPS} consecutive steps max>4000 at step ${STEP}" >> "$LOGFILE"
        echo "EXIT:IGNITION_E1 STEP:${STEP} RLEN_MAX:${RLEN_MAX} RLEN_MEAN:${RLEN_MEAN}" > "$STATUSFILE"
        exit 3
    fi

    POLL=$((POLL + 1))
    sleep 30
done

echo "EXIT: TIMEOUT poll ${MAX_POLLS}" >> "$LOGFILE"
echo "EXIT:TIMEOUT STEP:${STEP}" > "$STATUSFILE"
exit 4
