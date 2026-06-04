#!/bin/bash
# Single poll script for EXP-20 r=77 monitoring
# Outputs: TMUX_STATUS STEP DONE_STATUS GPU_PCTS ERROR_COUNT
# Called from monitor loop

set -e

OUT=$(ssh -i /Users/shamane/.ssh/vast_ai_name \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=12 \
  -o ServerAliveInterval=5 \
  -o ServerAliveCountMax=3 \
  -p 40009 root@84.8.106.109 '
    tmux has-session -t exp-20-powersgd-r77 2>&1 && echo "TMUX=ALIVE" || echo "TMUX=DEAD"
    ls /workspace/runs/EXP-20/powersgd_r77_arm.done 2>/dev/null && echo "R77=FOUND" || echo "R77=ABSENT"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    grep -aoE "step:[0-9]+ - global_seqlen" /workspace/runs/EXP-20/ce_powersgd_r77_clean5_50s_gsm8k.log 2>/dev/null | tail -1 | grep -oE "step:[0-9]+" || echo "step:?"
    grep -c "Traceback" /workspace/runs/EXP-20/ce_powersgd_r77_clean5_50s_gsm8k.log 2>/dev/null || echo "0"
  ' 2>&1 | grep -v "Welcome\|Have fun\|authentication fails")

echo "$OUT"
