#!/usr/bin/env bash
# EXP-20 matched-budget follow-up — PowerSGD r=77 arm (the EQUAL-budget
# head-to-head the r=102 (+33% budget) run couldn't give). r=77 is byte-matched
# to the mask p=0.95 at H=1536 (77 ~= 76.8 coords). ONLY change from r=102 is
# rank=77. Mask bar = val-acc 0.7384 @step50; r=102 (+33%) scored 0.7437.
# Same canonical launcher (HEAD f748dbc1: backgrounds training + tail --pid
# watcher, propagates exit status, writes per-cell log + done.flag).
set -uo pipefail   # deliberately NOT -e: a benign non-zero launcher exit (verl
                   # SIGKILLs Ray dataloader workers at teardown -> non-zero RC
                   # despite training+metrics complete) must still let us write
                   # the recovery done-flag + status line.
cd /workspace/verl
echo "=== EXP-20 powersgd r=77 arm relaunch @ $(git rev-parse --short HEAD) ($(date -u +%FT%TZ)) ==="

export PROJECT_NAME=verl_compression_research
export WANDB_ENTITY="${WANDB_ENTITY:-shamanework-pl}"
export EXPERIMENT_NAME=ce_powersgd_r77_clean5_50s_gsm8k
export TOTAL_TRAINING_STEPS=50
export VAL_BEFORE_TRAIN=True
export SAVE_FREQ=50
export TEST_FREQ=25
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}"

# Codec: PowerSGD r=77 + clean@5, anchor OFF, spectral OFF. ONLY rank differs.
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77             # <-- the ONLY change vs r=102 (equal-budget vs mask p=0.95 @ H=1536)
export COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true
export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_SYNC_BASIS=true
export COMM_EFF_POWERSGD_QR_DTYPE=fp32
export COMM_EFF_CLEAN_CADENCE=5
export COMM_EFF_ANCHOR_ENABLED=false
export COMM_EFF_SPECTRAL_ENABLED=false

export LOG="/workspace/runs/EXP-20/${EXPERIMENT_NAME}.log"
mkdir -p "$(dirname "$LOG")"

echo "=== launching ${EXPERIMENT_NAME}: compression_type=powersgd rank=77 clean_cadence=5 sync_basis=true (50 steps) ==="
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RC=$?

echo "$(date -Iseconds) powersgd-r77-arm exit_rc=$RC" > /workspace/runs/EXP-20/powersgd_r77_arm.done
echo "=== EXP-20 powersgd r=77 arm relaunch finished rc=$RC at $(date -u +%FT%TZ) ==="
