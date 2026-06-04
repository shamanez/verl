#!/usr/bin/env bash
# EXP-20 RECOVERY — relaunch ONLY the PowerSGD r=102 arm (the mask bar=0.7384 is
# already locked in; the driver launch.sh exited after the mask cell instead of
# advancing to step 2). Runs the canonical comm-eff launcher directly with the
# step-2 codec knobs. The canonical launcher (HEAD f748dbc1) backgrounds
# training + tail --pid watcher, propagates the training exit status, writes the
# per-cell log ($LOG) and touches runs/<EXP>/done.flag on completion.
set -uo pipefail   # NOTE: deliberately NOT -e here so a non-zero launcher exit
                   # still lets us write the recovery done-flag + status line.
cd /workspace/verl
echo "=== EXP-20 powersgd-arm relaunch @ $(git rev-parse --short HEAD) ($(date -u +%FT%TZ)) ==="

# Full-run schedule (50 steps), val before train + periodic eval (mirrors step 2).
export PROJECT_NAME=verl_compression_research
export WANDB_ENTITY="${WANDB_ENTITY:-shamanework-pl}"
export EXPERIMENT_NAME=ce_powersgd_r102_clean5_50s_gsm8k
export TOTAL_TRAINING_STEPS=50
export VAL_BEFORE_TRAIN=True
export SAVE_FREQ=50
export TEST_FREQ=25
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}"

# Codec: PowerSGD r=102 + clean@5, anchor OFF, spectral OFF (Run A — the candidate).
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=102            # KEEP 102 (operator decision; budget mismatch noted)
export COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true
export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_SYNC_BASIS=true     # single shared consensus Q across DP ranks
export COMM_EFF_POWERSGD_QR_DTYPE=fp32
export COMM_EFF_CLEAN_CADENCE=5
export COMM_EFF_ANCHOR_ENABLED=false
export COMM_EFF_SPECTRAL_ENABLED=false

export LOG="/workspace/runs/EXP-20/${EXPERIMENT_NAME}.log"
mkdir -p "$(dirname "$LOG")"

echo "=== launching ${EXPERIMENT_NAME}: compression_type=powersgd rank=102 clean_cadence=5 sync_basis=true (50 steps) ==="
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RC=$?

# Recovery-specific completion marker (the canonical launcher already touches
# runs/<EXP>/done.flag on a clean exit; this is a belt-and-braces record with RC).
echo "$(date -Iseconds) powersgd-arm exit_rc=$RC" > /workspace/runs/EXP-20/powersgd_arm.done
echo "=== EXP-20 powersgd-arm relaunch finished rc=$RC at $(date -u +%FT%TZ) ==="
