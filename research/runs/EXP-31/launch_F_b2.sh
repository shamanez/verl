#!/usr/bin/env bash
# EXP-31 Cell F — B2 control (delayed_ef substrate, NO sub-basis) at data.seed=$BSEED.
# Pins the B2 BAND. delta_subbasis_rank=0 ⇒ bitwise B2 (the sub-basis branch is skipped),
# so it runs correctly on the current box checkout (exp/31). Carries disable_custom_all_reduce
# (controlled variable). seed 0 = Cell A (0.7400); this adds seeds 1,2 for the band.
set -euo pipefail
cd /workspace/verl
BSEED="${BSEED:-1}"
echo "=== B2 seed=$BSEED on $(git rev-parse --short HEAD) (delta_subbasis_rank=0 = bitwise B2) ==="
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef
export COMM_EFF_SPECTRAL_BETA_ANC=0.0
export COMM_EFF_SPECTRAL_EF_DECAY=0.0
export COMM_EFF_SPECTRAL_EF_CLIP=0.0
export COMM_EFF_SPECTRAL_BLEND_ETA=0.3
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
export PPO_MAX_TOKEN_LEN_PER_GPU=18432
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="exp31_F_b2_seed${BSEED}"
export LOG="/workspace/runs/EXP-31/train_F_b2_s${BSEED}.log"

mkdir -p /workspace/runs/EXP-31/metrics
ln -sf "$LOG" /workspace/train.log

RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_rank=0 \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false \
  data.seed="$BSEED" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true || RC=$?
echo "$(date -Iseconds) F_b2_seed${BSEED} done rc=$RC" > "/workspace/runs/EXP-31/done_F_b2_s${BSEED}.flag"
exit "$RC"
