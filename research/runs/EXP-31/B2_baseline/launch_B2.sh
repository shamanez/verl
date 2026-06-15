#!/usr/bin/env bash
# EXP-30 cell 2: B2_delayed_ef_valid_residual — GATE-B2 OPEN (stepA_gate.md:
# median m5_ratio 1.0528 in [0.1,1.5], max loss_mismatch 0.0103 <= 0.02).
# GATE-B1 CLOSED — B1 never launches. Plan 30.md sequence step 5 VERBATIM:
# Q3 — is the codec's weight-gradient error recoverable by a K-delayed exact
# residual? correction_mode=delayed_ef lambda=1.0 beta_anc=0.0 on the EXP-29
# replay substrate; PRODUCTION posture (probe OFF, capture OFF, tier-1 scalars
# + the one named per-fire scalar ||delta||/||G_comp|| from the
# [comm_eff][EXP-30][delayed_ef] line); 50 steps, test_freq=25 (val 0/25/50).
set -euo pipefail
cd /workspace/verl
# Checkout pinned to the exp branch tip by the runner (c56c13b); tolerate an
# offline origin by proceeding on the current (already-correct) checkout.
git fetch origin exp/30-valid-m-geometry 2>/dev/null && git reset --hard FETCH_HEAD >/dev/null 2>&1 || true
echo "=== B2 code: $(git log --oneline -1) ==="
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

# --- B2 cell knobs (merger hygiene: delayed_ef is the ONLY active correction;
#     ef_powersgd/signed_ema dead; blend NOT active; beta_anc=0; standing OOM
#     guards unchanged from Step A). Controlled-variables contract: vs Step A
#     the diff is confined to {correction_mode, probe flag, total_training_steps}
#     (+ the run name); lambda/blend_eta were already pinned in Step A.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # standing OOM guard
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef
export COMM_EFF_SPECTRAL_BETA_ANC=0.0
export COMM_EFF_SPECTRAL_EF_DECAY=0.0
export COMM_EFF_SPECTRAL_EF_CLIP=0.0
export COMM_EFF_SPECTRAL_BLEND_ETA=0.3                    # inert at delayed_ef; pinned == Step A
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu                   # standing OOM guard
export PPO_MAX_TOKEN_LEN_PER_GPU=18432                    # standing OOM guard
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25                                       # val at 0/25/50
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME=exp30_B2_delayed_ef
export LOG=/workspace/runs/EXP-30/train_B2_delayed_ef_valid_residual.log

mkdir -p /workspace/runs/EXP-30/metrics
# Liveness + sync-metrics contract: /workspace/train.log IS the live cell log.
ln -sf "$LOG" /workspace/train.log

RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false || RC=$?
echo "$(date -Iseconds) B2_delayed_ef_valid_residual done rc=$RC" > /workspace/runs/EXP-30/done_B2_delayed_ef_valid_residual.flag
exit "$RC"
