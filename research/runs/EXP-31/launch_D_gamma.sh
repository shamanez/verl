#!/usr/bin/env bash
# EXP-31 Cell D γ-decay — the over-amplification fix: sub-basis weight γ decays
# linearly 1→0 over the run (strong early to keep the head-start, off late to
# avoid the regression that sank the constant-γ run). Plan/design: cellD_design.md.
#
# Ships exp_gamma.bundle (tip 73bf833cf = Cell D code + γ weight/decay knob + the
# proper actor.yaml struct declaration). The earlier copy of this script inherited
# launch_D.sh's `git fetch <bundle> branch:branch` checkout, which FAILS to update
# the currently-checked-out branch ref (error swallowed by `|| true`) → the box ran
# stale fe6234384 code → "delta_subbasis_weight not in struct". Fixed here: fetch to
# FETCH_HEAD, force a fresh branch, and HARD-ASSERT the SHA so a stale checkout dies
# loudly. No actor.yaml self-patch needed — the branch declares all sub-basis fields.
set -euo pipefail
cd /workspace/verl

BUNDLE=/workspace/runs/EXP-31/exp_gamma.bundle
EXPECT_SHA=73bf833cf6745fd78dca282a32eafb693082c005
[[ -f "$BUNDLE" ]] || { echo "FATAL: $BUNDLE missing" >&2; exit 3; }
git fetch "$BUNDLE" 'exp/31-subbasis-merger'                 # -> FETCH_HEAD (no ref conflict)
git checkout -f -B exp31-subbasis-gamma FETCH_HEAD           # fresh branch at the bundle tip
git reset --hard FETCH_HEAD
HEAD_SHA=$(git rev-parse HEAD)
echo "=== Cell D γ code: $HEAD_SHA (expect $EXPECT_SHA) ==="
[[ "$HEAD_SHA" == "$EXPECT_SHA"* ]] || { echo "FATAL: checkout SHA mismatch — γ code NOT applied (got $HEAD_SHA)" >&2; exit 4; }

python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1
# Fail fast if the γ knob is not in the config schema (wrong code / stale checkout).
python3 -c "from verl.workers.config.comm_eff import CommEffSpectralConfig as _C; assert hasattr(_C,'delta_subbasis_weight') and hasattr(_C,'delta_subbasis_decay_steps'), 'gamma knobs missing - wrong code'; print('gamma knobs present')"

# --- B2 substrate knobs (IDENTICAL to Cell A / launch_D.sh) ---
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
export EXPERIMENT_NAME=exp31_D_subbasis_gamma_decay50
export LOG=/workspace/runs/EXP-31/train_D_gamma.log

mkdir -p /workspace/runs/EXP-31/metrics
ln -sf "$LOG" /workspace/train.log

# Cell D γ-decay: rank-2 tail sub-basis, weight γ decaying 1→0 over 50 steps.
RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_rank=2 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_family=tail \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_weight=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_decay_steps=50 \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true || RC=$?
echo "$(date -Iseconds) D_gamma done rc=$RC" > /workspace/runs/EXP-31/done_D_gamma.flag
exit "$RC"
