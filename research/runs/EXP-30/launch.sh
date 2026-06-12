#!/usr/bin/env bash
# EXP-30 cell 0: stepA_geometry_probe — runs inside the Vast.ai container.
# Operator-provided box 40697545 (4xH200): /workspace/verl pre-exists on
# vast-ai-workload; this script checks out the pushed exp/30-valid-m-geometry
# branch (bundle fallback) and launches ONLY Step A via the canonical
# comm-eff launcher + env/Hydra overrides (VAST_README stability contract).
# B1/B2 are GATED on the Step-A geometry gates — NOT launched here.
set -euo pipefail
cd /workspace/runs/EXP-30

# Git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# --- code: exp/30-valid-m-geometry (pushed pre-launch; bundle = offline fallback) ---
cd /workspace/verl
git fetch origin exp/30-valid-m-geometry 2>/dev/null \
  || git fetch /workspace/runs/EXP-30/exp.bundle exp/30-valid-m-geometry:refs/remotes/bundle/exp-30 \
  || { echo "FATAL: cannot fetch exp/30-valid-m-geometry from origin OR the bundle" >&2; exit 1; }
git checkout exp/30-valid-m-geometry 2>/dev/null \
  || git checkout -b exp/30-valid-m-geometry FETCH_HEAD
git log --oneline -1
# verl is editable-installed by the template onstart; a branch checkout is
# live immediately. Defensive: install only if the import is missing.
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

# --- Step-A cell knobs (plan 30.md sequence step 2; merger hygiene pinned) ---
# Correction INERT: mode=none (anchor_grad_corrected==0 the whole run);
# retired ef_powersgd/signed_ema knobs pinned DEAD; beta_anc=0 (M_rep = latest
# paired G_anc_rep); B-cell weights (blend_eta=0.3 / delayed_ef_lambda=1.0)
# pinned now so a gated B cell differs ONLY in correction_mode + probe flag +
# total_training_steps (the controlled-variables assert).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # standing OOM guard
export COMM_EFF_SPECTRAL_CORRECTION_MODE=none
export COMM_EFF_SPECTRAL_BETA_ANC=0.0
export COMM_EFF_SPECTRAL_EF_DECAY=0.0
export COMM_EFF_SPECTRAL_EF_CLIP=0.0
export COMM_EFF_SPECTRAL_BLEND_ETA=0.3
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu                   # standing OOM guard
export PPO_MAX_TOKEN_LEN_PER_GPU=18432                    # standing OOM guard
export TOTAL_TRAINING_STEPS=20                            # 40 ticks = 8 fires (7 post-warmup)
export TEST_FREQ=25                                       # val only at step 0
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME=exp30_stepA_geometry_probe
export LOG=/workspace/runs/EXP-30/train_stepA_geometry_probe.log

mkdir -p /workspace/runs/EXP-30/metrics
# Liveness + sync-metrics contract: /workspace/train.log IS the live cell log.
ln -sf "$LOG" /workspace/train.log

# The launcher runs the verbatim main_ppo under set -x — train.log's expanded
# command is the ground truth resolved_params.txt is extracted from.
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.out_dir=/workspace/runs/EXP-30/metrics \
  actor_rollout_ref.actor.comm_eff.probe.rank0_only=true \
  actor_rollout_ref.actor.comm_eff.probe.m4_lags=5 \
  actor_rollout_ref.actor.comm_eff.probe.per_target_sidecar=true || RC=$?
RC=${RC:-0}
echo "$(date -Iseconds) stepA_geometry_probe done rc=$RC" > /workspace/runs/EXP-30/done.flag
exit "$RC"
