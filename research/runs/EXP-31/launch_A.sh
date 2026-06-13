#!/usr/bin/env bash
# EXP-31 Cell A — bitwise-B2-reproduce + substrate control (NO code change).
# Plan: research/.claude/plans/31.md §"Experiment sequence" Cell A.
#
# This re-establishes the EXP-30 B2 result (val@50 ~ 0.7528) on the box's
# CURRENT vast-ai-workload checkout (commit 7d132ca -- already contains the
# delayed_ef + replay substrate). It is the live substrate control that every
# code-change cell's OFF path (Cell D / Cell C) must reproduce bitwise.
#
# Differences vs EXP-30/launch_B2.sh (by design):
#   (a) NO git-fetch / no exp/* checkout -- stay on the box's vast-ai-workload
#       checkout (this cell is no-code-change; Cell D code is a separate effort).
#   (b) run dir = /workspace/runs/EXP-31 (was /workspace/runs/EXP-30).
#   (c) EXPERIMENT_NAME=exp31_A_b2_reproduce; LOG=/workspace/runs/EXP-31/train_A.log.
# Everything else (the B2 knob set) is IDENTICAL to launch_B2.sh /
# resolved_params_B2.txt -- the controlled-variable contract for the whole plan.
set -euo pipefail
cd /workspace/verl
echo "=== Cell A code: $(git log --oneline -1) (branch $(git rev-parse --abbrev-ref HEAD)) ==="
# verl is already pip-installed editable on this box; import-guard is a fast no-op.
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

# --- B2 substrate knobs (EXACT match to runs/EXP-30/resolved_params_B2.txt) ---
#     delayed_ef is the ONLY active correction; ef_powersgd/signed_ema dead;
#     blend inert at delayed_ef; beta_anc=0; standing OOM guards on.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # standing OOM guard
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef
export COMM_EFF_SPECTRAL_BETA_ANC=0.0
export COMM_EFF_SPECTRAL_EF_DECAY=0.0
export COMM_EFF_SPECTRAL_EF_CLIP=0.0
export COMM_EFF_SPECTRAL_BLEND_ETA=0.3                    # inert at delayed_ef; pinned == B2
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu                   # standing OOM guard
export PPO_MAX_TOKEN_LEN_PER_GPU=18432                    # standing OOM guard
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25                                       # val at 0/25/50
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME=exp31_A_b2_reproduce
export LOG=/workspace/runs/EXP-31/train_A.log

mkdir -p /workspace/runs/EXP-31/metrics
# Liveness + sync-metrics contract: /workspace/train.log IS the live cell log.
ln -sf "$LOG" /workspace/train.log

RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false || RC=$?
echo "$(date -Iseconds) A done rc=$RC" > /workspace/runs/EXP-31/done_A.flag
exit "$RC"
