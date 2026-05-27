#!/usr/bin/env bash
# EXP-5 launcher — runs inside the Vast.ai container.
#
# The template onstart cloned shamanez/verl @ vast-ai-workload into
# /workspace/verl and pip-installed it. For this code_change=true experiment we
# replace that tree with the exp/5-actor-mask branch from the shipped bundle so
# the masking code is on the box (and survives even if the laptop dies).
#
# Three smoke cells, back-to-back, on ONE instance:
#   m2-actor-mask-p95       comm_eff.enabled=true  comm_eff.mask.p=0.95
#   m2-actor-mask-p90       comm_eff.enabled=true  comm_eff.mask.p=0.90
#   m2-actor-mask-disabled  comm_eff.enabled=false  (EXP-4 no-op contract regression)
#
# Each cell's train.log lands at runs/EXP-5/metrics/<EXPERIMENT_NAME>/train.log
# so the analyst's greps resolve.
set -euo pipefail

RUN_DIR=/workspace/runs/EXP-5
METRICS_DIR="$RUN_DIR/metrics"
mkdir -p "$METRICS_DIR"

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---------------------------------------------------------------------------
# Apply the experimental bundle (exp/5-actor-mask) onto /workspace/verl.
# ---------------------------------------------------------------------------
cd /workspace
if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload
  git clone -b exp/5-actor-mask "$RUN_DIR/exp.bundle" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.log 2>&1
fi

# Confirm we are on the exp branch before launch.
cd /workspace/verl
BR=$(git rev-parse --abbrev-ref HEAD)
echo "=== /workspace/verl on branch: $BR ==="
if [[ "$BR" != "exp/5-actor-mask" ]]; then
  echo "FATAL: expected exp/5-actor-mask, got $BR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Fixed smoke shape (EXP-5 plan, verbatim) shared by all three cells.
# ---------------------------------------------------------------------------
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_TP=1
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1
export PROJECT_NAME=verl_compression_research

# Smoke token budgets (plan: 4096 each).
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096

# Extra Hydra overrides appended after the baseline launcher's arrays (so they
# win): 2 trainer steps, no pre-train validation, single ppo epoch, fixed seed.
SMOKE_HYDRA=(
  trainer.total_training_steps=2
  trainer.val_before_train=False
  actor_rollout_ref.actor.ppo_epochs=1
  data.shuffle=False
)

# Pin a deterministic seed so the masking determinism claims hold across cells.
BASE_SEED=1234
SMOKE_HYDRA+=( "actor_rollout_ref.actor.comm_eff.mask.seed=${BASE_SEED}" )

run_cell () {
  local name="$1"; shift
  local celldir="$METRICS_DIR/$name"
  mkdir -p "$celldir"
  echo "=================================================================="
  echo "=== EXP-5 cell: $name ==="
  echo "=================================================================="
  # The baseline launcher writes its own tee'd log to $LOG; point it at the
  # cell's train.log so the analyst's path resolves.
  EXPERIMENT_NAME="$name" \
  LOG="$celldir/train.log" \
    bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
      "${SMOKE_HYDRA[@]}" \
      "$@"
  echo "=== cell $name finished at $(date -Iseconds) ==="
}

# ---------------------------------------------------------------------------
# Cell 1: p=0.95 (masking active)
# ---------------------------------------------------------------------------
run_cell m2-actor-mask-p95 \
  actor_rollout_ref.actor.comm_eff.enabled=True \
  actor_rollout_ref.actor.comm_eff.mask.enabled=True \
  actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.mask.pp_size=8

# ---------------------------------------------------------------------------
# Cell 2: p=0.90 (masking active)
# ---------------------------------------------------------------------------
run_cell m2-actor-mask-p90 \
  actor_rollout_ref.actor.comm_eff.enabled=True \
  actor_rollout_ref.actor.comm_eff.mask.enabled=True \
  actor_rollout_ref.actor.comm_eff.mask.p=0.90 \
  actor_rollout_ref.actor.comm_eff.mask.pp_size=8

# ---------------------------------------------------------------------------
# Cell 3: disabled (EXP-4 no-op contract regression)
# ---------------------------------------------------------------------------
run_cell m2-actor-mask-disabled \
  actor_rollout_ref.actor.comm_eff.enabled=False

echo "$(date -Iseconds) all cells done" > "$RUN_DIR/done.flag"
echo "=== EXP-5 all three cells complete ==="
