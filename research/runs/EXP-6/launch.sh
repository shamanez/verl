#!/usr/bin/env bash
# EXP-6 — M2 mask contamination guard. Runs inside the Vast.ai container.
# The template onstart has already cloned shamanez/verl @ vast-ai-workload into
# /workspace/verl and pip-installed it. We replace that tree with the
# exp/6-mask-invariants branch from the shipped bundle, then run a two-step GRPO
# smoke twice: mask-ON then mask-OFF reference. val_before_train + TEST_FREQ=1 +
# SAVE_FREQ=1 force the validation and checkpoint paths to execute inside the
# 2-step window so the contamination invariants are observed live.
set -euo pipefail
RUN_DIR=/workspace/runs/EXP-6
cd "$RUN_DIR"

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# --- ensure /workspace/verl is on the exp/6 branch ------------------------- #
# The runner already `git fetch`+checked out exp/6-mask-invariants on this box
# (see experiment-runner step 4) and the template pip-installed verl editable,
# so an editable reinstall is unnecessary. Fall back to the shipped bundle only
# if the working tree is NOT already on the exp branch (e.g. fresh reprovision).
cd /workspace/verl
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
if [[ "$CUR_BRANCH" != "exp/6-mask-invariants" ]]; then
  echo "=== /workspace/verl on '$CUR_BRANCH', applying exp/6 bundle ==="
  if [[ -f "$RUN_DIR/exp.bundle" ]]; then
    cd /workspace
    [[ -d verl ]] && mv verl "verl.upstream-$(date +%s)"
    git clone -b exp/6-mask-invariants "$RUN_DIR/exp.bundle" verl
    cd /workspace/verl
    git remote set-url origin https://github.com/shamanez/verl.git || true
    uv pip install --no-deps -e . > /workspace/pip.log 2>&1
  else
    echo "FATAL: not on exp/6 branch and no bundle to apply" >&2; exit 1
  fi
fi
echo "=== verl @ $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) ==="

cd /workspace/verl

# --- CARRYOVER BUG WORKAROUND (EXP-4/EXP-5) -------------------------------- #
# vast_baseline_qwen25_1p5b_grpo_gsm8k.sh:196 does an unconditional
#   touch /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/done.flag
# at the END of each invocation. Under `set -euo pipefail` that touch aborts the
# whole chain if the parent dir does not exist, which kills a multi-cell run
# after the first cell. Pre-create the dir so the touch always succeeds and the
# mask-off reference cell still runs.
mkdir -p /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline

# --- shared smoke env ------------------------------------------------------ #
export PROJECT_NAME=verl_compression_research
export EXPERIMENT_NAME=m2-mask-invariants
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2
export MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=256
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export SAVE_FREQ=1 TEST_FREQ=1 TOTAL_EPOCHS=1

LAUNCHER=examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh

run_cell () {
  local name="$1"; shift
  local cell_log="$RUN_DIR/${name}.train.log"
  echo "=== EXP-6 cell ${name} start $(date -u +%FT%TZ) ==="
  # Per-cell LOG path so the two cells do not clobber each other's train.log.
  # `|| true` so a single-cell failure does not abort the chain under set -e;
  # the analyst reads per-cell logs and the per-path counters either way.
  LOG="$cell_log" bash "$LAUNCHER" \
    trainer.total_training_steps=2 \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    "$@" || echo "=== EXP-6 cell ${name} exited nonzero $(date -u +%FT%TZ) ==="
  echo "=== EXP-6 cell ${name} done $(date -u +%FT%TZ) ==="
}

# comm_eff is a field on the ACTOR config; the Hydra override path is
# actor_rollout_ref.actor.comm_eff.* — bare comm_eff.* fails the Hydra merge.
# cell 0 — masking ENABLED (the subject)
run_cell mask_on \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.mask.pp_size=8

# cell 1 — mask-off reference (dense-path regression + equality reference)
run_cell mask_off \
  actor_rollout_ref.actor.comm_eff.enabled=false

echo "$(date -Iseconds) done" > "$RUN_DIR/done.flag"
echo "=== EXP-6 all cells complete $(date -u +%FT%TZ) ==="
