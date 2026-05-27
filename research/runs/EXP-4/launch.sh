#!/usr/bin/env bash
# EXP-4 launch.sh — M2 comm_eff no-op scaffolding parity smoke.
#
# Runs inside the Vast.ai container. The verl-research-vllm020 template's onstart
# has already cloned shamanez/verl @ vast-ai-workload into /workspace/verl and
# pip-installed it --no-deps. This script:
#   1. Replaces /workspace/verl with the exp/4-commeff-noop branch (from the shipped
#      bundle), reinstalls --no-deps, and CONFIRMS the branch before Runs A/B.
#   2. Runs A (explicit comm_eff.enabled=false) and B (no override) on the exp branch.
#   3. Restores the unmodified vast-ai-workload tree (NO comm_eff code on the path)
#      and runs the reference smoke at the SAME seed for the criterion-7 rel-tol check.
# All three use the SAME fixed smoke shape and seed so the disabled-path actor
# scalars are comparable to the unmodified-launcher reference.
set -euo pipefail

RUN_DIR=/workspace/runs/EXP-4
LAUNCHER=examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh
EXP_BRANCH="exp/4-commeff-noop"
BASE_BRANCH="vast-ai-workload"

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"
git config --global --add safe.directory /workspace/verl || true

mkdir -p "$RUN_DIR/metrics"

# ---------------------------------------------------------------------------
# Fixed smoke shape (plan ## Experiment design smoke_shape). Exported so the
# baseline launcher picks them up; the launcher passes its positional "$@" to
# the recipe, and OUR extra positional args (appended below) land last and win.
# ---------------------------------------------------------------------------
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1
export PROJECT_NAME=verl_compression_research
# A modest TP that divides both 4- and 8-GPU boxes.
export ROLLOUT_TP=2

SMOKE_SEED=1

# Smoke-specific Hydra overrides not exposed as launcher env vars. These come
# LAST so they override the recipe + launcher positional args. We pin data.seed
# identically across all three runs; the vLLM rollout seed defaults to 0 in every
# run (config.get("seed",0)), so the rollout RNG is identical by construction —
# we deliberately do NOT add +rollout.seed (unregistered key; the structured
# RolloutConfig would reject it).
SMOKE_HYDRA=(
  trainer.total_training_steps=2
  trainer.val_before_train=False
  actor_rollout_ref.actor.ppo_epochs=1
  data.seed=${SMOKE_SEED}
  data.shuffle=False
)

run_smoke() {
  # $1 = EXPERIMENT_NAME (also the metrics subdir), $2.. = extra Hydra overrides
  local name="$1"; shift
  local extra=("$@")
  local metrics_subdir="$RUN_DIR/metrics/$name"
  mkdir -p "$metrics_subdir"
  echo "================================================================"
  echo "=== EXP-4 run: $name"
  echo "=== verl branch on box: $(git -C /workspace/verl rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "=== extra hydra: ${extra[*]:-<none>}"
  echo "================================================================"
  # The launcher writes its own LOG via tee; we point it at the per-run subdir
  # AND mirror to the top-level train.log so the liveness tail + sync-metrics see it.
  export EXPERIMENT_NAME="$name"
  export LOG="$metrics_subdir/train.log"
  set +e
  bash "$LAUNCHER" "${SMOKE_HYDRA[@]}" "${extra[@]}" 2>&1 | tee -a /workspace/train.log
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== EXP-4 run $name finished rc=$rc ===" | tee -a /workspace/train.log
  echo "$(date -Iseconds) $name rc=$rc" >> "$RUN_DIR/run-status.log"
  return $rc
}

# ---------------------------------------------------------------------------
# 0. Swap /workspace/verl to the exp/4-commeff-noop branch from the bundle.
# ---------------------------------------------------------------------------
cd /workspace
if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  echo "=== applying exp.bundle -> $EXP_BRANCH ==="
  # Preserve the template-installed upstream tree so we can restore it for the reference run.
  if [[ -d verl && ! -d verl.upstream-vast-ai-workload ]]; then
    cp -a verl verl.upstream-vast-ai-workload
  fi
  cd /workspace/verl
  git config --global --add safe.directory /workspace/verl || true
  # Fetch the exp branch from the shipped bundle and check it out.
  git fetch "$RUN_DIR/exp.bundle" "$EXP_BRANCH:$EXP_BRANCH" 2>&1 | tail -3 || \
    git fetch "$RUN_DIR/exp.bundle" "+refs/heads/$EXP_BRANCH:refs/heads/$EXP_BRANCH" 2>&1 | tail -3
  git checkout "$EXP_BRANCH"
  git remote set-url origin https://github.com/shamanez/verl.git || true
  echo "=== reinstall verl (--no-deps) from $EXP_BRANCH ==="
  uv pip install --no-deps -e . > /workspace/pip.exp.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.exp.log 2>&1
else
  echo "FATAL: $RUN_DIR/exp.bundle missing — cannot apply comm_eff scaffolding." >&2
  exit 1
fi

# Hard gate: Runs A/B MUST execute on the exp branch (plan ## Notes for runner).
ACTUAL_BRANCH="$(git -C /workspace/verl rev-parse --abbrev-ref HEAD)"
if [[ "$ACTUAL_BRANCH" != "$EXP_BRANCH" ]]; then
  echo "FATAL: expected /workspace/verl on $EXP_BRANCH, got '$ACTUAL_BRANCH'. Aborting." >&2
  exit 1
fi
echo "=== confirmed /workspace/verl on $EXP_BRANCH ==="

cd /workspace/verl

# ---------------------------------------------------------------------------
# Run A — explicit comm_eff.enabled=false (registered plain key on the exp branch).
# ---------------------------------------------------------------------------
run_smoke "m2-commeff-noop-disabled" \
  actor_rollout_ref.actor.comm_eff.enabled=false

# ---------------------------------------------------------------------------
# Run B — NO comm_eff override (relies on the registered config default false).
# ---------------------------------------------------------------------------
run_smoke "m2-commeff-noop-default"

# ---------------------------------------------------------------------------
# Reference — unmodified vast-ai-workload, NO comm_eff code on the path, same
# seed, for the criterion-7 rel-tol 1e-4 parity check. Restore the template tree.
# ---------------------------------------------------------------------------
echo "=== restoring unmodified $BASE_BRANCH tree for the reference run ==="
cd /workspace
if [[ -d verl.upstream-vast-ai-workload ]]; then
  rm -rf verl.exp-4-commeff-noop && mv verl verl.exp-4-commeff-noop
  mv verl.upstream-vast-ai-workload verl
  cd /workspace/verl
  uv pip install --no-deps -e . > /workspace/pip.ref.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.ref.log 2>&1
else
  # Fallback: checkout the base branch in place (bundle had no preserved copy).
  cd /workspace/verl
  git fetch origin "$BASE_BRANCH" 2>&1 | tail -2 || true
  git checkout "$BASE_BRANCH" 2>&1 | tail -2
  uv pip install --no-deps -e . > /workspace/pip.ref.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.ref.log 2>&1
fi
REF_BRANCH="$(git -C /workspace/verl rev-parse --abbrev-ref HEAD)"
echo "=== reference run on branch: $REF_BRANCH (expect $BASE_BRANCH, no comm_eff code) ==="
cd /workspace/verl
run_smoke "m2-commeff-noop-reference"

echo "$(date -Iseconds) all-runs-done" > "$RUN_DIR/done.flag"
echo "=== EXP-4 all three smokes complete at $(date -u +%FT%TZ) ==="
