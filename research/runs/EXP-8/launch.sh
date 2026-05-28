#!/usr/bin/env bash
# launch.sh — EXP-8 M2 anchor-circuit 3-cell, 5-step smoke (faithful / anchor-off / lean).
#
# Runs inside the Vast.ai container. The locked template's onstart has already
# cloned shamanez/verl @ vast-ai-workload into /workspace/verl and pip-installed
# it. EXP-8 is code_change=true, so we REPLACE that tree with the
# exp/8-anchor-circuit branch shipped in exp.bundle, then run three single-purpose
# integration cells on one box, sequentially.
#
# Modeled on EXP-7's proven multi-cell launcher
# (examples/grpo_trainer/vast_exp7_spectral_smoke.sh): each cell gets its OWN
# per-cell LOG and its OWN done_<cell>.flag, and a non-zero exit in one cell does
# NOT abort the chain (the baseline launcher hardcodes a fixed tee-log + done.flag
# path under set -euo pipefail, so we redirect the whole invocation per cell and
# guard with `|| echo`). All comm_eff overrides are prefixed
# actor_rollout_ref.actor.comm_eff.* (the schema lives at
# actor_rollout_ref.actor.comm_eff — EXP-7 proved this prefix; root-level keys die
# at OmegaConf struct-merge). use_orig_params=true + ENTROPY_COEFF=0.001 inherit
# the EXP-7 FSDP regime the anchor's per-target grad extraction needs.
set -uo pipefail   # NOT -e: a single cell failing must not kill the other cells.

RUN_DIR="/workspace/runs/EXP-8"
VERL_ROOT="/workspace/verl"
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/hotfix-patches"

# Configure git identity for any in-container commits (commit-hotfix.sh).
git config --global user.email "harness@verl-research.local" || true
git config --global user.name  "verl-research-harness" || true

# ---------------------------------------------------------------------------
# 0. Apply the experimental bundle (code_change=true) — exp/8-anchor-circuit.
# ---------------------------------------------------------------------------
SETUP_LOG="$RUN_DIR/setup.log"
{
  echo "=== EXP-8 setup $(date -u +%FT%TZ) ==="
  if [[ -f "$RUN_DIR/exp.bundle" ]]; then
    cd /workspace
    if [[ -d "$VERL_ROOT" ]]; then
      mv "$VERL_ROOT" /workspace/verl.upstream-vast-ai-workload
      echo "preserved template tree -> /workspace/verl.upstream-vast-ai-workload"
    fi
    git clone -b "exp/8-anchor-circuit" "$RUN_DIR/exp.bundle" "$VERL_ROOT"
    cd "$VERL_ROOT"
    # Point origin at the fork so any in-container push lands on shamanez/verl.
    git remote set-url origin https://github.com/shamanez/verl.git || true
    echo "=== installing exp/8 verl (--no-deps) ==="
    # Prefer uv when present, else pip. --no-deps so we only re-register the
    # editable package (the template already installed all heavy deps).
    if command -v uv >/dev/null 2>&1; then
      uv pip install --no-deps -e . 2>&1 | tail -8
    else
      pip install --no-deps -e . 2>&1 | tail -8
    fi
    echo "=== HEAD ==="; git log --oneline -1
    echo "=== anchor module present? ==="; test -f verl/workers/comm_eff/anchor.py && echo "anchor.py OK" || echo "anchor.py MISSING"
  else
    echo "WARN: $RUN_DIR/exp.bundle missing — running the template's vast-ai-workload tree (NO anchor code!)"
  fi
} > "$SETUP_LOG" 2>&1
cat "$SETUP_LOG"

cd "$VERL_ROOT"

# ---------------------------------------------------------------------------
# 1. Shared smoke env (5-step mandate; tiny shape; default logger console+wandb).
#    Mirrors the plan's ### Smoke launch commands env block.
# ---------------------------------------------------------------------------
export PROJECT_NAME="verl_compression_research"
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export ENTROPY_COEFF=0.001
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1

# Common Hydra overrides shared by all three cells (5 steps, no pre-train val,
# 1 PPO epoch, EXP-7 FSDP regime: use_orig_params=true).
COMMON_ARGS=(
  trainer.total_training_steps=5
  trainer.val_before_train=False
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true
)

# ---------------------------------------------------------------------------
# 2. Per-cell runner. $1=EXPERIMENT_NAME, remaining args = cell comm_eff overrides.
# ---------------------------------------------------------------------------
run_cell() {
  local cell="$1"; shift
  local log="$RUN_DIR/train_${cell}.log"
  echo "=== EXP-8 launching cell=$cell at $(date -u +%FT%TZ) -> $log ==="
  EXPERIMENT_NAME="$cell" \
  bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "$log" 2>&1 \
    || echo "[EXP-8] cell $cell exited non-zero ($?) — chain continues (see $log)"
  # Per-cell done flag (NOT the baseline launcher's hardcoded one).
  touch "$RUN_DIR/done_${cell}.flag"
  echo "=== EXP-8 cell=$cell finished at $(date -u +%FT%TZ) ==="
}

# ---- Cell 1 — faithful integrated M95+AP (headline) -----------------------
run_cell m2-anchor-faithful \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true actor_rollout_ref.actor.comm_eff.anchor.cadence=1 actor_rollout_ref.actor.comm_eff.anchor.delay_K=1

# ---- Cell 2 — anchor-OFF regression (reproduce EXP-7 seeded spectral) ------
run_cell m2-anchor-off \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=true \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false actor_rollout_ref.actor.comm_eff.anchor.cadence=1 actor_rollout_ref.actor.comm_eff.anchor.delay_K=1

# ---- Cell 3 — memory-lean storage (CPU EMA + low-rank SVD, all targets) ----
run_cell m2-anchor-lean \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu actor_rollout_ref.actor.comm_eff.spectral.svd_mode=lowrank actor_rollout_ref.actor.comm_eff.spectral.rank=8 actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true actor_rollout_ref.actor.comm_eff.anchor.cadence=1 actor_rollout_ref.actor.comm_eff.anchor.delay_K=1

echo "=== EXP-8 all cells complete at $(date -u +%FT%TZ) ==="
touch "$RUN_DIR/done.flag"
