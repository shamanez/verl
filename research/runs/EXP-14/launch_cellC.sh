#!/usr/bin/env bash
# EXP-14 follow-up launcher — runs ONLY test2_cellC, the validation of the
# explosion root-cause fix (mask PRF substep counter never reset -> old_logprob
# and train forwards drew DIFFERENT masks at the same global step -> corrupted
# PPO ratio -> the test2_cellA step-1 grad_norm=771 explosion).
#
# The fix lives on branch exp/14-mask-consistent-rng (commit 49363ca4):
# the new knob comm_eff.mask.consistent_across_forwards (default true) holds the
# substep fixed so the mask is identical across all forwards of one global update.
#
# IMPORTANT: this script does NOT re-apply exp.bundle and does NOT re-clone or
# re-checkout. The box's /workspace/verl was advanced to exp/14-mask-consistent-rng
# via `git fetch origin && git checkout -f exp/14-mask-consistent-rng` (49363ca4 =
# c399d781 + the one fix commit, so clean_cadence is still present + YAML-mirrored).
# The editable -e install means the live tree's code/YAML is used directly.
# Re-cloning from the shipped bundle would REVERT the fix.
#
# Cell: test2_cellC = IDENTICAL to test2_cellA EXCEPT consistent_across_forwards=true.
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-14
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/logs" "$RUN_DIR/hotfix-patches"
CHAIN_LOG="$RUN_DIR/train.cellC.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=================================================================="
echo "[EXP-14] cellC chain start $(date -Iseconds)"
echo "[EXP-14] host: $(hostname)  gpus: $(nvidia-smi -L 2>/dev/null | wc -l)"
cd /workspace/verl
echo "[EXP-14] verl at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "[EXP-14] clean_cadence in YAML: $(grep -c '^  clean_cadence:' verl/trainer/config/actor/actor.yaml) (expect 1)"
echo "=================================================================="

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

# Hard-verify BOTH knobs resolve via Hydra BEFORE spending GPU time (fail fast).
python3 - <<'PYEOF' || { echo "[EXP-14] FATAL: required comm_eff Hydra overrides rejected — aborting cellC"; exit 1; }
from hydra import initialize_config_dir, compose
import os
with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
    cfg = compose(config_name="actor", overrides=[
        "comm_eff.clean_cadence=0",
        "comm_eff.mask.consistent_across_forwards=true",
    ])
    assert int(cfg.comm_eff.clean_cadence) == 0
    assert bool(cfg.comm_eff.mask.consistent_across_forwards) is True
print("[EXP-14] preflight OK: comm_eff.mask.consistent_across_forwards=true resolves")
PYEOF

export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"   # already preprocessed by the first run
COMM_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

run_cell() {
  local cell="$1"; shift
  local steps="$1"; shift
  local cell_log="$RUN_DIR/logs/${cell}.log"
  local cell_jsonl="$RUN_DIR/metrics/${cell}.jsonl"

  echo ""
  echo "------------------------------------------------------------------"
  echo "[EXP-14] CELL ${cell}  steps=${steps}  start $(date -Iseconds)"
  echo "[EXP-14]   metrics -> ${cell_jsonl}"
  echo "[EXP-14]   log     -> ${cell_log}"
  echo "------------------------------------------------------------------"

  export VERL_FILE_LOGGER_PATH="$cell_jsonl"
  export EXPERIMENT_NAME="exp14-${cell}"
  export PROJECT_NAME="verl_compression_research"

  export PPO_MINI_BATCH_SIZE=64
  export PPO_MAX_TOKEN_LEN_PER_GPU=36864
  export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
  export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
  export ROLLOUT_GPU_MEM_UTIL=0.4
  export TRAIN_BATCH_SIZE=128
  export ROLLOUT_N=8
  export MAX_PROMPT_LENGTH=1024
  export MAX_RESPONSE_LENGTH=16384
  export TOTAL_EPOCHS=2
  export VAL_BEFORE_TRAIN=True
  export TEST_FREQ=25
  export SAVE_FREQ=1000000
  export TOTAL_TRAINING_STEPS="$steps"
  export LOG="$cell_log"

  set +e
  bash "$COMM_LAUNCHER" \
    trainer.logger='["console","wandb","file"]' \
    trainer.total_training_steps="$steps" \
    trainer.default_local_dir="$RUN_DIR/ckpt/${cell}" \
    "$@"
  local rc=$?
  set +e
  echo "[EXP-14] CELL ${cell} exit rc=${rc} at $(date -Iseconds)"
  echo "$(date -Iseconds) ${cell} rc=${rc} steps=${steps}" > "$RUN_DIR/${cell}.done"
  return 0
}

NOKL=(actor_rollout_ref.actor.use_kl_loss=False
      algorithm.use_kl_in_reward=False
      actor_rollout_ref.actor.entropy_coeff=0)
USE_ORIG=(actor_rollout_ref.actor.fsdp_config.use_orig_params=true)

# =========================== TEST 2 Cell C (FIX VALIDATION) ===========================
# IDENTICAL to test2_cellA (mask-only recompute=true, clean_cadence=0), EXCEPT the
# new knob comm_eff.mask.consistent_across_forwards=true. If the substep-RNG fix is
# correct, step-1 grad_norm should collapse from ~771 toward the dense ~0.35 with a
# sane PPO ratio (old_logprob == train mask -> ratio ~ 1 at step 1).
run_cell test2_cellC 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.mask.consistent_across_forwards=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

echo ""
echo "=================================================================="
echo "[EXP-14] cellC chain COMPLETE $(date -Iseconds)"
echo "[EXP-14] cell: test2_cellC (mask-only + consistent_across_forwards=true)"
echo "=================================================================="
echo "$(date -Iseconds) cellC chain done" > "$RUN_DIR/done.cellC.flag"
