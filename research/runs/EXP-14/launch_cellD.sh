#!/usr/bin/env bash
# EXP-14 follow-up launcher — runs ONLY test2_cellD, the validation of the
# BIAS (magnitude-collapse) hypothesis.
#
# Context: test2_cellC (consistent_across_forwards=true) did NOT fix the step-1
# grad_norm explosion (gn=881.9 vs cellA 771.4) -> the IS/substep inconsistency
# angle was refuted. New diagnosis: the mask drops ~90% of boundary activations
# with NO rescale -> boundary hidden-state RMS collapses to ~0.32x -> large OOD
# shift -> huge gradients. Fix: comm_eff.mask.rescale (inverted-dropout
# h*mask/(1-p), preserves E[h]).
#
# The fix lives on branch exp/14-mask-pertoken-rescale (commit 905f4742):
# clean_cadence + consistent_across_forwards + the new rescale knob.
#
# IMPORTANT: this script does NOT re-apply exp.bundle and does NOT re-clone.
# The box's /workspace/verl was advanced to exp/14-mask-pertoken-rescale via
# `git fetch origin && git checkout -f exp/14-mask-pertoken-rescale` (905f4742).
# The editable -e install means the live tree's code/YAML is used directly.
# Re-cloning from the shipped bundle would REVERT the fix.
#
# Cell: test2_cellD = IDENTICAL to test2_cellC EXCEPT mask.rescale=true.
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-14
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/logs" "$RUN_DIR/hotfix-patches"
CHAIN_LOG="$RUN_DIR/train.cellD.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=================================================================="
echo "[EXP-14] cellD chain start $(date -Iseconds)"
echo "[EXP-14] host: $(hostname)  gpus: $(nvidia-smi -L 2>/dev/null | wc -l)"
cd /workspace/verl
echo "[EXP-14] verl at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "[EXP-14] clean_cadence in YAML: $(grep -c '^  clean_cadence:' verl/trainer/config/actor/actor.yaml) (expect 1)"
echo "[EXP-14] rescale in YAML: $(grep -c '^    rescale:' verl/trainer/config/actor/actor.yaml) (expect 1)"
echo "=================================================================="

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

# Hard-verify ALL THREE knobs resolve via Hydra BEFORE spending GPU time (fail fast).
python3 - <<'PYEOF' || { echo "[EXP-14] FATAL: required comm_eff Hydra overrides rejected — aborting cellD"; exit 1; }
from hydra import initialize_config_dir, compose
import os
with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
    cfg = compose(config_name="actor", overrides=[
        "comm_eff.clean_cadence=0",
        "comm_eff.mask.consistent_across_forwards=true",
        "comm_eff.mask.rescale=true",
    ])
    assert int(cfg.comm_eff.clean_cadence) == 0
    assert bool(cfg.comm_eff.mask.consistent_across_forwards) is True
    assert bool(cfg.comm_eff.mask.rescale) is True
print("[EXP-14] preflight OK: mask.rescale=true + consistent_across_forwards=true + clean_cadence=0 resolve")
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

# =========================== TEST 2 Cell D (BIAS FIX VALIDATION) ===========================
# IDENTICAL to test2_cellC (mask-only recompute=true, clean_cadence=0,
# consistent_across_forwards=true), EXCEPT the new knob comm_eff.mask.rescale=true.
# If the magnitude-collapse hypothesis is correct, inverted-dropout rescaling
# preserves E[h] across the masked boundary -> step-1 grad_norm should collapse
# from ~771 (cellA) / ~882 (cellC) toward the dense ~0.35.
run_cell test2_cellD 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.mask.consistent_across_forwards=true \
  actor_rollout_ref.actor.comm_eff.mask.rescale=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

echo ""
echo "=================================================================="
echo "[EXP-14] cellD chain COMPLETE $(date -Iseconds)"
echo "[EXP-14] cell: test2_cellD (mask-only + consistent_across_forwards=true + rescale=true)"
echo "=================================================================="
echo "$(date -Iseconds) cellD chain done" > "$RUN_DIR/done.cellD.flag"
