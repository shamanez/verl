#!/usr/bin/env bash
# EXP-14 RESUME launcher — runs ONLY the remaining cells after the corrective
# clean_cadence Hydra-schema fix (c399d781). Test 1 (test1_cellA, test1_cellB)
# PASSED 10/10 steps and is NOT re-run; its metrics/logs are preserved.
#
# IMPORTANT: this script does NOT re-apply exp.bundle. The box's /workspace/verl
# was advanced to c399d781 via `git pull origin exp/14-clean-cadence` (which has
# the actor.yaml clean_cadence declaration). Re-cloning from the shipped bundle
# (tip 5ac7f9ea, pre-YAML-fix) would REVERT the fix and re-trigger the crash.
# The editable -e install means the live tree's YAML is used directly.
#
# Cells (all 10 steps, sequential, shared docker/verl/dataset):
#   test2_cellA (mask-only recompute=true, clean_cadence=0)
#   test2_cellB (mask-only recompute=false, clean_cadence=0)
#   test3_cellA REUSES test2_cellA (jsonl copy, no re-run)
#   test3_cellB (mask + clean_cadence=10; single clean step @ step 10)
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-14
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/logs" "$RUN_DIR/hotfix-patches"
CHAIN_LOG="$RUN_DIR/train.resume.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=================================================================="
echo "[EXP-14] RESUME chain start $(date -Iseconds)"
echo "[EXP-14] host: $(hostname)  gpus: $(nvidia-smi -L 2>/dev/null | wc -l)"
cd /workspace/verl
echo "[EXP-14] verl at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "[EXP-14] clean_cadence in YAML: $(grep -c '^  clean_cadence:' verl/trainer/config/actor/actor.yaml) (expect 1)"
echo "=================================================================="

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

# Hard-verify the Hydra override resolves BEFORE spending GPU time (fail fast if the
# YAML fix somehow isn't live).
python3 - <<'PYEOF' || { echo "[EXP-14] FATAL: clean_cadence Hydra override still rejected — aborting resume"; exit 1; }
from hydra import initialize_config_dir, compose
import os
with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
    cfg = compose(config_name="actor", overrides=["comm_eff.clean_cadence=10"])
    assert int(cfg.comm_eff.clean_cadence) == 10
print("[EXP-14] preflight OK: comm_eff.clean_cadence=10 resolves")
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

# =========================== TEST 2 (PEEL, diagnosis) ===========================
# Pure masked GRPO: mask on (p=0.9), anchor+spectral OFF (not allocated), clean_cadence=0.
run_cell test2_cellA 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

run_cell test2_cellB 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=false \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

# =========================== TEST 3 (FIX, mandatory headline) ===========================
echo ""
echo "[EXP-14] TEST 3 Cell A: reusing Test 2 Cell A (identical config) — copying artifacts"
[[ -f "$RUN_DIR/metrics/test2_cellA.jsonl" ]] && cp -f "$RUN_DIR/metrics/test2_cellA.jsonl" "$RUN_DIR/metrics/test3_cellA.jsonl"
[[ -f "$RUN_DIR/logs/test2_cellA.log" ]] && cp -f "$RUN_DIR/logs/test2_cellA.log" "$RUN_DIR/logs/test3_cellA.log"
echo "$(date -Iseconds) test3_cellA reuse_of=test2_cellA" > "$RUN_DIR/test3_cellA.done"

# Cell B: mask-only + clean_cadence=10. At 10 steps the clean step fires once (step 10).
run_cell test3_cellB 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=10 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

echo ""
echo "=================================================================="
echo "[EXP-14] RESUME chain COMPLETE $(date -Iseconds)"
echo "[EXP-14] cells: test2_cellA test2_cellB test3_cellB (test3_cellA reused test2_cellA)"
echo "[EXP-14] Test 1 was preserved from the prior pass; Test 4 conditional, not run."
echo "=================================================================="
echo "$(date -Iseconds) resume chain done" > "$RUN_DIR/done.resume.flag"
