#!/usr/bin/env bash
# EXP-20 launch driver — runs INSIDE the Vast.ai container.
# The verl-research-vllm020 template onstart already cloned shamanez/verl @
# vast-ai-workload into /workspace/verl and pip-installed it. We replace that
# tree with the exp/20-powersgd-activation branch from the shipped bundle, then
# run the sequence back-to-back on this ONE box:
#   step 0 = PowerSGD correctness probe (2 steps; HARD GATE for the FSDP/dtype
#            integration + frozen-Q rho~1 — the CPU math invariants are already
#            proven by the in-repo tests).
#   step 1 = PRF mask p=0.95 + clean_cadence=5, 50 steps (THE bar, Run B).
#   step 2 = PowerSGD r=102 + clean_cadence=5, 50 steps (the candidate, Run A).
#   step 3 = OPTIONAL dense ceiling, 50 steps (gates nothing; budget-permitting).
#
# Both 50-step arms call the SAME canonical launcher with only codec knobs
# differing (stability contract). If the probe surfaces NaN/OOM/non-finite
# q_cond, this script STOPS before the sweep (writes PROBE_FAILED) so the
# operator/runner fixes on the exp/* branch and re-probes rather than paying
# for an uninterpretable 50-step run.
set -euo pipefail

RUN_ID=EXP-20
RUN_DIR=/workspace/runs/${RUN_ID}
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
SLUG=20-powersgd-activation
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/hotfix-patches"

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---------------------------------------------------------------------------
# Apply the experimental bundle (exp/20-powersgd-activation).
# ---------------------------------------------------------------------------
cd /workspace
if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  if [[ ! -d /workspace/verl/.git ]] || ! git -C /workspace/verl rev-parse --verify "exp/${SLUG}" >/dev/null 2>&1; then
    [[ -d verl ]] && mv verl "verl.upstream-vast-ai-workload.$(date +%s)" || true
    git clone -b "exp/${SLUG}" "$RUN_DIR/exp.bundle" verl
    cd /workspace/verl
    git remote set-url origin https://github.com/shamanez/verl.git || true
    uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || \
      pip install --no-deps -e . > /workspace/pip.log 2>&1
  fi
fi
cd /workspace/verl
echo "=== verl @ $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) ==="

# Quick in-container import sanity (catches a broken install before GPU spend).
python3 -c "from verl.workers.comm_eff.powersgd_activation import PowerSGDActivationCompressor; \
from verl.workers.config import CommEffPowerSGDConfig; print('[EXP-20] powersgd codec import OK')"

# Common knobs shared by every step (Vast multi-GPU detect happens inside the launcher).
export PROJECT_NAME=verl_compression_research
export WANDB_ENTITY="${WANDB_ENTITY:-shamanework-pl}"
# anchor + spectral OFF for the whole experiment.
export COMM_EFF_ANCHOR_ENABLED=false
export COMM_EFF_SPECTRAL_ENABLED=false
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}"

run_step () {  # $1=experiment_name ; rest = extra env already exported by caller
  local name="$1"; shift || true
  export EXPERIMENT_NAME="$name"
  export LOG="$RUN_DIR/${name}.log"
  echo "=== [EXP-20] launching $name (total_steps=$TOTAL_TRAINING_STEPS, compression_type=${COMM_EFF_COMPRESSION_TYPE:-dense}) ==="
  bash "$LAUNCHER" "$@" 2>&1 | tee -a /workspace/train.log
}

# ===========================================================================
# STEP 0 — PowerSGD correctness probe (HARD GATE). 2 steps, pure-compressed.
# ===========================================================================
echo "===================== STEP 0: powersgd correctness probe ====================="
export TOTAL_TRAINING_STEPS=2
export VAL_BEFORE_TRAIN=False           # skip the pre-train eval for the cheap probe
export SAVE_FREQ=1000 TEST_FREQ=1000
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=102
export COMM_EFF_CLEAN_CADENCE=0
run_step ce_powersgd_probe_2s_gsm8k || true

# r=H lossless sub-check (M_hat == M => reconstruction_rel_error <= 1e-4).
export COMM_EFF_POWERSGD_RANK=2048
run_step ce_powersgd_probe_rankH_2s_gsm8k || true
export COMM_EFF_POWERSGD_RANK=102

# off-path parity sub-check (dense; the train/old-logprob forward byte-identical).
export COMM_EFF_ENABLED=false
export COMM_EFF_COMPRESSION_TYPE=dense
run_step ce_dense_probe_2s_gsm8k || true

# --- Probe verdict: scan the probe logs for the hard-gate falsifiers. ---
PROBE_LOG="$RUN_DIR/ce_powersgd_probe_2s_gsm8k.log"
RANKH_LOG="$RUN_DIR/ce_powersgd_probe_rankH_2s_gsm8k.log"
PROBE_BAD=0
# Numeric blow-up / OOM / single-GPU collapse / FSDP-hook break = hard fail.
FAIL_RE='([Nn]a[Nn] detected|CUDA out of memory|OutOfMemoryError|q_cond[^A-Za-z].{0,40}(nan|inf|Inf)|single-GPU fallback|world_size=1|RuntimeError: .*use_orig_params|summon_full_params.*(error|Error|assert))'
for L in "$PROBE_LOG" "$RANKH_LOG"; do
  [[ -f "$L" ]] || { echo "PROBE_FAILED: missing $L (step 0 did not run)"; PROBE_BAD=1; continue; }
  if grep -nE "$FAIL_RE" "$L" >/dev/null 2>&1; then
    echo "PROBE_FAILED: hard-gate falsifier in $(basename "$L"):"
    grep -nE "$FAIL_RE" "$L" | head -3
    PROBE_BAD=1
  fi
  # The run must reach >=1 optimizer step (a step:1 metric line / grad_norm).
  if ! grep -qE "(step:1|'global_step': 1|grad_norm)" "$L" 2>/dev/null; then
    echo "PROBE_FAILED: $(basename "$L") never reached a training step"
    PROBE_BAD=1
  fi
done

if [[ "$PROBE_BAD" -ne 0 ]]; then
  echo "===== EXP-20 PROBE_FAILED — NOT proceeding to the 50-step sweep. ====="
  echo "Fix on exp/20-powersgd-activation (commit-hotfix.sh), re-run this script." | tee -a /workspace/train.log
  printf 'EXP-20\tPROBE_FAILED\n' > "$RUN_DIR/PROBE_FAILED"
  exit 3
fi
echo "===== EXP-20 PROBE PASSED — proceeding to the 50-step sweep. =====" | tee -a /workspace/train.log
printf 'EXP-20\tPROBE_PASSED\n' > "$RUN_DIR/PROBE_PASSED"

# Restore the full-run schedule for the 50-step arms.
export TOTAL_TRAINING_STEPS=50
export VAL_BEFORE_TRAIN=True
export SAVE_FREQ=50 TEST_FREQ=25

# ===========================================================================
# STEP 1 — PRF mask p=0.95 + clean_cadence=5, 50 steps (Run B, THE bar).
# ===========================================================================
echo "===================== STEP 1: mask p=0.95 + clean@5 (50 steps) ====================="
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=prf_mask
export COMM_EFF_MASK_ENABLED=true
export COMM_EFF_MASK_P=0.95
export COMM_EFF_MASK_RESCALE=true
export COMM_EFF_MASK_RECOMPUTE=true
export COMM_EFF_CLEAN_CADENCE=5
run_step ce_mask_p95_clean5_50s_gsm8k

# ===========================================================================
# STEP 2 — PowerSGD r=102 + clean_cadence=5, 50 steps (Run A, the candidate).
# ===========================================================================
echo "===================== STEP 2: powersgd r=102 + clean@5 (50 steps) ====================="
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=102
export COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true
export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_SYNC_BASIS=false
export COMM_EFF_POWERSGD_QR_DTYPE=fp32
export COMM_EFF_CLEAN_CADENCE=5
run_step ce_powersgd_r102_clean5_50s_gsm8k

# Mark the REQUIRED science captured before the optional ceiling.
echo "$(date -Iseconds) required-arms-done" > "$RUN_DIR/done.flag"

# ===========================================================================
# STEP 3 — OPTIONAL dense ceiling, 50 steps (gates nothing). Best-effort.
# ===========================================================================
echo "===================== STEP 3 (optional): dense ceiling (50 steps) ====================="
export COMM_EFF_ENABLED=false
export COMM_EFF_COMPRESSION_TYPE=dense
run_step ce_dense_50s_gsm8k || echo "STEP 3 dense ceiling skipped/failed (optional — gates nothing)"

echo "$(date -Iseconds) all-done" > "$RUN_DIR/done.flag"
echo "=== EXP-20 sequence complete at $(date -u +%FT%TZ) ==="
