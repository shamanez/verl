#!/usr/bin/env bash
# EXP-23 smoke gate (step 0). Runs INSIDE the Vast.ai container.
# Two <=2-step probes prove the anchor+spectral circuit composes with the
# PowerSGD codec AND the launcher wires correction_mode/delay_K/blend_eta.
#   PROBE_ON  : powersgd r=77 + anchor(delay_K=5,cadence=5) + spectral(inject,gamma=1,cadence=5),
#               clean_cadence=0, ema_device=cpu, ppo_max_token_len=18432 (OOM guard).
#   PROBE_OFF : same powersgd r=77 codec, anchor+spectral OFF, ppo_max_token_len=36864 (A1 setting).
# The codec block (rank=77, sync_basis, update_cadence=1, warm_start, compress_recompute,
# qr_dtype=fp32, seed=0) is copied verbatim from EXP-20 and held constant in both probes.
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-23
LOG_ON="$RUN_DIR/smoke_on.log"
LOG_OFF="$RUN_DIR/smoke_off.log"

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# --- Apply the experimental bundle (code_change=true): replace the
#     template-installed /workspace/verl with the exp/23-stale-reanchor branch
#     (which carries the patched launcher with the 3 wired knobs). ---
if [[ -f "$RUN_DIR/exp.bundle" && ! -f "$RUN_DIR/.bundle_applied" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload
  git clone -b "exp/23-stale-reanchor" "$RUN_DIR/exp.bundle" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  echo "=== pip install --no-deps -e . (exp/23 branch) ==="
  uv pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1 || pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1
  touch "$RUN_DIR/.bundle_applied"
fi
cd /workspace/verl
echo "=== verl HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) ==="
echo "=== launcher knob wiring check (expect 3 hits) ==="
grep -cE "spectral.correction_mode=|spectral.inject_gamma=|spectral.blend_eta=" \
  examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# --- Shared codec env (verbatim EXP-20 PowerSGD r=77 block). Applies to BOTH probes. ---
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77
export COMM_EFF_POWERSGD_SYNC_BASIS=true
export COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true
export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_QR_DTYPE=fp32
export COMM_EFF_POWERSGD_SEED=0
export COMM_EFF_POWERSGD_PP_SIZE=8
export COMM_EFF_POWERSGD_REORTHO_EPS=1e-6
export COMM_EFF_CLEAN_CADENCE=0
# Smoke horizon: 2 steps, no validation.
export TOTAL_TRAINING_STEPS=2
export TEST_FREQ=0
export VAL_BEFORE_TRAIN=False

run_probe () {
  local name="$1"; local log="$2"; shift 2
  echo "################################################################"
  echo "### EXP-23 SMOKE $name  $(date -u +%FT%TZ)"
  echo "################################################################"
  # The launcher runs training under `set -x` (run_qwen3_4b_fsdp.sh) and redirects
  # ITS OWN stdout+stderr to $LOG (its internal var). Pin LOG=$log per probe so the
  # fully-expanded main_ppo command (resolved comm_eff.* knobs) + all training
  # output land in ONE known file — the ground truth for capture_resolved_config.py.
  ( "$@" LOG="$log" bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh ) > "${log}.driver" 2>&1
  local rc=$?
  echo "=== $name exit rc=$rc (training log: $log ; driver: ${log}.driver) ==="
  return $rc
}

# --- PROBE_ON: the integration proof (anchor + spectral inject on PowerSGD). ---
# delay_K trap: launcher default is 20 — pass 5 explicitly. OOM guard: 18432 + ema_device=cpu.
run_probe PROBE_ON "$LOG_ON" \
  env EXPERIMENT_NAME=exp-23-smoke-on \
      COMM_EFF_ANCHOR_ENABLED=true \
      COMM_EFF_ANCHOR_CADENCE=5 \
      COMM_EFF_ANCHOR_DELAY_K=5 \
      COMM_EFF_SPECTRAL_ENABLED=true \
      COMM_EFF_SPECTRAL_CORRECTION_MODE=inject \
      COMM_EFF_SPECTRAL_INJECT_GAMMA=1.0 \
      COMM_EFF_SPECTRAL_CADENCE=5 \
      COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
      PPO_MAX_TOKEN_LEN_PER_GPU=18432 \
      LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
      REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432
RC_ON=$?

# --- PROBE_OFF: off-path parity proof (PowerSGD r=77 byte-identical when circuit OFF). ---
run_probe PROBE_OFF "$LOG_OFF" \
  env EXPERIMENT_NAME=exp-23-smoke-off \
      COMM_EFF_ANCHOR_ENABLED=false \
      COMM_EFF_SPECTRAL_ENABLED=false \
      PPO_MAX_TOKEN_LEN_PER_GPU=36864 \
      LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864 \
      REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
RC_OFF=$?

echo "=== EXP-23 SMOKE done  PROBE_ON rc=$RC_ON  PROBE_OFF rc=$RC_OFF  $(date -u +%FT%TZ) ==="
echo "{\"probe_on_rc\": $RC_ON, \"probe_off_rc\": $RC_OFF, \"ts\": \"$(date -u +%FT%TZ)\"}" > "$RUN_DIR/smoke.done.flag"
