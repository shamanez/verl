#!/usr/bin/env bash
# EXP-33 β_anc sweep driver — 5 cells SEQUENTIAL on ONE box. Plan: research/.claude/plans/33.md
# config-only: B2 SOTA substrate + Hydra passthrough spectral.beta_anc=<val>. NO branch, NO verl/ patch.
# Cells (EXPERIMENT_NAME / beta_anc / steps):
#   b0p00 / 0.00 / 55   <- C0 control (must reproduce B2 band; gross-failure-gated)
#   b0p25 / 0.25 / 55   <- C1 curve point
#   b0p50 / 0.50 / 55   <- C2 curve point
#   b0p75 / 0.75 / 55   <- C3 curve point
#   b1p00 / 1.00 / 30   <- C4 degenerate bracket (β=1 -> frozen-zero M -> no-merger floor); val@25 read only
# NOTE: NOT `set -e` — a single cell's nonzero RC must NOT abort the chain (ignition/OOM of a
# non-C0 cell IS that cell's result; continue the others — plan §Notes for runner).
set -uo pipefail
cd /workspace/verl

RUNROOT=/workspace/runs/EXP-33
mkdir -p "$RUNROOT/metrics"
DRIVERLOG="$RUNROOT/driver.log"
exec > >(tee -a "$DRIVERLOG") 2>&1

echo "=== EXP-33 driver start $(date -Iseconds) ==="
echo "=== code: $(git rev-parse --abbrev-ref HEAD) $(git log --oneline -1) ==="

# --- substrate guards (config-only — assert the B2 SOTA path exists; do NOT fetch) ---
test -f examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
  || { echo "FATAL: b2_sota launcher missing — stale box"; exit 3; }
grep -q "delayed_ef" verl/workers/comm_eff/spectral_filter.py \
  || { echo "FATAL: delayed_ef merger missing — stale code"; exit 3; }
grep -q "beta_anc" verl/workers/comm_eff/spectral_filter.py \
  || { echo "FATAL: beta_anc EMA missing — stale code"; exit 3; }
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

# Box-dependent vLLM CUDA-IPC workaround. DEFAULT false (B2 ran custom_all_reduce; keep
# generation byte-identical to B2). If C0 hits the EngineCore custom_all_reduce crash at
# KV-cache init, relaunch the driver with DISABLE_CUSTOM_ALL_REDUCE=true (applies to ALL cells).
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-false}"

# extract the latest GSM8K greedy val acc from a cell log (permissive on quote/colon style)
val_acc () {
  grep -oE "val-core/openai/gsm8k/acc/mean@1['\"]?[: ]+[0-9.]+" "$1" 2>/dev/null \
    | tail -1 | grep -oE "[0-9.]+$"
}

run_cell () {
  local name="$1" beta="$2" steps="$3"
  local celldir="/workspace/verl/runs/$name"
  local log="$celldir/train.log"
  mkdir -p "$celldir"
  ln -sf "$log" /workspace/train.log    # liveness + sync-metrics contract
  echo "=== [$(date -Iseconds)] CELL $name beta_anc=$beta steps=$steps START ==="
  local rc=0
  PROJECT_NAME=verl_compression_research_beta_sweep \
  EXPERIMENT_NAME="$name" \
  TOTAL_TRAINING_STEPS="$steps" \
  TEST_FREQ=25 \
  DISABLE_CUSTOM_ALL_REDUCE="$DISABLE_CUSTOM_ALL_REDUCE" \
    bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
      actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$beta" || rc=$?
  echo "$(date -Iseconds) $name beta_anc=$beta done rc=$rc" > "$RUNROOT/done_${name}.flag"
  cp -f "$log" "$RUNROOT/train_${name}.log" 2>/dev/null || true
  local v; v=$(val_acc "$log")
  echo "=== [$(date -Iseconds)] CELL $name done rc=$rc val(last)=${v:-NA} ==="
  CELL_VAL="$v"
  return 0
}

# ---- C0 (control, β=0) FIRST -------------------------------------------------
run_cell b0p00 0.00 55
C0="$CELL_VAL"
echo "=== C0 (b0p00) val(last)=${C0:-NA} ==="
# Gross-failure gate: abort the chain ONLY if C0 is parsed AND clearly broken (< 0.55 =
# stall / box anomaly). A merely-noisy-but-valid control proceeds; the strict B2 band
# [0.716,0.774] is the analyst's call, not the driver's. Empty parse => proceed (fail-safe).
if [[ -n "${C0:-}" ]] && awk "BEGIN{exit !($C0 < 0.55)}"; then
  echo "=== CONTROL_FAIL: C0 val=$C0 < 0.55 — box/seed anomalous; ABORT C1-C4 (broken control voids the curve) ==="
  echo "$(date -Iseconds) CONTROL_FAIL C0=$C0" > "$RUNROOT/CONTROL_FAIL.flag"
  echo "$(date -Iseconds) EXP-33 driver aborted: control fail C0=$C0" > "$RUNROOT/done.flag"
  exit 0
fi

# ---- C1-C3 curve points (full 55 steps -> val@50) ---------------------------
run_cell b0p25 0.25 55
run_cell b0p50 0.50 55
run_cell b0p75 0.75 55
# ---- C4 degenerate bracket (β=1; frozen-zero M -> no-merger floor); val@25 read only ----
run_cell b1p00 1.00 30

echo "$(date -Iseconds) EXP-33 all cells done" > "$RUNROOT/done.flag"
echo "=== EXP-33 driver complete $(date -Iseconds) ==="
