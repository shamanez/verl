#!/usr/bin/env bash
# EXP-33 β_anc sweep RESUME driver — C2→C3→C4 ONLY (C0/C1 already done+banked).
# Plan: research/.claude/plans/33.md §RESUME STATE. config-only: B2 SOTA substrate +
# Hydra passthrough spectral.beta_anc=<val> trainer.val_before_train=false. NO branch, NO verl/ patch.
# Cells (EXPERIMENT_NAME / beta_anc / steps):
#   b0p50 / 0.50 / 55   <- C2 curve point (died gs=4 on old box; clean re-run)
#   b0p75 / 0.75 / 55   <- C3 curve point
#   b1p00 / 1.00 / 30   <- C4 degenerate bracket (β=1 -> frozen-zero M -> no-merger floor); val@25 read only
# NOT `set -e` — a single cell's nonzero RC must NOT abort the chain (ignition/OOM of a
# non-C0 cell IS that cell's result; continue the others — plan §Notes for runner).
# val_before_train=false: operator-directed measurement opt (val@0 identical across cells;
# reuse C0's 0.08188). Hydra passthrough, last-wins. See runs/EXP-33/MEASUREMENT_NOTE.md.
set -uo pipefail
cd /workspace/verl

RUNROOT=/workspace/runs/EXP-33
mkdir -p "$RUNROOT/metrics"
DRIVERLOG="$RUNROOT/driver_resume.log"
exec > >(tee -a "$DRIVERLOG") 2>&1

echo "=== EXP-33 RESUME driver start $(date -Iseconds) ==="
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
# generation byte-identical to B2). If a cell hits the EngineCore custom_all_reduce crash at
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
      actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$beta" \
      trainer.val_before_train=false || rc=$?
  echo "$(date -Iseconds) $name beta_anc=$beta done rc=$rc" > "$RUNROOT/done_${name}.flag"
  cp -f "$log" "$RUNROOT/train_${name}.log" 2>/dev/null || true
  local v; v=$(val_acc "$log")
  echo "=== [$(date -Iseconds)] CELL $name done rc=$rc val(last)=${v:-NA} ==="
  CELL_VAL="$v"
  return 0
}

# ---- C2 (β=0.50) curve point — 55 steps -> val@50 ----------------------------
run_cell b0p50 0.50 55
# ---- C3 (β=0.75) curve point — 55 steps -> val@50 ----------------------------
run_cell b0p75 0.75 55
# ---- C4 degenerate bracket (β=1; frozen-zero M -> no-merger floor); val@25 read only ----
run_cell b1p00 1.00 30

echo "$(date -Iseconds) EXP-33 resume cells (C2-C4) done" > "$RUNROOT/done.flag"
echo "=== EXP-33 RESUME driver complete $(date -Iseconds) ==="
