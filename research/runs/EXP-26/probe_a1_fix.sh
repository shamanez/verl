#!/usr/bin/env bash
# EXP-26 on-box verification probe — runs INSIDE the container. Verifies the
# Defect 1/2 fixes on the REAL training path BEFORE the authoritative re-run:
#   - anchor_q_updates > 0 (Defect 1: anchor-owns-Q fires without a merger)
#   - powersgd_reconstruction_rel_error DROPS from ~0.975 post-warmup (Q warms)
#   - G_dense capture targets > 0 (Defect 2)
#   - off-path-parity / probe-never-feeds-optimizer asserts still hold (no crash)
# Uses the A1 arm config (plain PowerSGD r77, anchor-owns-Q, NO merger) — the arm
# that exposed Defect 1 — with anchor.cadence=1 so Q updates EVERY tick (fast
# signal) and a tiny 2-step budget. NOT authoritative (cadence!=5); diagnostic only.
set -euo pipefail
cd /workspace/verl
# Disk pre-check: refuse to start if < 30 GB free (capture + rollouts need headroom).
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
echo "=== probe disk pre-check: ${AVAIL_GB}G free ===" 
if [ "${AVAIL_GB:-0}" -lt 30 ]; then echo "PROBE_ABORT: only ${AVAIL_GB}G free (<30G)"; exit 9; fi
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"
echo "=== probe: reset /workspace/verl to origin/exp/26 (Defect 1/2/3 fixes) ===" | tee /workspace/runs/EXP-26/probe_fix.log
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2 | tee -a /workspace/runs/EXP-26/probe_fix.log
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1 | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "=== at $(git rev-parse --short HEAD) ===" | tee -a /workspace/runs/EXP-26/probe_fix.log
uv pip install --no-deps -e . > /workspace/probe_pip.log 2>&1 || pip install --no-deps -e . > /workspace/probe_pip.log 2>&1 || true

CAPDIR=/workspace/captures/PROBE_a1
rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
# Fast Q-warm: cadence=1 (anchor fires every tick), delay_K=1, 2 steps.
COMM_EFF_ENABLED=true \
COMM_EFF_COMPRESSION_TYPE=powersgd \
COMM_EFF_MASK_ENABLED=false \
COMM_EFF_SPECTRAL_ENABLED=false \
COMM_EFF_ANCHOR_ENABLED=true \
COMM_EFF_ANCHOR_OWNS_Q=true \
COMM_EFF_ANCHOR_CADENCE=1 \
COMM_EFF_ANCHOR_DELAY_K=1 \
COMM_EFF_CAPTURE_ENABLED=true \
COMM_EFF_CAPTURE_DIR="$CAPDIR" \
COMM_EFF_CAPTURE_MAX_TICKS=4 \
COMM_EFF_CAPTURE_STRATIFIED=2 \
COMM_EFF_CAPTURE_G_DENSE=true \
COMM_EFF_CAPTURE_FRESH_ANCHOR=true \
COMM_EFF_CAPTURE_DUMP_DTYPE=fp32 \
TOTAL_TRAINING_STEPS=2 \
TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=1000 \
EXPERIMENT_NAME=exp26_probe_a1_fix \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-26/train_probe_a1_fix.log 2>&1 || echo "PROBE_ARM_FAILED" | tee -a /workspace/runs/EXP-26/probe_fix.log

echo "=== PROBE RESULTS ===" | tee -a /workspace/runs/EXP-26/probe_fix.log
IL=/workspace/verl/runs/exp26_probe_a1_fix/train.log
echo "-- anchor_q_updates (want >0) --" | tee -a /workspace/runs/EXP-26/probe_fix.log
grep -aoE "anchor_q_updates:[0-9.]+" "$IL" 2>/dev/null | tail -3 | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "-- recon_rel_error (want DROP from 0.975) --" | tee -a /workspace/runs/EXP-26/probe_fix.log
grep -aoE "powersgd_reconstruction_rel_error:[0-9.]+" "$IL" 2>/dev/null | tail -3 | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "-- [bcast] lines (want present) --" | tee -a /workspace/runs/EXP-26/probe_fix.log
grep -ac "\[comm_eff\]\[bcast\]" "$IL" 2>/dev/null | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "-- g_dense targets (want >0) --" | tee -a /workspace/runs/EXP-26/probe_fix.log
grep -aoE "captured G_dense targets=[0-9]+" "$IL" 2>/dev/null | tail -4 | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "-- PROBE capture roles --" | tee -a /workspace/runs/EXP-26/probe_fix.log
M="$CAPDIR/rank0/manifest.jsonl"
[ -f "$M" ] && python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; print(dict(collections.Counter(x['role'] for x in r)))" | tee -a /workspace/runs/EXP-26/probe_fix.log
echo "-- any PROBE_LEAKS / REALISM_INVARIANT / Traceback? --" | tee -a /workspace/runs/EXP-26/probe_fix.log
grep -aE "PROBE_LEAKS|REALISM_INVARIANT|Traceback|AssertionError" "$IL" 2>/dev/null | grep -avE "rollout_corr|log_ppl" | tail -4 | tee -a /workspace/runs/EXP-26/probe_fix.log || echo "(none)" | tee -a /workspace/runs/EXP-26/probe_fix.log
touch /workspace/runs/EXP-26/probe_fix.done.flag
echo "=== PROBE DONE ===" | tee -a /workspace/runs/EXP-26/probe_fix.log
