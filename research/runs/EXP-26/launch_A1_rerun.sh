#!/usr/bin/env bash
# EXP-26 A1_powersgd_r77 RE-RUN — captures the no-merger G_comp (Defect 5 fix).
# A0/A2 already have valid captures (their mergers dumped G_comp; the audit-side
# canon pairs them); only A1 (plain PowerSGD, NO merger) lacked G_comp. Resets the
# warm checkout to the fix commit, runs ONLY A1 with the LOCKED substrate
# (cadence=5, delay_K=5), rank0-only + stratified=2 captures.
set -euo pipefail
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
echo "=== A1 re-run disk pre-check: ${AVAIL_GB}G free ===" | tee -a /workspace/runs/EXP-26/a1_rerun.log
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "A1_RERUN_ABORT: <40G free"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2 | tee -a /workspace/runs/EXP-26/a1_rerun.log
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1 | tee -a /workspace/runs/EXP-26/a1_rerun.log
echo "=== at $(git rev-parse --short HEAD) ===" | tee -a /workspace/runs/EXP-26/a1_rerun.log
uv pip install --no-deps -e . > /workspace/a1_pip.log 2>&1 || pip install --no-deps -e . > /workspace/a1_pip.log 2>&1 || true
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q >> /workspace/runs/EXP-26/a1_rerun.log 2>&1 || { echo "PROBE_FAILED"; exit 7; }
echo "=== pre-run invariants GREEN ===" | tee -a /workspace/runs/EXP-26/a1_rerun.log

CAPDIR=/workspace/captures/A1_powersgd_r77
rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
COMM_EFF_SPECTRAL_ENABLED=false \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_DIR="$CAPDIR" COMM_EFF_CAPTURE_MAX_TICKS=8 \
COMM_EFF_CAPTURE_STRATIFIED=2 COMM_EFF_CAPTURE_G_DENSE=true COMM_EFF_CAPTURE_FRESH_ANCHOR=true \
COMM_EFF_CAPTURE_DUMP_DTYPE=fp32 \
TOTAL_TRAINING_STEPS=6 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=1000 \
EXPERIMENT_NAME=exp26_A1_powersgd_r77_rerun2 \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-26/train_A1_rerun2.log 2>&1 || echo "A1_ARM_nonzero (likely benign wandb teardown; check train_rc)" | tee -a /workspace/runs/EXP-26/a1_rerun.log

echo "=== A1 capture roles (expect G_comp now present) ===" | tee -a /workspace/runs/EXP-26/a1_rerun.log
M="$CAPDIR/rank0/manifest.jsonl"
[ -f "$M" ] && python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; print(dict(collections.Counter(x['role'] for x in r)))" | tee -a /workspace/runs/EXP-26/a1_rerun.log
grep -aoE "anchor_q_updates:[0-9.]+|powersgd_reconstruction_rel_error:[0-9.]+" /workspace/verl/runs/exp26_A1_powersgd_r77_rerun2/train.log 2>/dev/null | tail -4 | tee -a /workspace/runs/EXP-26/a1_rerun.log
grep -aoE "g_comp-no-merger.*targets=[0-9]+|done at.*train_rc=[0-9]+" /workspace/runs/EXP-26/train_A1_rerun2.log 2>/dev/null | tail -3 | tee -a /workspace/runs/EXP-26/a1_rerun.log
touch /workspace/runs/EXP-26/a1_rerun.done.flag
echo "=== A1 RE-RUN DONE ===" | tee -a /workspace/runs/EXP-26/a1_rerun.log
