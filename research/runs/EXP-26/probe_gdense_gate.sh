#!/usr/bin/env bash
# EXP-26 G_dense ACCEPTANCE-GATE probe — verifies the Defect-6 fix on a CODEC-ON
# arm: does the parallel uncompressed G_dense backward now match the trusted
# G_fresh_anchor@delay_K=0? GATE: cos>=0.95 AND norm_ratio in [0.8,1.25].
# Fast config: powersgd r77, anchor cadence=1 (Q warms at tick 1), min_tick=2
# (post-warm), 3 steps. Runs gate_check.py on-box at the end.
set -euo pipefail
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk: ${AVAIL_GB}G"
[ "${AVAIL_GB:-0}" -lt 30 ] && { echo "GATE_PROBE_ABORT <30G"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1
echo "=== at $(git rev-parse --short HEAD) ==="
uv pip install --no-deps -e . > /workspace/gate_pip.log 2>&1 || pip install --no-deps -e . > /workspace/gate_pip.log 2>&1 || true
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q > /workspace/runs/EXP-26/gate_probe.log 2>&1 || { echo "PROBE_TESTS_FAILED"; exit 7; }
echo "=== CPU invariants GREEN ===" | tee -a /workspace/runs/EXP-26/gate_probe.log

CAPDIR=/workspace/captures/GATE_probe; rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
COMM_EFF_SPECTRAL_ENABLED=false \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=1 COMM_EFF_ANCHOR_DELAY_K=1 \
COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_DIR="$CAPDIR" COMM_EFF_CAPTURE_MAX_TICKS=6 \
COMM_EFF_CAPTURE_STRATIFIED=2 COMM_EFF_CAPTURE_G_DENSE=true COMM_EFF_CAPTURE_FRESH_ANCHOR=true \
COMM_EFF_CAPTURE_DUMP_DTYPE=fp32 COMM_EFF_CAPTURE_MIN_TICK=2 \
TOTAL_TRAINING_STEPS=3 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=1000 \
EXPERIMENT_NAME=exp26_gate_probe \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-26/train_gate_probe.log 2>&1 || echo "(arm nonzero — check train_rc; likely benign wandb teardown)"

echo "=== capture roles ===" | tee -a /workspace/runs/EXP-26/gate_probe.log
M="$CAPDIR/rank0/manifest.jsonl"
[ -f "$M" ] && python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; print('roles:',dict(collections.Counter(x['role'] for x in r))); print('ticks:',sorted(set(x['optimizer_tick'] for x in r)))" | tee -a /workspace/runs/EXP-26/gate_probe.log
echo "=== g_dense mean_norm log lines (should be O(0.1-1), NOT ~0.01) ===" | tee -a /workspace/runs/EXP-26/gate_probe.log
grep -aoE "g_dense.*mean_norm=[0-9.]+ .*" /workspace/runs/EXP-26/train_gate_probe.log 2>/dev/null | tail -4 | tee -a /workspace/runs/EXP-26/gate_probe.log
echo "=== forward-hook strip assert fire? (should NOT) ===" | tee -a /workspace/runs/EXP-26/gate_probe.log
grep -aE "still has .* forward hook|G_dense: clone still" /workspace/runs/EXP-26/train_gate_probe.log 2>/dev/null | tail -2 | tee -a /workspace/runs/EXP-26/gate_probe.log || echo "(no assert fire — good)" | tee -a /workspace/runs/EXP-26/gate_probe.log
echo "=== THE GATE ===" | tee -a /workspace/runs/EXP-26/gate_probe.log
python /workspace/runs/EXP-26/gate_check.py "$CAPDIR" 2>&1 | tee -a /workspace/runs/EXP-26/gate_probe.log
echo "GATE_EXIT=$?" | tee -a /workspace/runs/EXP-26/gate_probe.log
touch /workspace/runs/EXP-26/gate_probe.done.flag
