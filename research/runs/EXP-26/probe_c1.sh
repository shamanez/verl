#!/usr/bin/env bash
# EXP-26 Step C1 GPU PROBE — exercise the NEW family-screen + byte-counter code on
# the LOCKED substrate for a few optimizer ticks and prove every hard correctness
# invariant binds the new code. Cheap fail-fast gate BEFORE the screen cell.
# Tick math: cadence=5, 2 ticks/global-step -> anchor fires at ticks {5,10}.
#   TOTAL_TRAINING_STEPS=3 -> 6 ticks; anchor fires at tick 5 => ONE family-screen
#   build + Q broadcast + capture pass (enough to validate the gates).
set -euo pipefail
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G"
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "PROBE_ABORT <40G"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
echo "=== at $(git rev-parse --short HEAD) ===" | tee /workspace/runs/EXP-26/probe_c1.log

python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py tests/workers/comm_eff/test_q_family_screen_exp26.py -q \
  >> /workspace/runs/EXP-26/probe_c1.log 2>&1 || { echo "PROBE_FAILED: CPU tests"; tail -30 /workspace/runs/EXP-26/probe_c1.log; exit 7; }
echo "=== CPU invariants GREEN ===" | tee -a /workspace/runs/EXP-26/probe_c1.log

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
export COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false
export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_SPECTRAL_ENABLED=false
export COMM_EFF_POWERSGD_Q_BASIS=act
export COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[act,grad,adv,tail,hybrid,ticket]'
export COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_MAX_TICKS=4 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export COMM_EFF_CAPTURE_MIN_TICK=0
export PPO_MAX_TOKEN_LEN_PER_GPU=18432
export TOTAL_TRAINING_STEPS=3 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=1000

CAPDIR=/workspace/captures/C1_probe; rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
echo "=== C1 PROBE START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/probe_c1.log
COMM_EFF_CAPTURE_DIR="$CAPDIR" EXPERIMENT_NAME=exp26_C1_probe \
  bash "$LAUNCHER" > /workspace/runs/EXP-26/train_c1_probe.log 2>&1 \
  || echo "(probe arm nonzero rc — inspect train log; benign wandb teardown is OK)" | tee -a /workspace/runs/EXP-26/probe_c1.log

L=/workspace/runs/EXP-26/train_c1_probe.log
M="$CAPDIR/rank0/manifest.jsonl"
{
echo "=== PROBE GATE CHECKS (grep $L) ==="
echo "-- NaN/OOM/Traceback (MUST be empty) --"
grep -iE "Traceback|CUDA out of memory|OutOfMemory|nan detected|RuntimeError" "$L" | grep -ivE "no_module|deprecat" | head -8 || echo "(none)"
echo "-- anchor refresh + counters (anchor_backwards>0, anchor_optimizer_steps=0) --"
grep -E "anchor refresh step=" "$L" | tail -3
echo "-- bcast (Q updated=True, powersgd_basis_updates=0) --"
grep -E "\[comm_eff\]\[bcast\]" "$L" | tail -4
echo "-- family screen fired (family_screen_builds>0) --"
grep -E "\[comm_eff\]\[EXP-26\]\[family-screen\]" "$L" | tail -4
echo "-- merger absent (plain PowerSGD; should be NO merger line) --"
grep -E "\[comm_eff\]\[merger\]" "$L" | tail -2 || echo "(no merger — correct for C1)"
echo "-- comm bytes (Step E counters) --"
grep -oE "comm/bytes_(compressed|dense_equiv|ratio)[^ ,}]*" "$L" | tail -6 || echo "(check wandb metrics dict)"
echo "-- PROBE_LEAKS / realism guard asserts (must NOT appear) --"
grep -iE "PROBE_LEAKS_INTO_OPTIMIZER|REALISM_INVARIANT|did NOT update Q|EMPTY sketch" "$L" | head -4 || echo "(none)"
echo "-- grad_norm finite --"
grep -oE "grad_norm[\":= ]+[0-9.eE+-]+" "$L" | tail -4
echo "=== MANIFEST roles ($M) ==="
[ -f "$M" ] && python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; c=collections.Counter(x['role'] for x in r); print('roles=',dict(c)); qf=[k for k in c if k.startswith('Q_')]; print('family Q roles=',sorted(qf)); print('G_b dumped=',c.get('G_b',0)); print('fp32 dtype=',set(x['dtype'] for x in r))" || echo "NO MANIFEST"
} | tee -a /workspace/runs/EXP-26/probe_c1.log

echo "$(date -Iseconds) probe_c1_done" > /workspace/runs/EXP-26/probe_c1.done.flag
echo "=== C1 PROBE DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/probe_c1.log
