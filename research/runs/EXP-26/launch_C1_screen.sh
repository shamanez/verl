#!/usr/bin/env bash
# EXP-26 Step C1 — single-cell PASSIVE Q-family screen on the LOCKED substrate.
# Live q_basis=act (control); families {act,grad,adv,tail,hybrid,ticket} accumulate
# PASSIVELY inside the anchor pass. Per-family candidate Q_f + G_b dumped at the
# anchor cadence; the analyst judges them offline (update-capture, off-principal
# preservation) against the captured G_fresh_anchor reference.
#
# Tick math: cadence=5, 2 ticks/global-step. Anchor fires at ticks {5,10,15,...}.
#   Tick 5 = anchor's FIRST refresh (forward used COLD seed Q); warm by tick 10/15.
#   TOTAL_TRAINING_STEPS=9  -> 18 ticks (anchor fires at 5,10,15)
#   CAPTURE_MIN_TICK=9      -> skip cold ticks 1-8; budget holds ticks 9-16
#   CAPTURE_MAX_TICKS=8     -> captures ticks 9..16 => POST-warm anchor fires at
#                             10,15 => 2 family-screen dumps (Q_f x 6 families x 7
#                             boundaries) + G_fresh_anchor pairings.
# (Identical window to Step-A launch_A1A2_optionA.sh, which validated it.)
set -euo pipefail
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G"
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "C1_ABORT <40G"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1
echo "=== at $(git rev-parse --short HEAD) ===" | tee /workspace/runs/EXP-26/c1_screen.log
uv pip install --no-deps -e . > /workspace/c1_screen_pip.log 2>&1 || pip install --no-deps -e . > /workspace/c1_screen_pip.log 2>&1 || true

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# ---- LOCKED substrate (C1 = Step A's A1 arm: plain PowerSGD r77, anchor owns Q,
#      cadence=5/delay_K=5, clean_cadence=0, spectral OFF / NO merger). ----
export COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false
export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_SPECTRAL_ENABLED=false                       # plain PowerSGD, no merger (A1/C1)
export COMM_EFF_POWERSGD_Q_BASIS=act                         # LIVE basis = act (control)
export COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[act,grad,adv,tail,hybrid,ticket]'  # PASSIVE screen
# Capture ON, POST-warm window (skip cold ticks so the budget holds the anchor fires).
export COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_MAX_TICKS=8 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export COMM_EFF_CAPTURE_MIN_TICK=9
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu   # EXP-16 OOM guard
# SAVE_FREQ huge so NO checkpoint save (the probe's final-step save loaded the host
# DataLoader to a Killed signal; we do not need checkpoints for the screen).
export TOTAL_TRAINING_STEPS=9 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000

CAPDIR=/workspace/captures/C1_screen; rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
echo "=== C1 SCREEN START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/c1_screen.log
COMM_EFF_CAPTURE_DIR="$CAPDIR" EXPERIMENT_NAME=exp26_C1_screen \
  bash "$LAUNCHER" > /workspace/train.log 2>&1 \
  || echo "(screen arm nonzero rc — inspect train.log; benign post-run teardown is OK)" | tee -a /workspace/runs/EXP-26/c1_screen.log

# Gate-check + manifest summary against the REAL main_ppo log ($LOG inside the launcher
# == /workspace/verl/runs/exp26_C1_screen/train.log) AND the tee'd /workspace/train.log.
LL=/workspace/verl/runs/exp26_C1_screen/train.log
[ -f "$LL" ] || LL=/workspace/train.log
M="$CAPDIR/rank0/manifest.jsonl"
{
echo "=== C1 SCREEN GATE CHECKS (grep $LL) ==="
echo "-- training completed all 9 steps --"; grep -oE "Training Progress: 100%[^]]*9/9" "$LL" | tail -1 || echo "(check progress)"
echo "-- NaN/OOM/training Traceback (benign post-run DataLoader/wandb teardown excluded) --"
grep -iE "CUDA out of memory|OutOfMemoryError|nan detected|REALISM_INVARIANT|PROBE_LEAKS_INTO_OPTIMIZER" "$LL" | head -5 || echo "(none)"
echo "-- family screen fires (POST-warm ticks 10,15; all 6 families x 7 boundaries) --"
grep -E "EXP-26\]\[family-screen\]" "$LL" | grep -oE "step=[0-9]+ tick=[0-9]+ families=\[[^]]*\] boundaries=[0-9]+ family_screen_builds=[0-9]+ adv_weight=[a-z]+" | head -6
echo "-- bcast (Q updated=True, powersgd_basis_updates=0, cross_rank_max_rel_dev=0.0) --"
grep -E "\[comm_eff\]\[bcast\]" "$LL" | grep -oE "Q updated=[A-Za-z]+.*powersgd_basis_updates=[0-9]+" | tail -3
echo "-- adv-weight diagnostic (if uniform, says which field is missing) --"
grep -E "EXP-26\]\[adv-weight\]" "$LL" | head -2 || echo "(adv weight set — not uniform)"
echo "-- comm bytes ratio (Step E ~= r/H = 0.05) --"
grep -oE "comm/bytes_ratio:[0-9.]+" "$LL" | tail -2
echo "-- grad_norm finite + reconstruction post-warm --"
grep -oE "actor/grad_norm:[0-9.eE+-]+" "$LL" | tail -3
grep -oE "powersgd_reconstruction_rel_error:[0-9.]+" "$LL" | tail -3
echo "=== MANIFEST ($M) ==="
} | tee -a /workspace/runs/EXP-26/c1_screen.log
[ -f "$M" ] && python3 - "$M" <<'PYEOF' | tee -a /workspace/runs/EXP-26/c1_screen.log
import json,collections,sys
r=[json.loads(l) for l in open(sys.argv[1])]
c=collections.Counter(x["role"] for x in r)
print("roles:", dict(c))
print("family Q roles:", sorted(k for k in c if k.startswith("Q_")))
print("G_b dumped:", c.get("G_b",0), "G_fresh_anchor:", c.get("G_fresh_anchor",0))
print("dtypes:", set(x["dtype"] for x in r))
print("ticks captured:", sorted(set((x["global_step"],x["optimizer_tick"]) for x in r)))
PYEOF

echo "$(date -Iseconds) done" > /workspace/captures/C1_screen/stepC1.done.flag
echo "$(date -Iseconds) stepC1_done" > /workspace/runs/EXP-26/stepC1.done.flag
echo "=== C1 SCREEN DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/c1_screen.log
