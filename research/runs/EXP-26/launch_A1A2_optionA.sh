#!/usr/bin/env bash
# EXP-26 Option-A authoritative re-run: A1_powersgd_r77 + A2_signed_ema_a0p5 on the
# LOCKED substrate (cadence=5/delay_K=5). Captures the dense reference
# G_fresh_anchor@delay_K=0 + G_comp/G_corr at POST-Q-warm anchor fires.
#
# Tick math: cadence=5, 2 ticks/global-step. Anchor fires at ticks {5,10,15,...}.
# Tick 5 = anchor's FIRST refresh (forward used COLD seed Q). Q is warm for
# forwards at ticks>=6. So post-warm anchor fires are at 10, 15. We set:
#   TOTAL_TRAINING_STEPS=9  -> 18 ticks (fires at 5,10,15)
#   CAPTURE_MIN_TICK=9      -> skip cold ticks 1-8; budget holds ticks 9-16
#   CAPTURE_MAX_TICKS=8     -> captures ticks 9..16 (incl. anchor fires 10,15 => 2
#                              post-warm G_fresh_anchor pairings x 14 targets)
# G_fresh_anchor ONLY dumps at anchor-fire ticks (capture_fresh_anchor=true), so
# the H1 pairing cos(G_fresh_anchor,G_comp) lands at ticks 10 and 15 (post-warm).
set -euo pipefail
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G"
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "A1A2_ABORT <40G"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -1
echo "=== at $(git rev-parse --short HEAD) ===" | tee /workspace/runs/EXP-26/a1a2.log
uv pip install --no-deps -e . > /workspace/a1a2_pip.log 2>&1 || pip install --no-deps -e . > /workspace/a1a2_pip.log 2>&1 || true
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q >> /workspace/runs/EXP-26/a1a2.log 2>&1 || { echo "PROBE_FAILED"; exit 7; }
echo "=== CPU invariants GREEN ===" | tee -a /workspace/runs/EXP-26/a1a2.log

# LOCKED substrate + post-warm capture (the science arms hold cadence=5/delay_K=5).
export COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false
export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_MAX_TICKS=8 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_G_DENSE=true COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export COMM_EFF_CAPTURE_MIN_TICK=9
export TOTAL_TRAINING_STEPS=9 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=1000

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

run_arm () {
  local arm="$1"; shift
  local capdir="/workspace/captures/$arm"; rm -rf "$capdir"; mkdir -p "$capdir"
  echo "=== A12 arm=$arm START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/a1a2.log
  COMM_EFF_CAPTURE_DIR="$capdir" "$@" bash "$LAUNCHER" \
    > "/workspace/runs/EXP-26/train_${arm}_optA.log" 2>&1 || echo "(arm nonzero — check train_rc; likely benign wandb teardown)" | tee -a /workspace/runs/EXP-26/a1a2.log
  M="$capdir/rank0/manifest.jsonl"
  [ -f "$M" ] && python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; c=collections.Counter(x['role'] for x in r); fa=sorted(set((x['global_step'],x['optimizer_tick']) for x in r if x['role']=='G_fresh_anchor')); gc=sorted(set((x['global_step'],x['optimizer_tick']) for x in r if x['role']=='G_comp')); print(f'[$arm] roles={dict(c)}'); print(f'[$arm] G_fresh_anchor ticks={fa}'); print(f'[$arm] G_comp ticks={gc}'); print(f'[$arm] H1-pairable ticks (fa & gc)={sorted(set(fa)&set(gc))}')" | tee -a /workspace/runs/EXP-26/a1a2.log
  echo "=== A12 arm=$arm DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/a1a2.log
}

# A1 — plain PowerSGD r77, anchor-owns-Q, NO merger (the H1 reference: needs G_fresh_anchor + G_comp).
run_arm A1_powersgd_r77 \
  env COMM_EFF_SPECTRAL_ENABLED=false EXPERIMENT_NAME=exp26_A1_optA

# A2 — anchor + signed_ema alpha=0.5 (G_fresh_anchor + G_comp + G_corr + M).
run_arm A2_signed_ema_a0p5 \
  env COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema \
      COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.5 EXPERIMENT_NAME=exp26_A2_optA

echo "$(date -Iseconds) a1a2_done" > /workspace/runs/EXP-26/a1a2.done.flag
echo "=== A1+A2 OPTION-A RE-RUN DONE ===" | tee -a /workspace/runs/EXP-26/a1a2.log
