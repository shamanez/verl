#!/usr/bin/env bash
# EXP-26 C2 pre-launch PROBE v2 — LIVE q_basis=hybrid (Defect-9 warm-act fix) +
# PASSIVE [grad,tail,hybrid] so G_b is DUMPED (Defect-8 fix verification).
# Verify: anchor_q_updates>0, powersgd_basis_updates==0, Q orthonormal (q_cond~1),
# recon_rel_error DROPS toward the act band post-fix, no NaN/OOM, EXP-12 warning gone,
# G_b NONZERO at every anchor fire (from the passive dumps).
# Substrate = LOCKED A1/C1 (plain PowerSGD r77, anchor owns Q, cadence=5/delay_K=5,
# clean_cadence=0, spectral OFF). 5 steps => fires at tick 5 (cold) and 10 (warm).
set -uo pipefail   # NOT -e: the gate-check post-processing must never abort the probe.
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G"
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "PROBE_ABORT <40G"; exit 9; }
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
echo "=== at $(git rev-parse --short HEAD) ===" | tee /workspace/runs/EXP-26/probe_c2.log

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
export COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false
export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_SPECTRAL_ENABLED=false
export COMM_EFF_POWERSGD_Q_BASIS=hybrid                                     # LIVE hybrid (Defect-9 fix under test)
export COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]'                              # live-only (passive screen OOM'd host RAM in v2)
export COMM_EFF_POWERSGD_HYBRID_ACT_COLS=-1 COMM_EFF_POWERSGD_HYBRID_GRAD_COLS=-1
export COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true
export COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_MAX_TICKS=6 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export COMM_EFF_CAPTURE_MIN_TICK=5
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
export TOTAL_TRAINING_STEPS=5 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000

CAPDIR=/workspace/captures/C2_probe; rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
rm -rf /workspace/verl/checkpoints/verl_compression_research/exp26_C2_probe_clean 2>/dev/null || true
rm -rf /workspace/verl/runs/exp26_C2_probe_clean 2>/dev/null || true
echo "=== C2 HYBRID PROBE v2 START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/probe_c2.log
COMM_EFF_CAPTURE_DIR="$CAPDIR" EXPERIMENT_NAME=exp26_C2_probe_clean \
  bash "$LAUNCHER" > /workspace/probe_c2_train.log 2>&1 \
  || echo "(probe nonzero rc — inspect; benign post-run teardown OK)" | tee -a /workspace/runs/EXP-26/probe_c2.log

LL=/workspace/verl/runs/exp26_C2_probe_clean/train.log
[ -f "$LL" ] || LL=/workspace/probe_c2_train.log
M="$CAPDIR/rank0/manifest.jsonl"
{
echo "=== C2 HYBRID PROBE v2 GATE CHECKS ==="
echo "-- reached 5/5 --"; grep -oE "Training Progress: 100%[^]]*5/5" "$LL" | tail -1 || true
echo "-- NaN/OOM/realism alarms --"; grep -iE "CUDA out of memory|OutOfMemoryError|nan detected|REALISM_INVARIANT|PROBE_LEAKS" "$LL" | head -6 || true
echo "-- bcast (anchor_q_updates>0, powersgd_basis_updates=0, dev=0) --"; grep -oE "Q updated=[A-Za-z]+ broadcast boundaries=[0-9]+ changed=[0-9]+ cross_rank_max_rel_dev=[0-9.eE+-]+ anchor_q_updates=[0-9]+ anchor_q_broadcasts=[0-9]+ powersgd_basis_updates=[0-9]+" "$LL" | sort -u || true
echo "-- EXP-12 stale warning count (post-fix should be 0) --"; grep -c "produced NO target grads" "$LL" || true
echo "-- recon_rel_error per step (Defect-9: should DROP toward act band) --"; grep -oE "training/global_step:[0-9]+|powersgd_reconstruction_rel_error:[0-9.]+" "$LL" | paste - - | tail -6 || true
echo "-- q_cond (orthonormal) --"; grep -oE "powersgd_q_cond:[0-9.]+" "$LL" | tail -3 || true
echo "-- grad_norm finite --"; grep -oE "actor/grad_norm:[0-9.eE+-]+" "$LL" | tail -3 || true
echo "-- comm bytes ratio --"; grep -oE "comm/bytes_ratio:[0-9.]+" "$LL" | tail -2 || true
} | tee -a /workspace/runs/EXP-26/probe_c2.log

if [ -f "$M" ]; then
python3 - "$M" <<'PYEOF' | tee -a /workspace/runs/EXP-26/probe_c2.log
import json,collections,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print("roles:",dict(collections.Counter(x["role"] for x in rows)))
gb=sorted([(x["global_step"],x["optimizer_tick"],x["target_name"],round(x.get("norm",-1),8)) for x in rows if x["role"]=="G_b"])
bad=sum(1 for *_,n in gb if n==0.0)
print("G_b dumps (Defect-8: nonzero at EVERY fire):")
for gs,tk,tn,nm in gb: print("  tick",(gs,tk),tn,"norm",nm,("ZERO-BAD" if nm==0.0 else ""))
print("G_b zero count =",bad,"of",len(gb),"->",("PASS_all_nonzero" if bad==0 and len(gb)>0 else ("NO_GB_DUMPED" if len(gb)==0 else "FAIL_still_zero")))
PYEOF
fi
echo "$(date -Iseconds) done" > "$CAPDIR/probe_c2.done.flag"
echo "=== C2 HYBRID PROBE v2 DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/probe_c2.log
