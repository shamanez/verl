#!/usr/bin/env bash
# EXP-27 (issue #27) — EXP-26.1 REVISE child: DAMPED ef_powersgd merger.
# Single cell exp27_B_ef_damped on the LOCKED substrate (PowerSGD r=77, sync_basis,
# anchor owns Q, cadence=5, delay_K=5, clean_cadence=0, q_basis=act, seed 0).
#
# THE ONLY science deltas vs the EXP-26 B-ef parent arm (launch_Bef_r2_cell.sh):
#     ef_clip   1.0 -> 0.5     (halve the residual norm cap = halve the dose)
#     ef_decay  0.9 -> 0.5     (faster residual bleed-off)
#     steps      50 -> 100     (val@25/50/75/100)
# Plus capture ON post-warm for cos(G_comp,G_corr) + EF residual dose (success criteria),
# but the OOM-risky parallel probes (capture_g_dense, capture_fresh_anchor) STAY OFF — the
# plan's cosine is cos(G_comp,G_corr), which needs neither. With expandable_segments +
# ema_device=cpu + fresh_anchor OFF this is strictly LESS GPU pressure than the parent r1
# (which OOM'd with fresh_anchor=true and no allocator fix).
#
# Run as a detached tmux session; the orchestrator promotes the ledger row to RUNNING
# after the liveness probe and dispatches training-log-monitor.
set -uo pipefail   # NOT -e: a nonzero launcher rc must still write done.flag + resolved params.

ARM=exp27_B_ef_damped
RUN=/workspace/runs/EXP-27
mkdir -p "$RUN"
cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"      2>/dev/null || true

# (a) allocator fix — STANDARD after the parent r1 OOM; set BEFORE ray.init so the
#     single-node-forked Ray workers inherit it before CUDA-context init.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Real training log at the monitor's expected per-cell path (it greps THIS for
# step/OOM/NaN/Traceback). Symlink /workspace/train.log -> it so the sync-metrics
# hook's `tail /workspace/train.log` heartbeat reads the real log too.
export LOG="$RUN/train_${ARM}.log"
ln -sf "$LOG" /workspace/train.log

# Defeat verl resume_mode=auto: clear any stale ckpt/run-dir so we start FRESH at step 1.
rm -rf "/workspace/verl/checkpoints/verl_compression_research/$ARM" 2>/dev/null || true
rm -rf "/workspace/verl/runs/$ARM"                                  2>/dev/null || true
CAPDIR="/workspace/captures/$ARM"; rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"

# Fixed control surface (identical to the EXP-26 B-ef arm except TOTAL_TRAINING_STEPS).
export TOTAL_TRAINING_STEPS=100 TEST_FREQ=25 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu

# Capture ON post-warm: cos(G_comp,G_corr) + EF residual dose. OOM-risky probes OFF.
export COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_DIR="$CAPDIR"
export COMM_EFF_CAPTURE_MIN_TICK=10 COMM_EFF_CAPTURE_MAX_TICKS=12 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=false COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32

echo "=== EXP-27 launch START $(date -Iseconds) @ $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "ARM=$ARM LOG=$LOG"
AVAIL_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9'); echo "disk ${AVAIL_GB:-?}G"

# --- damped-ef merger arm ---------------------------------------------------------------
COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_CLEAN_CADENCE=0 \
COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true \
COMM_EFF_POWERSGD_Q_BASIS=act COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]' \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
COMM_EFF_SPECTRAL_EF_DECAY=0.5 COMM_EFF_SPECTRAL_EF_CLIP=0.5 \
EXPERIMENT_NAME="$ARM" \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RC=$?
echo "=== launcher returned rc=$RC $(date -Iseconds) ==="

# Ground-truth resolved Hydra params from the set -x trace in $LOG.
python3 research/scripts/capture_resolved_config.py "$LOG" > "$RUN/resolved_params.txt" 2>&1 \
  || echo "(resolved-config capture rc nonzero — inspect $LOG)"

# Done flags: aggregate (monitor DONE_AGGREGATE) + per-cell.
echo "$(date -Iseconds) done rc=$RC" > "$RUN/${ARM}.done.flag"
echo "$(date -Iseconds) done rc=$RC" > "$RUN/done.flag"
echo "=== EXP-27 launch DONE rc=$RC $(date -Iseconds) ==="
exit "$RC"
