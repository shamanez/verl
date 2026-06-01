#!/usr/bin/env bash
# Big-Math chain: EXP-19 (masked comm-eff) -> EXP-20 (dense baseline), back-to-back
# on the same box to minimise GPU idle. Each child launch.sh runs a full 120-step
# training and writes its own done.flag on success (their internal `set -euo pipefail`
# means done.flag is only written if main_ppo exits 0). `ray stop` between runs frees
# the GPUs cleanly so the second run starts from a clean Ray cluster.
#
# Reward (corrected, mirrors min_rl_add): the parquet at /root/data/bigmath carries
# data_source="DigitalLearningGmbH/MATH-lighteval" -> math_reward.compute_score
# (\boxed{} + is_equiv, returns float 0/1; no None-pred val crash).
set -uo pipefail

log() { echo "[$(date -Iseconds)] [bigmath-chain] $*"; }

log "EXP-19 (masked p=0.9 clean_cadence=20) START"
bash /workspace/runs/EXP-19/launch.sh
log "EXP-19 END (rc=$?); ray stop"
ray stop --force >/dev/null 2>&1 || true
sleep 12

log "EXP-20 (dense baseline, comm-eff OFF) START"
bash /workspace/runs/EXP-20/launch.sh
log "EXP-20 END (rc=$?); ray stop"
ray stop --force >/dev/null 2>&1 || true

log "CHAIN COMPLETE"
echo "$(date -Iseconds) chain done" > /workspace/runs/run_bigmath_chain.done
