#!/usr/bin/env bash
# EXP-26 B-ef RE-RUN (r2) — mitigated relaunch after the first B-ef cell OOM'd at a
# LATE step (>32, before 50) inside the anchor refresh backward
# (transformer_impl.py:1768 -> _forward_backward_batch_inner:991 loss.backward;
# 129 GiB PyTorch-allocated + ~1 GiB reserved-unallocated fragmentation + 4.39 GiB
# co-resident vLLM worker; 4.66 GiB request had only ~3.4 GiB free). val@25=0.6232
# (healthy, merger engaged, no collapse) but no val@50 / no checkpoint.
#
# SCIENCE UNCHANGED. Mitigations ONLY:
#   (a) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the allocator fix for the
#       slow fragmentation ratchet across variable-seqlen steps (PyTorch's own OOM
#       message recommends exactly this). Exported here so the driver + the locally-
#       forked Ray workers (single-node ray.init) inherit it before CUDA-context init.
#   (b) capture.enabled=false — geometry (12 paired G_comp/G_corr ticks) already
#       captured on the failed run (same seed/config; captures are probed side-effect-
#       free so the trajectory is identical). Frees the capture buffers' GPU pressure.
#   (c) ema_device=cpu STAYS (already on from the chain's shared env) — the EF residual
#       e_t + anchor M EMA already live on pinned CPU (~6 GB fp32 across 196 targets
#       kept off-GPU). No additional offload knob exists; no new code.
#   Byte counters stay ON (cheap scalars).
#
# Same LOCKED substrate + q_basis=act + spectral ef_powersgd ef_decay=0.9 ef_clip=1.0,
# seed 0, 50 steps, TEST_FREQ=25. SAME run_arm conventions (fresh ckpt+rundir clear,
# done flag, resolved-params, manifest), final flag bef_r2.done.flag.
set -uo pipefail
cd /workspace/verl
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"

# (a) the allocator fix — set BEFORE main_ppo/ray.init so workers inherit it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RESOLVED=research/scripts/capture_resolved_config.py
LOGF=/workspace/runs/EXP-26/bef_r2.log

echo "=== B-ef r2 at $(git rev-parse --short HEAD) START $(date -Iseconds) ===" | tee "$LOGF"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF (driver)" | tee -a "$LOGF"
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G" | tee -a "$LOGF"
[ "${AVAIL_GB:-0}" -lt 40 ] && { echo "B-ef-r2 ABORT <40G" | tee -a "$LOGF"; exit 9; }

# ---- shared env (identical to launch_C2B_chain.sh fixed control surface) ----
# NOTE capture.enabled=false this run => MIN_TICK/MAX_TICKS/STRATIFIED are inert, but
# we leave ema_device=cpu (the standing OOM guard) + byte counters (default ON).
export TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu

run_arm () {
  local arm="$1"; shift
  local cap_on="$1"; shift
  local capdir="/workspace/captures/$arm"
  echo "=== B-ef-r2 cell=$arm START $(date -Iseconds) ===" | tee -a "$LOGF"
  rm -rf "/workspace/verl/checkpoints/verl_compression_research/$arm" 2>/dev/null || true
  rm -rf "/workspace/verl/runs/$arm" 2>/dev/null || true
  local cap_env=()
  if [ "$cap_on" = "true" ]; then
    rm -rf "$capdir"; mkdir -p "$capdir"
    cap_env=(COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_DIR="$capdir")
  else
    cap_env=(COMM_EFF_CAPTURE_ENABLED=false)
  fi
  env "${cap_env[@]}" "$@" bash "$LAUNCHER" \
    > "/workspace/runs/EXP-26/train_${arm}.log" 2>&1 \
    || echo "(cell $arm nonzero rc — inspect train_${arm}.log; benign post-run teardown is OK)" | tee -a "$LOGF"
  local LL="/workspace/verl/runs/${arm}/train.log"
  [ -f "$LL" ] || LL="/workspace/runs/EXP-26/train_${arm}.log"
  python3 "$RESOLVED" "$LL" >> "$LOGF" 2>&1 || echo "(resolved-config capture rc nonzero for $arm)" | tee -a "$LOGF"
  {
    echo "--- [$arm] val (critic/score/mean @ test = val@25 should reproduce ~0.6232) ---"; grep -oE "val-core/[^ ]*score/mean[^ ]*:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] grad_norm (last 3) ---"; grep -oE "actor/grad_norm:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] response_length mean (length-collapse watch) ---"; grep -oE "response_length/mean:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] ef merger rel_change + residual resets ---"; grep -oE "correction_mode=ef_powersgd rel_change=[^ ]+|residual_reset_on_shape_mismatch:[0-9]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] spectral_corrections + max_memory_allocated_gb (OOM watch) ---"; grep -oE "spectral_corrections:[0-9.]+|max_memory_allocated_gb:[0-9.]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] comm bytes ratio (Step E) ---"; grep -oE "comm/bytes_ratio:[0-9.]+" "$LL" 2>/dev/null | tail -2
    echo "--- [$arm] OOM hit? ---"; grep -cE "CUDA out of memory|OutOfMemoryError" "$LL" 2>/dev/null
    echo "--- [$arm] Training Progress 100% 50/50 ---"; grep -oE "Training Progress: 100%[^]]*50/50" "$LL" 2>/dev/null | tail -1 || echo "(check progress)"
  } | tee -a "$LOGF"
  echo "$(date -Iseconds) done" > "/workspace/runs/EXP-26/${arm}.done.flag"
  echo "=== B-ef-r2 cell=$arm DONE $(date -Iseconds) ===" | tee -a "$LOGF"
}

# Cell B-ef-r2: q_basis=act, spectral ON + ef_powersgd (decay 0.9, clip 1.0), CAPTURE OFF.
run_arm exp26_B_ef_r2 false \
  COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
  COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_CLEAN_CADENCE=0 \
  COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true \
  COMM_EFF_POWERSGD_Q_BASIS=act COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]' \
  COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
  COMM_EFF_SPECTRAL_EF_DECAY=0.9 COMM_EFF_SPECTRAL_EF_CLIP=1.0 \
  EXPERIMENT_NAME=exp26_B_ef_r2

echo "$(date -Iseconds) bef_r2_done" > /workspace/runs/EXP-26/bef_r2.done.flag
echo "=== B-ef r2 CELL DONE $(date -Iseconds) ===" | tee -a "$LOGF"
