#!/usr/bin/env bash
# EXP-26 Step B 4th cell — B-ef, GATED behind the C2+B chain (cells 2-3).
# C2_hybrid FAILED its gate (val@50=0.373 << 0.7414 floor 0.6914; train-vs-rollout
# divergence, NOT collapse). Per STEP_C_SPEC.md decision tree, the ef arm therefore
# runs with q_basis=act (NOT hybrid).
#
# This waiter blocks on /workspace/runs/EXP-26/c2b_chain.done.flag, then runs ONE
# 50-step cell exp26_B_ef with the SAME run_arm conventions as launch_C2B_chain.sh
# (fresh ckpt+rundir clear, capture ON post-warm, per-cell done flag + log +
# resolved-params capture + manifest). Runs in its OWN tmux so it survives the chain's
# tmux exiting. Zero idle on the box: starts the instant the chain done-flag appears.
#
# B-ef config: LOCKED substrate (powersgd r=77, sync_basis, anchor owns_q, cadence=5,
# delay_K=5, clean_cadence=0), q_basis=act, spectral ON correction_mode=ef_powersgd,
# ef_decay=0.9 ef_clip=1.0 (config-documented typical live value; NO sign term — the
# direction-preserving merger). Capture ON => the terminal analyst gets G_comp AND
# G_corr pairs for cos(G_comp,G_corr). Step-E byte counters ON. 50 steps, val@25/50.
set -uo pipefail   # NOT -e: post-run gate-check must never abort the cell.
cd /workspace/verl
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RESOLVED=research/scripts/capture_resolved_config.py
CHAIN_FLAG=/workspace/runs/EXP-26/c2b_chain.done.flag
LOGF=/workspace/runs/EXP-26/bef_chain.log

echo "=== B-ef WAITER armed $(date -Iseconds): blocking on $CHAIN_FLAG ===" | tee "$LOGF"
# Wait (indefinitely — the chain runs ~6h more) for the C2+B chain to finish.
# Poll every 30s; print a heartbeat every ~10 min so the log shows it is alive.
hb=0
until [ -f "$CHAIN_FLAG" ]; do
  sleep 30
  hb=$((hb+1))
  if [ $((hb % 20)) -eq 0 ]; then echo "=== B-ef waiter still waiting ($((hb/2)) min) $(date -Iseconds) ===" | tee -a "$LOGF"; fi
done
echo "=== chain done-flag SEEN ($(cat "$CHAIN_FLAG" 2>/dev/null)); starting B-ef $(date -Iseconds) ===" | tee -a "$LOGF"
echo "=== at $(git rev-parse --short HEAD) ===" | tee -a "$LOGF"
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G" | tee -a "$LOGF"

# ---- shared env (identical to launch_C2B_chain.sh: fixed control surface + post-warm capture) ----
export TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu   # EXP-16 OOM guard
export COMM_EFF_CAPTURE_MIN_TICK=10 COMM_EFF_CAPTURE_MAX_TICKS=12 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32

# run_arm <cell-name> <capture-on:true|false> <env KEY=VAL ...> — VERBATIM conventions
# from launch_C2B_chain.sh (fresh ckpt clear, done flag, log, resolved-params, manifest).
run_arm () {
  local arm="$1"; shift
  local cap_on="$1"; shift
  local capdir="/workspace/captures/$arm"
  echo "=== B-ef cell=$arm START $(date -Iseconds) ===" | tee -a "$LOGF"
  # Defeat verl auto-resume (resume_mode=auto): clear any stale checkpoint + run dir
  # so the cell starts FRESH at global_step 1 (a stale ckpt would resume past step 50).
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
    || echo "(cell $arm nonzero rc — inspect train_${arm}.log; benign post-run wandb/DataLoader teardown is OK)" | tee -a "$LOGF"
  local LL="/workspace/verl/runs/${arm}/train.log"
  [ -f "$LL" ] || LL="/workspace/runs/EXP-26/train_${arm}.log"
  python3 "$RESOLVED" "$LL" >> "$LOGF" 2>&1 || echo "(resolved-config capture rc nonzero for $arm)" | tee -a "$LOGF"
  {
    echo "--- [$arm] val (critic/score/mean @ test) ---"; grep -oE "val-core/[^ ]*score/mean[^ ]*:[0-9.eE+-]+|critic/score/mean:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] grad_norm (last 3) ---"; grep -oE "actor/grad_norm:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] response_length mean (length-collapse watch) ---"; grep -oE "response_length/mean:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] ef merger rel_change + residual resets ---"; grep -oE "correction_mode=ef_powersgd rel_change=[^ ]+|residual_reset_on_shape_mismatch:[0-9]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] spectral_corrections (merger ran) ---"; grep -oE "spectral_corrections:[0-9.]+" "$LL" 2>/dev/null | tail -2
    echo "--- [$arm] comm bytes ratio (Step E) ---"; grep -oE "comm/bytes_ratio:[0-9.]+" "$LL" 2>/dev/null | tail -2
    echo "--- [$arm] Training Progress 100% 50/50 ---"; grep -oE "Training Progress: 100%[^]]*50/50" "$LL" 2>/dev/null | tail -1 || echo "(check progress)"
  } | tee -a "$LOGF"
  if [ "$cap_on" = "true" ] && [ -f "$capdir/rank0/manifest.jsonl" ]; then
    python3 -c "import json,collections; r=[json.loads(l) for l in open('$capdir/rank0/manifest.jsonl')]; c=collections.Counter(x['role'] for x in r); print('[$arm] roles=',dict(c)); print('[$arm] ticks=',sorted(set((x['global_step'],x['optimizer_tick']) for x in r))); gc=sorted(set((x['global_step'],x['optimizer_tick']) for x in r if x['role']=='G_comp')); gr=sorted(set((x['global_step'],x['optimizer_tick']) for x in r if x['role']=='G_corr')); print('[$arm] G_comp&G_corr paired ticks=',sorted(set(gc)&set(gr)))" 2>&1 | tee -a "$LOGF"
  fi
  echo "$(date -Iseconds) done" > "/workspace/runs/EXP-26/${arm}.done.flag"
  echo "=== B-ef cell=$arm DONE $(date -Iseconds) ===" | tee -a "$LOGF"
}

# ---- Cell B-ef: q_basis=act, spectral ON + ef_powersgd (decay 0.9, clip 1.0), capture ON ----
run_arm exp26_B_ef true \
  COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
  COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_CLEAN_CADENCE=0 \
  COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true \
  COMM_EFF_POWERSGD_Q_BASIS=act COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]' \
  COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
  COMM_EFF_SPECTRAL_EF_DECAY=0.9 COMM_EFF_SPECTRAL_EF_CLIP=1.0 \
  EXPERIMENT_NAME=exp26_B_ef

echo "$(date -Iseconds) bef_chain_done" > /workspace/runs/EXP-26/bef_chain.done.flag
echo "=== B-ef CELL DONE $(date -Iseconds) ===" | tee -a "$LOGF"
