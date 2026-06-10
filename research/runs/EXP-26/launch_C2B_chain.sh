#!/usr/bin/env bash
# EXP-26 stage C2+B chain — THREE gate-independent 50-step training arms, SEQUENTIAL,
# one tmux session, chain-doesn't-abort (a failed cell never blocks the next). The
# 4th arm (B-ef) is gated on C2's outcome and is a LATER dispatch — NOT launched here.
#
# Cells (all on the fixed control surface: batch 128, mini 64, lr 1e-6, n=8, resp 16384,
# GRPO no-KL/no-entropy, seed 0, GSM8K):
#   1) exp26_C2_hybrid : LOCKED substrate (powersgd r77, sync_basis, anchor owns_q,
#                        cadence=5, delay_K=5, clean_cadence=0), spectral/merger OFF,
#                        q_basis=hybrid LIVE. 50 steps, val@25/50. fp32 capture ON post-warm.
#   2) exp26_B_plain   : same substrate, q_basis=act, spectral/merger OFF. 50 steps,
#                        val@25/50. The missing 50-step plain-PowerSGD-substrate reference.
#   3) exp26_B_dense   : comm_eff fully OFF (dense control, byte-identical upstream). 50
#                        steps, val@25/50. Expected to reproduce W&B 0.7536 +/- 0.01.
# Per-arm: capture (codec arms only — dense has no codec) + Step-E byte counters (codec
# arms) + resolved-params capture so the analyst can assert identical controlled vars.
set -uo pipefail   # NOT -e: the chain must survive a failed cell.
cd /workspace/verl
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc "0-9"); echo "disk ${AVAIL_GB}G"
git config --global user.email "harness@verl-research.local"; git config --global user.name "verl-research-harness"
echo "=== chain at $(git rev-parse --short HEAD) START $(date -Iseconds) ===" | tee /workspace/runs/EXP-26/c2b_chain.log

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RESOLVED=research/scripts/capture_resolved_config.py

# ---- fixed control surface + 50-step / val@25 (shared across all 3 cells) ----
export TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 VAL_BEFORE_TRAIN=False SAVE_FREQ=100000
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu   # EXP-16 OOM guard
# Post-warm fp32 capture window (codec arms). cadence=5 => anchor fires 5,10,15,20,...
# Hold ticks 10..21 (warm fires 10,15,20) with the stratified per-layer-type subset
# (plan Notes item 5) to keep dump volume sane over 50 steps.
export COMM_EFF_CAPTURE_MIN_TICK=10 COMM_EFF_CAPTURE_MAX_TICKS=12 COMM_EFF_CAPTURE_STRATIFIED=2
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true COMM_EFF_CAPTURE_G_DENSE=false COMM_EFF_CAPTURE_DUMP_DTYPE=fp32

# run_arm <cell-name> <capture-on:true|false> <env KEY=VAL ...> -- (trailing env applied to the launcher)
run_arm () {
  local arm="$1"; shift
  local cap_on="$1"; shift
  local capdir="/workspace/captures/$arm"
  echo "=== CHAIN cell=$arm START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/c2b_chain.log
  # Defeat verl auto-resume (resume_mode=auto): clear any stale checkpoint + run
  # dir for THIS cell so it starts FRESH at global_step 1 (a stale ckpt would
  # resume past step 50 and the cell would do nothing — the probe-v2/v3 trap).
  rm -rf "/workspace/verl/checkpoints/verl_compression_research/$arm" 2>/dev/null || true
  rm -rf "/workspace/verl/runs/$arm" 2>/dev/null || true
  local cap_env=()
  if [ "$cap_on" = "true" ]; then
    rm -rf "$capdir"; mkdir -p "$capdir"
    cap_env=(COMM_EFF_CAPTURE_ENABLED=true COMM_EFF_CAPTURE_DIR="$capdir")
  else
    cap_env=(COMM_EFF_CAPTURE_ENABLED=false)
  fi
  # The arm-specific env is whatever remains in "$@".
  env "${cap_env[@]}" "$@" bash "$LAUNCHER" \
    > "/workspace/runs/EXP-26/train_${arm}.log" 2>&1 \
    || echo "(cell $arm nonzero rc — inspect train_${arm}.log; benign post-run wandb/DataLoader teardown is OK)" | tee -a /workspace/runs/EXP-26/c2b_chain.log
  # Resolved-params provenance (per arm) from the launcher's set -x trace.
  local LL="/workspace/verl/runs/${arm}/train.log"
  [ -f "$LL" ] || LL="/workspace/runs/EXP-26/train_${arm}.log"
  python3 "$RESOLVED" "$LL" >> /workspace/runs/EXP-26/c2b_chain.log 2>&1 || echo "(resolved-config capture rc nonzero for $arm)" | tee -a /workspace/runs/EXP-26/c2b_chain.log
  # Gate summary: val points + grad_norm finite + (codec) manifest roles/ticks.
  {
    echo "--- [$arm] val (critic/score/mean @ test) ---"; grep -oE "val-core/[^ ]*score/mean[^ ]*:[0-9.eE+-]+|critic/score/mean:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -4
    echo "--- [$arm] grad_norm (last 3) ---"; grep -oE "actor/grad_norm:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] response_length mean (length-collapse watch) ---"; grep -oE "response_length/mean:[0-9.eE+-]+" "$LL" 2>/dev/null | tail -3
    echo "--- [$arm] comm bytes ratio (Step E; codec) ---"; grep -oE "comm/bytes_ratio:[0-9.]+" "$LL" 2>/dev/null | tail -2
    echo "--- [$arm] Training Progress 100% 50/50 ---"; grep -oE "Training Progress: 100%[^]]*50/50" "$LL" 2>/dev/null | tail -1 || echo "(check progress)"
  } | tee -a /workspace/runs/EXP-26/c2b_chain.log
  if [ "$cap_on" = "true" ] && [ -f "$capdir/rank0/manifest.jsonl" ]; then
    python3 -c "import json,collections; r=[json.loads(l) for l in open('$capdir/rank0/manifest.jsonl')]; c=collections.Counter(x['role'] for x in r); gb=[round(x.get('norm',-1),8) for x in r if x['role']=='G_b']; print('[$arm] roles=',dict(c)); print('[$arm] ticks=',sorted(set((x['global_step'],x['optimizer_tick']) for x in r))); print('[$arm] G_b zero count=',sum(1 for n in gb if n==0.0),'/',len(gb))" 2>&1 | tee -a /workspace/runs/EXP-26/c2b_chain.log
  fi
  echo "$(date -Iseconds) done" > "/workspace/runs/EXP-26/${arm}.done.flag"
  echo "=== CHAIN cell=$arm DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/c2b_chain.log
}

# ---- Cell 1: C2 — LIVE hybrid q_basis on the LOCKED substrate, spectral OFF, capture ON ----
run_arm exp26_C2_hybrid true \
  COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
  COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_SPECTRAL_ENABLED=false \
  COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true \
  COMM_EFF_POWERSGD_Q_BASIS=hybrid COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]' \
  COMM_EFF_POWERSGD_HYBRID_ACT_COLS=-1 COMM_EFF_POWERSGD_HYBRID_GRAD_COLS=-1 \
  EXPERIMENT_NAME=exp26_C2_hybrid

# ---- Cell 2: B-plain — plain PowerSGD r77 on the SAME substrate, q_basis=act, spectral OFF, capture ON ----
run_arm exp26_B_plain true \
  COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd COMM_EFF_MASK_ENABLED=false \
  COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_OWNS_Q=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_SPECTRAL_ENABLED=false \
  COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true \
  COMM_EFF_POWERSGD_Q_BASIS=act COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[]' \
  EXPERIMENT_NAME=exp26_B_plain

# ---- Cell 3: B-dense — comm-eff FULLY OFF (dense control, byte-identical upstream), NO capture ----
run_arm exp26_B_dense false \
  COMM_EFF_ENABLED=false \
  EXPERIMENT_NAME=exp26_B_dense

echo "$(date -Iseconds) c2b_chain_done" > /workspace/runs/EXP-26/c2b_chain.done.flag
echo "=== C2+B CHAIN DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/c2b_chain.log
