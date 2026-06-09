#!/usr/bin/env bash
# EXP-26 Step A FULL RE-RUN — runs INSIDE the container. The first Step-A pass
# produced UNUSABLE captures on ALL THREE arms (A0: comm_eff.enabled=false => no
# capture writer at all; A1+A2: the powersgd activation hook keyed dumps by
# fwd_generation and starved the max_ticks budget so NO G_dense/G_comp/G_corr/M
# landed). The hotfix (origin/exp/26 @ f28880b8+) unifies the capture tick so all
# roles co-locate. This re-runs all 3 arms SEQUENTIALLY (max_parallel=1) on the
# warm box after fast-forwarding /workspace/verl to the hotfix commit.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"
echo "=== Step A re-run: fetch + reset /workspace/verl to origin/exp/26 hotfix ===" \
  | tee -a /workspace/runs/EXP-26/stepA_rerun.log
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -3 | tee -a /workspace/runs/EXP-26/stepA_rerun.log
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2 | tee -a /workspace/runs/EXP-26/stepA_rerun.log
echo "=== /workspace/verl now at $(git rev-parse --short HEAD) ===" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
uv pip install --no-deps -e . > /workspace/stepA_rerun_pip.log 2>&1 || pip install --no-deps -e . > /workspace/stepA_rerun_pip.log 2>&1 || true

# Pre-run probe (hard gate): the unified-tick + ef invariants must be green here.
echo "=== Step A re-run pre-run probe (hard gate) ===" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q \
  >> /workspace/runs/EXP-26/stepA_rerun.log 2>&1 || {
    echo "PROBE_FAILED: Step A re-run invariants did not pass" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
    exit 7
  }
echo "=== pre-run invariants GREEN ===" | tee -a /workspace/runs/EXP-26/stepA_rerun.log

# Common Step-A capture env.
export COMM_EFF_CAPTURE_ENABLED=true
export COMM_EFF_CAPTURE_MAX_TICKS=8
export COMM_EFF_CAPTURE_STRATIFIED=4
export COMM_EFF_CAPTURE_G_DENSE=true
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true
export COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export TOTAL_TRAINING_STEPS=6
export TEST_FREQ=1000
export VAL_BEFORE_TRAIN=False
export SAVE_FREQ=1000

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

run_arm () {
  local arm="$1"; shift
  local capdir="/workspace/captures/$arm"
  rm -rf "$capdir"; mkdir -p "$capdir"   # fresh dir (discard the broken first-pass dumps)
  echo "=== EXP-26 Step A(re-run) arm=$arm START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
  COMM_EFF_CAPTURE_DIR="$capdir" \
    "$@" bash "$LAUNCHER" \
    > "/workspace/runs/EXP-26/train_${arm}_rerun.log" 2>&1 || {
      echo "ARM_FAILED: $arm (see train_${arm}_rerun.log)" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
    }
  mkdir -p "/workspace/runs/EXP-26/captures/$arm"
  rm -rf "/workspace/runs/EXP-26/captures/$arm"/*   # replace the broken first-pass mirror
  cp -r "$capdir"/. "/workspace/runs/EXP-26/captures/$arm/" 2>/dev/null || true
  # Quick role check so the log shows whether gradient roles landed this time.
  M="$capdir/rank0/manifest.jsonl"
  if [ -f "$M" ]; then
    python3 -c "import json,collections; r=[json.loads(l) for l in open('$M')]; print('[$arm] capture roles:', dict(collections.Counter(x['role'] for x in r)))" \
      | tee -a /workspace/runs/EXP-26/stepA_rerun.log
  fi
  echo "=== EXP-26 Step A(re-run) arm=$arm DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/stepA_rerun.log
}

# A0 dense reference: comm_eff ENABLED + TRUE-IDENTITY codec/merger (so capture +
# G_dense backward fire while applying ZERO compression: G_comp==G_corr==G_dense).
run_arm A0_dense \
  env COMM_EFF_ENABLED=true \
      COMM_EFF_COMPRESSION_TYPE=dense \
      COMM_EFF_MASK_ENABLED=false \
      COMM_EFF_SPECTRAL_ENABLED=true \
      COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
      COMM_EFF_SPECTRAL_EF_DECAY=0.0 \
      COMM_EFF_SPECTRAL_EF_CLIP=0.0 \
      EXPERIMENT_NAME=exp26_A0_dense_rerun

# A1 plain PowerSGD r77 (anchor on + owns Q, NO merger) — H1 discriminator.
run_arm A1_powersgd_r77 \
  env COMM_EFF_SPECTRAL_ENABLED=false EXPERIMENT_NAME=exp26_A1_powersgd_r77_rerun

# A2 EXP-25 anchor + signed_ema alpha=0.5 — the falsified merger; confirms H1.
run_arm A2_signed_ema_a0p5 \
  env COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema \
      COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.5 \
      EXPERIMENT_NAME=exp26_A2_signed_ema_a0p5_rerun

echo "$(date -Iseconds) stepA_rerun_all_done" > /workspace/runs/EXP-26/stepA_rerun.done.flag
echo "=== EXP-26 Step A RE-RUN COMPLETE — captures under /workspace/captures/<arm>/ ===" \
  | tee -a /workspace/runs/EXP-26/stepA_rerun.log
