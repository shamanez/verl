#!/usr/bin/env bash
# Issue #93 cell a10: ANCHOR-OWNED FRLR + UNBIASED residual gain.
#
# a9 plus frlr_unbiased=true: the residual gain becomes the CONSTANT H/k instead
# of the capped, detached, data-dependent gamma, so E[h_hat | h, Q] = h exactly,
# at zero extra wire. One env var from a9, which is what makes it an attribution.
#
# Why this arm exists at all (operator pushback, 2026-07-26). a8 flattening the
# gap trend while still biased shows estimator variance is SUFFICIENT to explain
# much of that trend; it does NOT show bias is excluded. The program's strongest
# bias evidence is the a1/a2 factorial: the BIASED round-to-nearest arm was
# killed at step 60 with 6.9x worse drift at z=+15 while the unbiased
# stochastic-rounding arm survived, one env var apart. FRLR as run is BIASED and
# PRF exact-k is not. That evidence sits on the actor/kl_loss channel this
# program has since discredited, so the law is OPEN, not settled either way, and
# the test is one variable and 6.5 GPU-h.
#
# THIS ARM REQUIRES NEW RUNTIME CODE (commit 1ff5e775). If the fetch cannot
# deliver it we do NOT fake it with a staged launcher: we fall back to the
# 600-step durability run on a8's config, which needs no new code and is the
# other run the program wants, so the GPU still does useful work.
set -uo pipefail
BRANCH=93-mismatch-control-kit
CELL=examples/grpo_trainer/run_93_cell.sh
CELLNAME=a10-frlr-anchorq-unbiased-200
NEXTLOG=/workspace/runs/$CELLNAME/train.log
mkdir -p "$(dirname "$NEXTLOG")"
pre() { echo "[preflight $(date -u +%H:%M:%SZ)] $*" | tee -a "$NEXTLOG"; }

cd /workspace/verl || { pre "FATAL: /workspace/verl missing"; exit 1; }

synced=no
for i in $(seq 1 10); do
  if git fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 \
     && git checkout -B "$BRANCH" FETCH_HEAD >/dev/null 2>&1 \
     && git reset --hard FETCH_HEAD >/dev/null 2>&1; then synced=yes; break; fi
  pre "fetch attempt $i/10 failed, retrying in 30s"; sleep 30
done
pre "checkout synced=$synced HEAD=$(git rev-parse --short HEAD)"

# Two independent proofs that the anchor-owned code actually arrived: the
# runtime method and the launcher arm. Either missing means the old tree.
runtime_ok=no
grep -q 'def anchor_update_basis' verl/workers/comm_eff/activation_mask.py 2>/dev/null \
  && grep -q 'do_anchor_frlr_q' verl/workers/engine/fsdp/transformer_impl.py 2>/dev/null \
  && runtime_ok=yes
arm_ok=no
ARM=a10 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1 && arm_ok=yes
pre "runtime_ok=$runtime_ok arm_ok=$arm_ok"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a

if [[ "$runtime_ok" != "yes" || "$arm_ok" != "yes" ]]; then
  pre "anchor-owned FRLR code NOT present. Falling back to the 600-step"
  pre "durability run on a8's config (no new code needed) to keep the GPU busy."
  export ARM=a7
  export EXPERIMENT_NAME=c600-frlr-qcad20-fallback-b
  export TOTAL_STEPS=600
  export TEST_FREQ=300
  export VAL_BEFORE_TRAIN=True
  export SAVE_FREQ=200
  export COMM_EFF_PROBE_EVERY=5
  export COMM_EFF_PROBE_CTRL_ENABLED=false
  export COMM_EFF_MASK_FRLR_Q_CADENCE=20
  export CKPT_R2_ENABLED=true
  export R2_EXPERIMENT=93-long-horizon-stability
  export R2_REGIME=c600-frlr-qcad20-fallback-b
  exec bash "$CELL"
fi

pre "ARM=a10 resolves with the anchor-owned runtime; launching $CELLNAME"
export ARM=a10
export EXPERIMENT_NAME=$CELLNAME
export TOTAL_STEPS=200
export TEST_FREQ=200
export SAVE_FREQ=200
export COMM_EFF_PROBE_EVERY=5
export COMM_EFF_PROBE_CTRL_ENABLED=false
# R2 sink ON from here on. a5b/a6/a7/a8 are local-only and the disk is at 57%;
# every further cell uploads as it saves so nothing depends on a teardown race.
export CKPT_R2_ENABLED=true
export R2_EXPERIMENT=93-long-horizon-stability
export R2_REGIME=$CELLNAME
exec bash "$CELL"
