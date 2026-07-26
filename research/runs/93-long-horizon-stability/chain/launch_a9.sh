#!/usr/bin/env bash
# Issue #93 cell a9: ANCHOR-OWNED FRLR.
#
# a7's exact codec, but the fast path is no longer a Q writer at all: the basis
# is harvested from the anchor's clean stale-weight forward and refreshed only
# when the anchor fires (cadence 20 optimizer ticks). That is the operator's
# instruction of 2026-07-26 ("Q update only in the anchor and only when it
# fires, like in normal powerSGD Q") and the governance PowerSGD has always had.
#
# Beyond a8's frlr_q_cadence=20, which only slowed the FAST refresh:
#   1. Q is fitted to the SLOW stale-weight net, so it cannot chase the policy.
#   2. the Q broadcast rides the slow circuit, which this program does not charge
#      to the boundary wire budget, so FRLR regains exact 1232-bit parity.
#
# THIS ARM REQUIRES NEW RUNTIME CODE (commit 1ff5e775). If the fetch cannot
# deliver it we do NOT fake it with a staged launcher: we fall back to the
# 600-step durability run on a8's config, which needs no new code and is the
# other run the program wants, so the GPU still does useful work.
set -uo pipefail
BRANCH=93-mismatch-control-kit
CELL=examples/grpo_trainer/run_93_cell.sh
CELLNAME=a9-frlr-anchorq-200
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
ARM=a9 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1 && arm_ok=yes
pre "runtime_ok=$runtime_ok arm_ok=$arm_ok"

set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a

if [[ "$runtime_ok" != "yes" || "$arm_ok" != "yes" ]]; then
  pre "anchor-owned FRLR code NOT present. Falling back to the 600-step"
  pre "durability run on a8's config (no new code needed) to keep the GPU busy."
  export ARM=a7
  export EXPERIMENT_NAME=c600-frlr-qcad20-fallback
  export TOTAL_STEPS=600
  export TEST_FREQ=300
  export VAL_BEFORE_TRAIN=True
  export SAVE_FREQ=200
  export COMM_EFF_PROBE_EVERY=5
  export COMM_EFF_PROBE_CTRL_ENABLED=false
  export COMM_EFF_MASK_FRLR_Q_CADENCE=20
  export CKPT_R2_ENABLED=true
  export R2_EXPERIMENT=93-long-horizon-stability
  export R2_REGIME=c600-frlr-qcad20-fallback
  exec bash "$CELL"
fi

pre "ARM=a9 resolves with the anchor-owned runtime; launching $CELLNAME"
export ARM=a9
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
