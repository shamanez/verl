#!/usr/bin/env bash
# Launch issue #93 cell a6 (incumbent PRF exact-k codec + token-IS + batch
# normalize). Env is the bar pre-registered in PREREG_a6.md; do not edit it
# without amending that file.
#
# PREFLIGHT MATTERS: run_93_cell.sh resolves the ARM at its line 41 but only
# re-syncs the checkout at its line 243, so an ARM the on-box checkout does not
# yet know about dies instantly, BEFORE the fetch that would have taught it.
# Arm a6 landed on the branch after this box last synced, so the checkout must
# be brought forward here, and the arm proved resolvable, before handing over.
set -uo pipefail

BRANCH=93-mismatch-control-kit
CELL=examples/grpo_trainer/run_93_cell.sh
STAGED=/workspace/run_93_cell.a6.sh
NEXTLOG=/workspace/runs/a6-prf-exactk-tis-bnorm-200/train.log

mkdir -p "$(dirname "$NEXTLOG")"
pre() { echo "[preflight $(date -u +%H:%M:%SZ)] $*" | tee -a "$NEXTLOG"; }

cd /workspace/verl || { pre "FATAL: /workspace/verl missing"; exit 1; }

# 1. Bring the checkout forward. Retry: a single transient network blip must not
#    cost a night of GPU time.
synced=no
for i in 1 2 3 4 5; do
  if git fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 \
     && git checkout -B "$BRANCH" FETCH_HEAD >/dev/null 2>&1 \
     && git reset --hard FETCH_HEAD >/dev/null 2>&1; then
    synced=yes; break
  fi
  pre "fetch attempt $i failed, retrying in 30s"
  sleep 30
done
pre "checkout synced=$synced HEAD=$(git rev-parse --short HEAD)"

# 2. Prove the arm resolves. If the fetch never landed, fall back to the copy
#    staged on the box while a5b was still running.
if ! ARM=a6 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1; then
  pre "ARM=a6 does not resolve in this checkout; falling back to staged $STAGED"
  if [[ -f "$STAGED" ]]; then
    cp "$STAGED" "$CELL"
    chmod +x "$CELL"
  else
    pre "FATAL: no staged fallback at $STAGED"; exit 1
  fi
fi
if ! ARM=a6 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1; then
  pre "FATAL: ARM=a6 still does not resolve after fallback. GPU IS IDLE."
  exit 1
fi
pre "ARM=a6 resolves; handing over to the cell launcher"

# 3. Launch.
set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a
export ARM=a6
export EXPERIMENT_NAME=a6-prf-exactk-tis-bnorm-200
export TOTAL_STEPS=200
export TEST_FREQ=200
export SAVE_FREQ=100
export COMM_EFF_PROBE_EVERY=25
export COMM_EFF_PROBE_CTRL_ENABLED=false
export ROLLOUT_IS_BATCH_NORMALIZE=true
exec bash "$CELL"
