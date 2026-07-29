#!/usr/bin/env bash
# Launch issue #93 cell a7: FRLR r48/k28 with NO token-IS. Codec verified
# byte-identical to a5/a5b by diffing the resolved DRY_RUN config; the only
# difference is rollout_is=null. Bar in PREREG_a7.md.
#
# Preflight matters: run_93_cell.sh resolves the ARM at line 41 but re-syncs the
# checkout only later, so an arm added after the box last synced dies instantly.
set -uo pipefail
BRANCH=93-mismatch-control-kit
CELL=examples/grpo_trainer/run_93_cell.sh
STAGED=/workspace/run_93_cell.a7.sh
NEXTLOG=/workspace/runs/a7-frlr-r48k28-notis-200/train.log
mkdir -p "$(dirname "$NEXTLOG")"
pre() { echo "[preflight $(date -u +%H:%M:%SZ)] $*" | tee -a "$NEXTLOG"; }
cd /workspace/verl || { pre "FATAL: /workspace/verl missing"; exit 1; }
synced=no
for i in 1 2 3 4 5; do
  if git fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 \
     && git checkout -B "$BRANCH" FETCH_HEAD >/dev/null 2>&1 \
     && git reset --hard FETCH_HEAD >/dev/null 2>&1; then synced=yes; break; fi
  pre "fetch attempt $i failed, retrying in 30s"; sleep 30
done
pre "checkout synced=$synced HEAD=$(git rev-parse --short HEAD)"
if ! ARM=a7 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1; then
  pre "ARM=a7 does not resolve; falling back to staged $STAGED"
  if [[ -f "$STAGED" ]]; then cp "$STAGED" "$CELL"; chmod +x "$CELL"
  else pre "FATAL: no staged fallback at $STAGED"; exit 1; fi
fi
if ! ARM=a7 DRY_RUN=1 bash "$CELL" >/dev/null 2>&1; then
  pre "FATAL: ARM=a7 still does not resolve. GPU IS IDLE."; exit 1
fi
pre "ARM=a7 resolves; handing over"
set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a
export ARM=a7
export EXPERIMENT_NAME=a7-frlr-r48k28-notis-200
export TOTAL_STEPS=200
export TEST_FREQ=200
export SAVE_FREQ=100
export COMM_EFF_PROBE_EVERY=5
export COMM_EFF_PROBE_CTRL_ENABLED=false
exec bash "$CELL"
