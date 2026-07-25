#!/usr/bin/env bash
# Launch issue #93 cell a6 (incumbent PRF exact-k codec + token-IS + batch
# normalize). Env is the bar pre-registered in PREREG_a6.md; do not edit it
# without amending that file.
set -uo pipefail
cd /workspace/verl || exit 1
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
exec bash examples/grpo_trainer/run_93_cell.sh
