#!/usr/bin/env bash
# EXP-3 in-container launcher (rsync'd to /workspace/runs/EXP-3/launch.sh).
#
# code_change=false: we do NOT apply any bundle. The template's onstart has
# already cloned shamanez/verl @ vast-ai-workload into /workspace/verl and
# pip-installed verl --no-deps editable. We `git pull` to pick up any
# launcher edits since the image was built, then exec the committed baseline
# launcher verbatim — its argv & env vars own the training config.
#
# The committed launcher writes its own log + metrics under
# /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/ (train.log, done.flag).
# We also tee /workspace/train.log so the harness liveness check works
# without knowing the launcher's internal path.
set -euo pipefail

cd /workspace/verl
git pull --ff-only origin vast-ai-workload

# Hand off to the committed launcher. It sources ~/.config/verl-research/secrets.env,
# probes GPU count + cgroup pids.max, fixes all training knobs, and runs
# run_qwen3_4b_fsdp.sh under tee so the run log lives under
# /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/train.log. It touches
# done.flag on clean exit.
exec bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh
