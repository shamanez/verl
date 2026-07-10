#!/usr/bin/env bash
# onstart for the verl-research vast.ai template.
#
# Runs inside the verlai/verl:vllm020.dev1 container at instance start, before
# SSH is routable. Idempotent — safe on every container restart.
#
# Contract:
#  - PRESERVE the image's bundled torch/vllm/megatron/transformer_engine/deepep
#    versions. They were CI-validated together by verlai; pip MUST NOT touch
#    them. The `--no-deps` flag is what enforces that.
#  - Clone shamanez/verl @ autonomous-harness-v1 so the box ships with the fork's
#    vast.ai-specific launchers (examples/grpo_trainer/vast_*.sh). The "edit
#    locally, push, git pull on the box" iteration loop is the whole point —
#    NO scp'd files, NO /tmp scripts. See examples/grpo_trainer/VAST_README.md.

set -euo pipefail

VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/shamanez/verl.git}"
VERL_REPO_BRANCH="${VERL_REPO_BRANCH:-autonomous-harness-v1}"

ONSTART_LOG=/var/log/onstart.log
exec > >(tee -a "$ONSTART_LOG") 2>&1

echo "[onstart] === verl-research onstart starting at $(date -u +%FT%TZ) ==="
echo "[onstart] repo: $VERL_REPO_URL @ $VERL_REPO_BRANCH"

# Pin record so we can detect drift later.
BEFORE_VLLM=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "missing")
BEFORE_TORCH=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "missing")
echo "[onstart] bundled before install: vllm=$BEFORE_VLLM torch=$BEFORE_TORCH"

# Idempotency: if verl is already pip-installed from /workspace/verl, just
# refresh the checkout (cheap) and exit clean. This handles container
# restarts and lets the box catch up with new pushes on every reboot.
if python3 -c 'import verl, os, sys; sys.exit(0 if "/workspace/verl/" in verl.__file__ else 1)' 2>/dev/null; then
  echo "[onstart] verl already installed editable from /workspace/verl — refreshing"
  cd /workspace/verl && git fetch origin "$VERL_REPO_BRANCH" --quiet \
    && git checkout "$VERL_REPO_BRANCH" --quiet \
    && git pull --ff-only --quiet || echo "[onstart] (refresh skipped: $?)"
  exit 0
fi

# 1. Clone shamanez/verl at the autonomous-harness-v1 branch.
mkdir -p /workspace
cd /workspace
if [[ ! -d verl/.git ]]; then
  echo "[onstart] cloning $VERL_REPO_URL @ $VERL_REPO_BRANCH into /workspace/verl"
  git clone --depth 1 --branch "$VERL_REPO_BRANCH" "$VERL_REPO_URL"
else
  echo "[onstart] /workspace/verl/.git exists; switching to $VERL_REPO_BRANCH and pulling"
  cd verl
  git remote set-url origin "$VERL_REPO_URL"
  git fetch origin "$VERL_REPO_BRANCH" --depth 1 --quiet
  git checkout "$VERL_REPO_BRANCH" --quiet
  git pull --ff-only --quiet
  cd ..
fi
cd verl

# 2. pip install --no-deps -e .
#    --no-deps: do not touch torch/vllm/megatron/te/deepep that the image ships.
#    -e .     : editable install so the harness can `git fetch && git checkout exp/<ID>`.
echo "[onstart] pip install --no-deps -e ."
pip install --no-deps -e . 2>&1 | tail -20

# 3. Verify the install didn't disturb the bundled stack. Hard-fail otherwise —
#    a drift here would silently break vllm rollouts at training time.
AFTER_VLLM=$(python3 -c 'import vllm; print(vllm.__version__)')
AFTER_TORCH=$(python3 -c 'import torch; print(torch.__version__)')
echo "[onstart] bundled after install:  vllm=$AFTER_VLLM torch=$AFTER_TORCH"

if [[ "$AFTER_VLLM" != "$BEFORE_VLLM" || "$AFTER_TORCH" != "$BEFORE_TORCH" ]]; then
  echo "[onstart] FATAL: bundled stack drifted during pip install" >&2
  echo "[onstart] vllm:  $BEFORE_VLLM -> $AFTER_VLLM"            >&2
  echo "[onstart] torch: $BEFORE_TORCH -> $AFTER_TORCH"          >&2
  exit 1
fi

# 4. Final smoke — import path, entrypoint reachable.
python3 -c 'import verl; print("[onstart] verl at", verl.__file__, "version", getattr(verl, "__version__", "?"))'
python3 -c 'from verl.trainer import main_ppo; print("[onstart] verl.trainer.main_ppo OK")'

echo "[onstart] === done at $(date -u +%FT%TZ); SSH will become routable shortly ==="
