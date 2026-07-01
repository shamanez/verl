#!/usr/bin/env bash
# EXP-58 bootstrap — runs inside the Vast.ai container ONCE before the probe.
# The template onstart already cloned shamanez/verl@vast-ai-workload into
# /workspace/verl + pip-installed it. code_change=true => replace that tree with
# the exp/58-ckpt-r2 branch from the shipped bundle, then prep Big-Math. The
# probe + collection PHASES are launched separately (research/scripts/
# ckpt_r2_collection_cell.sh probe|collection) so the runner can gate between them.
set -euo pipefail
cd /workspace/runs/EXP-58

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---- apply the experimental branch from the bundle (code_change=true) ----------
if [[ -f exp.bundle ]]; then
  cd /workspace
  if [[ -d verl && ! -d verl.upstream-vast-ai-workload ]]; then
    mv verl verl.upstream-vast-ai-workload      # preserve template-installed tree
  fi
  rm -rf verl
  git clone -b "exp/58-ckpt-r2" exp.bundle verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  echo "=== pip install -e . --no-deps (exp/58-ckpt-r2) ==="
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || pip install --no-deps -e . > /workspace/pip.log 2>&1
  echo "=== verl now at: $(git -C /workspace/verl rev-parse --short HEAD) branch=$(git -C /workspace/verl rev-parse --abbrev-ref HEAD) ==="
fi

# ---- copy the collection cell script into /workspace/verl (it lives under
#      research/ on the branch, already present after the clone) -----------------
ls -la /workspace/verl/research/scripts/ckpt_r2_collection_cell.sh

# ---- Big-Math dataset prep (idempotent; skip if already present) ---------------
DATA_DIR=/root/data/bigmath
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== prep Big-Math -> $DATA_DIR ==="
  cd /workspace/verl
  python3 research/scripts/bigmath_dapo.py --local_save_dir "$DATA_DIR" --train-cap 0 --val-size 500 --seed 42
fi
ls -la "$DATA_DIR"/*.parquet
echo "=== EXP-58 bootstrap complete at $(date -u +%FT%TZ) ==="
