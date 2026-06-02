#!/usr/bin/env bash
# EXP-18 / M4 candidate C1 — stale-anchor additive injection (correction_mode=inject).
# Runs INSIDE the reused Vast.ai box (instance 39132674, 4xH200). The box's
# /workspace/verl currently holds the spectral-floor tree; this script REPLACES it
# with the exp/18-anchorinject-c5d5 branch shipped as exp-c1.bundle, smoke-tests the
# new inject path, then launches the 50-step C1 training.
#
# Lifecycle: this box's EXP-18 ledger row stays RUNNING across all cells (it owns the
# rental). This script does NOT provision and does NOT touch the ledger.
set -uo pipefail   # NOT -e: we want to capture rc and always write the done flag

RUN_DIR=/workspace/runs/EXP-18
LOG=/workspace/runs/EXP-18/train_curvematch_anchorinject_c5_d5.log
DONE_FLAG=/workspace/runs/EXP-18/done_curvematch_anchorinject_c5_d5.flag

mkdir -p "$RUN_DIR"

# Configure git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local" || true
git config --global user.name  "verl-research-harness" || true

echo "$(date -Iseconds) [launch_c1] starting C1 anchorinject on $(hostname)"

# --- (a) kill any stale tmux of the same name (defensive). We are usually running
#         INSIDE the target session, so only kill OTHER sessions of this name —
#         never self (tmux refuses to kill the attached session anyway). ---------
SELF_SESSION="$(tmux display-message -p '#S' 2>/dev/null || true)"
for s in $(tmux ls 2>/dev/null | awk -F: '/^exp-18-208_64_254_75/ {print $1}'); do
  if [[ "$s" != "$SELF_SESSION" ]]; then
    echo "[launch_c1] killing stale tmux session: $s"
    tmux kill-session -t "$s" 2>/dev/null || true
  fi
done

# --- (b) apply the exp-c1.bundle: replace /workspace/verl ----------------------
cd /workspace
if [[ ! -f "$RUN_DIR/exp-c1.bundle" ]]; then
  echo "[launch_c1] FATAL: $RUN_DIR/exp-c1.bundle missing — cannot apply C1 patch" >&2
  echo "$(date -Iseconds) rc=90 (bundle missing)" > "$DONE_FLAG"
  exit 90
fi
# NOTE: `git bundle verify` needs an enclosing git repo (we are in /workspace,
# which is not one) so it would falsely fail here. `git bundle list-heads` works
# standalone and proves the expected branch ref is present + the header is sane;
# the `git clone` below is the real integrity gate (it fails loudly on a corrupt
# pack). The bundle was verified complete on the laptop before rsync.
if ! git bundle list-heads "$RUN_DIR/exp-c1.bundle" 2>/dev/null | grep -q "refs/heads/exp/18-anchorinject-c5d5"; then
  echo "[launch_c1] FATAL: exp-c1.bundle missing expected ref refs/heads/exp/18-anchorinject-c5d5" >&2
  echo "$(date -Iseconds) rc=91 (bundle corrupt)" > "$DONE_FLAG"
  exit 91
fi
if [[ -d /workspace/verl ]]; then
  BAK="/workspace/verl.bak-$(date +%s)"
  echo "[launch_c1] preserving existing tree -> $BAK"
  mv /workspace/verl "$BAK"
fi
echo "[launch_c1] cloning exp/18-anchorinject-c5d5 from bundle"
if ! git clone -b exp/18-anchorinject-c5d5 "$RUN_DIR/exp-c1.bundle" /workspace/verl; then
  echo "[launch_c1] FATAL: git clone from bundle failed" >&2
  echo "$(date -Iseconds) rc=92 (clone failed)" > "$DONE_FLAG"
  exit 92
fi
cd /workspace/verl
# Point origin at the fork so any in-container push (commit-hotfix.sh) goes to the
# right repo and the right branch.
git remote set-url origin https://github.com/shamanez/verl.git || true
echo "[launch_c1] HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# Editable (re)install. verl is installed editable pointing at /workspace/verl
# (the template's onstart did `pip install -e .` once), so swapping the source
# tree in-place — which the bundle clone above just did — ALREADY makes the new
# code live; the .pth dispatch picks it up with no reinstall. We still attempt a
# best-effort editable reinstall (prefer uv, fall back to pip) to be safe, but a
# reinstall failure is NOT fatal as long as the new inject code is importable
# from the live verl (asserted right after). This is what makes the run robust
# to `uv` being absent on the box (it is) and to the editable-install already
# being correct.
echo "[launch_c1] editable (re)install — prefer uv, fall back to pip (log: $RUN_DIR/pip_c1.log)"
INSTALL_OK=0
if command -v uv >/dev/null 2>&1; then
  echo "[launch_c1] using: uv pip install --no-deps -e ."
  uv pip install --no-deps -e . > "$RUN_DIR/pip_c1.log" 2>&1 && INSTALL_OK=1
elif command -v pip >/dev/null 2>&1; then
  echo "[launch_c1] uv not found — using: pip install --no-deps -e ."
  pip install --no-deps -e . > "$RUN_DIR/pip_c1.log" 2>&1 && INSTALL_OK=1
else
  echo "[launch_c1] neither uv nor pip found; relying on the live editable install" >> "$RUN_DIR/pip_c1.log"
fi
# The REAL gate: the inject code must be importable from the live verl. The
# editable .pth points at /workspace/verl (now the exp-branch tree), so this
# holds even when the reinstall above was a no-op or failed.
if ! python -c "import verl, os; from verl.workers.comm_eff.spectral_filter import SpectralFilter; assert hasattr(SpectralFilter, 'inject_matrix'), 'inject_matrix missing'; assert os.path.realpath(verl.__file__).startswith('/workspace/verl/'), ('verl not from /workspace/verl: '+verl.__file__)" >> "$RUN_DIR/pip_c1.log" 2>&1; then
  echo "[launch_c1] FATAL: live verl lacks inject code or points off /workspace/verl (install_ok=$INSTALL_OK); tail of pip_c1.log:" >&2
  tail -40 "$RUN_DIR/pip_c1.log" >&2 || true
  echo "$(date -Iseconds) rc=93 (verl inject import failed)" > "$DONE_FLAG"
  exit 93
fi
echo "[launch_c1] editable install verified (install_ok=$INSTALL_OK; inject code live from /workspace/verl)"

# --- (c) inject-path smoke (ABORT on failure — catches a broken patch BEFORE we
#         burn GPU minutes on a 50-step run) --------------------------------------
echo "[launch_c1] running inject smoke"
SMOKE_OUT=$(python -c "import torch; from verl.workers.comm_eff.spectral_filter import SpectralFilter; sf=SpectralFilter(correction_mode='inject', inject_gamma=1.0); sf.update_anchor('w', torch.randn(8,8)); print('inject shape', tuple(sf.inject_matrix('w', torch.randn(8,8)).shape))" 2>&1)
SMOKE_RC=$?
echo "$SMOKE_OUT"
if [[ $SMOKE_RC -ne 0 ]] || ! echo "$SMOKE_OUT" | grep -q "inject shape (8, 8)"; then
  echo "[launch_c1] FATAL: inject smoke failed (rc=$SMOKE_RC) — aborting before training" >&2
  echo "$(date -Iseconds) rc=94 (inject smoke failed)" > "$DONE_FLAG"
  exit 94
fi
echo "[launch_c1] inject smoke OK"

# --- (d) launch the 50-step C1 training ---------------------------------------
# VERBATIM env from the dispatch: 18432 anchor-OOM fix + inject knobs.
# MANDATORY pins (run is INVALID if violated): ANCHOR_DELAY_K=5 (launcher default
# is 20!), CLEAN_CADENCE=0, ANCHOR_CADENCE=5. MAX_RESPONSE_LENGTH (16384) untouched.
echo "[launch_c1] launching 50-step C1 training (correction_mode=inject, gamma=1.0)"
cd /workspace/verl
PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorinject_c5_d5 \
COMM_EFF_ENABLED=true \
COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True \
NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=inject \
  actor_rollout_ref.actor.comm_eff.spectral.inject_gamma=1.0
TRAIN_RC=$?

echo "$(date -Iseconds) rc=$TRAIN_RC" > "$DONE_FLAG"
echo "[launch_c1] training exited rc=$TRAIN_RC; wrote $DONE_FLAG"
exit $TRAIN_RC
