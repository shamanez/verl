#!/usr/bin/env bash
# EXP-58 checkpoint->R2 + fp32 weight-trajectory collection cell — runs ON the box.
#
# One box, two phases (this script runs ONE phase per invocation, chosen by $1):
#   probe       : TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 — the CHEAP correctness gate.
#                 Runs the comm-eff launcher with CKPT_R2 + weight-traj on Big-Math.
#   collection  : TOTAL_TRAINING_STEPS=1000 TOTAL_EPOCHS=2 SAVE_FREQ=20 — the real
#                 multi-hour run that fills R2 with 50 fp32 weight snapshots + 50
#                 full checkpoints, on-the-go (upload-then-delete, disk-bounded).
#
# Deliverable A (fp32 weight trajectory) reuses EXP-57's proven regimeA path
# UNCHANGED: plain GRPO (COMM_EFF_ENABLED=false => dense, anchor+spectral OFF),
# probe.weight_traj.* fp32 every-20 -> R2 weights/ prefix. Deliverable B (full
# checkpoint mirror) is the NEW EXP-58 hook: trainer.checkpoint_r2_enabled=true ->
# every global_step_<N>/ tree mirrored to R2 checkpoints/ prefix, on-the-go.
#
# Both streams share R2_EXPERIMENT/R2_REGIME (=> distinct .../weights vs
# .../checkpoints prefixes, no key collision) and cadence 20. Single-GPU is
# operator-AUTHORISED for this collection (ladder rungs 1-3): ALLOW_SINGLE_GPU=1,
# ROLLOUT_TP=<gpu_count>. FSDP1 only (strategy=fsdp), use_orig_params=true.
#
# Gate success on ARTIFACT COMPLETENESS, not launcher rc (benign atexit rc=1 is
# expected even after full success).
set -uo pipefail
PHASE="${1:?usage: ckpt_r2_collection_cell.sh probe|collection}"
cd /workspace/verl

# R2 creds/prefix + HF/WandB from the box auth env (NEVER echoed).
[[ -f "$HOME/.verl_auth.env" ]] && source "$HOME/.verl_auth.env" || true
[[ -f "$HOME/.config/verl-research/secrets.env" ]] && source "$HOME/.config/verl-research/secrets.env" || true

# ---- dataset: Big-Math (data_source => DigitalLearningGmbH/MATH-lighteval) -----
export DATA_DIR="${DATA_DIR:-/root/data/bigmath}"

# ---- R2 object-key prefixes (BOTH weights/ and checkpoints/ hang off these) ----
export R2_EXPERIMENT="${R2_EXPERIMENT:-EXP-58}"
export R2_REGIME="${R2_REGIME:-regimeA}"

# ---- method: plain GRPO dense path (anchor OFF per EXP-58 resolved default) -----
# COMM_EFF_ENABLED=false => byte-identical dense verl training; the fp32 weight
# trajectory it produces is EXP-57's regimeA trajectory (Deliverable A unchanged).
export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-false}"

# ---- hardware / FSDP1 (single-GPU rungs 1-3 auto-safe for the fp32 summon) ------
export ALLOW_SINGLE_GPU="${ALLOW_SINGLE_GPU:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"                 # clamped to detected GPU count by the launcher
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# ---- checkpoint -> R2 (Deliverable B; the NEW hook) -----------------------------
# For the probe's byte-parity leg the CALLER sets CKPT_R2_ENABLED unset/false; for
# the on-the-go leg + the collection run it is true. Defaults here = the ON recipe.
export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-true}"
export CKPT_R2_ASYNC="${CKPT_R2_ASYNC:-true}"
export CKPT_R2_DELETE_LOCAL="${CKPT_R2_DELETE_LOCAL:-true}"
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-50}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-4}"

# ---- fp32 weight trajectory (Deliverable A; unchanged from EXP-57) --------------
export WEIGHT_TRAJ_FULL_DTYPE="${WEIGHT_TRAJ_FULL_DTYPE:-fp32}"
export WEIGHT_TRAJ_PER_TICK="${WEIGHT_TRAJ_PER_TICK:-false}"
export WEIGHT_TRAJ_FULL_EVERY="${WEIGHT_TRAJ_FULL_EVERY:-20}"
export WEIGHT_TRAJ_R2_ENABLED="${WEIGHT_TRAJ_R2_ENABLED:-true}"
export WEIGHT_TRAJ_R2_ASYNC="${WEIGHT_TRAJ_R2_ASYNC:-true}"
export WEIGHT_TRAJ_R2_FLUSH_EVERY="${WEIGHT_TRAJ_R2_FLUSH_EVERY:-20}"
export WEIGHT_TRAJ_R2_WORKERS="${WEIGHT_TRAJ_R2_WORKERS:-4}"
export WEIGHT_TRAJ_R2_MAX_STAGED_GB="${WEIGHT_TRAJ_R2_MAX_STAGED_GB:-80}"
export WEIGHT_TRAJ_R2_FLUSH_TIMEOUT="${WEIGHT_TRAJ_R2_FLUSH_TIMEOUT:-1800}"

# ---- run schedule (phase-dependent) ---------------------------------------------
case "$PHASE" in
  probe)
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
    export SAVE_FREQ="${SAVE_FREQ:-1}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
    export TEST_FREQ="${TEST_FREQ:-100}"          # >steps => no mid-probe val
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    RUN_DIR="${RUN_DIR:-/workspace/runs/EXP-58/probe-${PROBE_TAG:-on}}"
    EXPN="${EXPERIMENT_NAME:-exp-58-probe-${PROBE_TAG:-on}}"
    ;;
  collection)
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
    export SAVE_FREQ="${SAVE_FREQ:-20}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"       # LOAD-BEARING: 965 steps/epoch < 1000 => need 2
    export TEST_FREQ="${TEST_FREQ:-9999}"          # collection run: skip mid-run val (telemetry-only)
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    RUN_DIR="${RUN_DIR:-/workspace/runs/EXP-58/collection}"
    EXPN="${EXPERIMENT_NAME:-exp-58-bigmath-1000step}"
    ;;
  *) echo "usage: ckpt_r2_collection_cell.sh probe|collection" >&2; exit 2 ;;
esac

export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="$EXPN"
OUT="$RUN_DIR"
WEIGHTS="$OUT/weights"          # weight-traj observer out_dir (local staging; R2 mirror is authoritative)
mkdir -p "$WEIGHTS"
INTERNAL_LOG="$OUT/train_${PHASE}_internal.log"
export LOG="$INTERNAL_LOG"

echo "=== EXP-58 $PHASE: comm_eff.enabled=$COMM_EFF_ENABLED steps=$TOTAL_TRAINING_STEPS epochs=$TOTAL_EPOCHS save_freq=$SAVE_FREQ resp=$MAX_RESPONSE_LENGTH data=$DATA_DIR"
echo "===         CKPT_R2_ENABLED=$CKPT_R2_ENABLED async=$CKPT_R2_ASYNC delete_local=$CKPT_R2_DELETE_LOCAL workers=$CKPT_R2_WORKERS max_staged_gb=$CKPT_R2_MAX_STAGED_GB"
echo "===         WEIGHT_TRAJ dtype=$WEIGHT_TRAJ_FULL_DTYPE every=$WEIGHT_TRAJ_FULL_EVERY r2=$WEIGHT_TRAJ_R2_ENABLED"
echo "===         R2 prefixes: verl-research/$R2_EXPERIMENT/$R2_REGIME/{weights,checkpoints}  exp=$EXPN out=$WEIGHTS"

# The comm-eff launcher exports CKPT_R2_* + threads trainer.checkpoint_r2_enabled
# through to main_ppo; we pass the probe.weight_traj.* Hydra keys + let the sink
# own local ckpt deletes (max_actor_ckpt_to_keep=null). driver.log holds the
# `set -x` resolved main_ppo command (ground truth for resolved_params).
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  trainer.max_actor_ckpt_to_keep=null \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.per_tick="$WEIGHT_TRAJ_PER_TICK" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.dump_dtype="$WEIGHT_TRAJ_FULL_DTYPE" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.every_steps="$WEIGHT_TRAJ_FULL_EVERY" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_enabled="$WEIGHT_TRAJ_R2_ENABLED" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_async="$WEIGHT_TRAJ_R2_ASYNC" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_flush_every_steps="$WEIGHT_TRAJ_R2_FLUSH_EVERY" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_upload_workers="$WEIGHT_TRAJ_R2_WORKERS" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_max_staged_gb="$WEIGHT_TRAJ_R2_MAX_STAGED_GB" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_flush_timeout_s="$WEIGHT_TRAJ_R2_FLUSH_TIMEOUT" \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.out_dir="$WEIGHTS" \
  "$@" \
  > "$OUT/driver.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) done $PHASE rc=$RC" > "$OUT/done.flag"
echo "=== EXP-58 $PHASE finished rc=$RC ; internal_log=$INTERNAL_LOG driver_log=$OUT/driver.log ==="
exit $RC
