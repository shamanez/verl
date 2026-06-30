#!/usr/bin/env bash
# EXP-42 per-REGIME launcher (WEIGHT-projection accuracy study) — runs ON the box.
# Two regimes, STRICTLY sequential: regimeA (plain GRPO) -> regimeB (PowerSGD r=77,
# codec ONLY). Re-materialised 2026-06-29 for the 2-regime single-GPU weight-traj
# design (the old 3-cell grad-proj scaffold is SUPERSEDED — prior gradient study).
#
# The two regimes differ ONLY in the compression keys (comm_eff.enabled,
# compression_type, powersgd.rank, anchor.enabled, spectral.enabled). The
# weight-trajectory instrument (probe.weight_traj.*), the RL surface, model, and
# data are IDENTICAL -> the resolved_params diff across regimes is a SUBSET of the
# allowed compression keys (success criterion: controlled variables identical).
#
# Single-GPU is operator-AUTHORISED for EXP-42 (2026-06-29): ALLOW_SINGLE_GPU=1
# relaxes the 4..8-GPU mandate, ROLLOUT_TP=1. Regime B codec is an IN-GRAPH
# activation projection (M_hat=(M@Q)@Qᵀ) so it fires on 1 GPU (no PP/DP needed) —
# verified post-hoc by theta_A != theta_B + nonzero reconstruction error.
set -uo pipefail
REGIME="${1:?usage: run_cell.sh regimeA|regimeB}"
cd /workspace/verl
[[ -f "$HOME/.verl_auth.env" ]] && source "$HOME/.verl_auth.env" || true
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

case "$REGIME" in
  regimeA)  # plain GRPO — byte-identical dense path (the clean predictability ceiling)
    export COMM_EFF_ENABLED=false
    ;;
  regimeB)  # GRPO + activation compression, CODEC ONLY with ADAPTIVE Q (anchor + spectral OFF)
    export COMM_EFF_ENABLED=true
    export COMM_EFF_COMPRESSION_TYPE=powersgd
    export COMM_EFF_POWERSGD_RANK=77
    export COMM_EFF_ANCHOR_ENABLED=false
    export COMM_EFF_SPECTRAL_ENABLED=false
    # Q-UPDATE IS MANDATORY. owns_q=false = the FAST-OWNED-Q path: the fast hook
    # accumulates V += M^T(MQ) on its own gradient-bearing forwards and runs
    # Q<-orth(V) at powersgd.update_cadence, so the rank-77 basis ADAPTS to the
    # activation subspace. The prior regime B left owns_q=true with the anchor
    # off, which FROZE Q at a random basis (basis_updates=0, recon ~0.97) and
    # collapsed the policy. A frozen-basis codec is not a valid compressed regime.
    export COMM_EFF_ANCHOR_OWNS_Q=false
    ;;
  *) echo "usage: run_cell.sh regimeA|regimeB"; exit 2 ;;
esac

# The observer ALWAYS saves the FULL weight matrices of EVERY floating param (the
# whole model, NOT a sketch, NOT a subset). Knobs (all env-overridable):
#   WEIGHT_TRAJ_PER_TICK  (true|false): true => one snapshot per optimizer TICK
#       (~160 over 80 steps for batch128/mini64); false (default) => one per step.
#   WEIGHT_TRAJ_FULL_DTYPE (bf16|fp32): dump precision. bf16 ~3 GB/snapshot,
#       fp32 ~6 GB/snapshot on Qwen2.5-1.5B.
#   WEIGHT_TRAJ_FULL_EVERY (N): per-STEP-mode cadence (ignored when per_tick).
#   WEIGHT_TRAJ_R2_ENABLED (true|false): upload each snapshot to Cloudflare R2
#       (bucket shamane-pluralis) then delete the local .pt — local disk is staging
#       only. The per-tick bf16 trajectory (~492 GB) does NOT fit the box, so set
#       true for the accepted collection. Creds come from $HOME/.verl_auth.env
#       (R2_ACCOUNT_ID/R2_ENDPOINT/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET).
#   WEIGHT_TRAJ_R2_ASYNC (true|false): OPT-IN async batched upload. false (default)
#       => synchronous (the dump path blocks on each upload — byte-identical to
#       before). true => a background worker pool overlaps uploads with compute;
#       the observer flushes every WEIGHT_TRAJ_R2_FLUSH_EVERY steps + at run end so
#       disk stays bounded (capped at WEIGHT_TRAJ_R2_MAX_STAGED_GB GiB) + fail-loud.
#   WEIGHT_TRAJ_R2_FLUSH_EVERY (N): async flush-barrier cadence in steps (def 10).
#   WEIGHT_TRAJ_R2_WORKERS (N): async upload worker threads (def 4) — parallel
#       aws s3 cp streams approaching the aggregate R2 bandwidth ceiling.
#   WEIGHT_TRAJ_R2_MAX_STAGED_GB (G): async disk-backpressure cap in GiB (def 80).
#   WEIGHT_TRAJ_R2_FLUSH_TIMEOUT (S): async flush/close drain timeout in seconds
#       (def 1800). A slow/hung uploader can't block the step/run-end forever on
#       queue.join(); a timeout raises fail-loud. <=0 => wait forever.
# Separate RUN_DIR keeps the full-weight dumps from clobbering another study and
# names the R2 key prefix (verl-research/<RUN_DIR-basename>/<regime>/weights/...).
RUN_DIR="${RUN_DIR:-/workspace/runs/EXP-42}"
PER_TICK="${WEIGHT_TRAJ_PER_TICK:-false}"
FULL_DTYPE="${WEIGHT_TRAJ_FULL_DTYPE:-bf16}"
FULL_EVERY="${WEIGHT_TRAJ_FULL_EVERY:-1}"
R2_ENABLED="${WEIGHT_TRAJ_R2_ENABLED:-false}"
# Async-upload knobs (default off / values identical to the synchronous path).
R2_ASYNC="${WEIGHT_TRAJ_R2_ASYNC:-false}"
R2_FLUSH_EVERY="${WEIGHT_TRAJ_R2_FLUSH_EVERY:-10}"
R2_WORKERS="${WEIGHT_TRAJ_R2_WORKERS:-4}"
R2_MAX_STAGED_GB="${WEIGHT_TRAJ_R2_MAX_STAGED_GB:-80}"
R2_FLUSH_TIMEOUT="${WEIGHT_TRAJ_R2_FLUSH_TIMEOUT:-1800}"
# R2 object key prefix is verl-research/$R2_EXPERIMENT/$R2_REGIME/weights/...
export R2_EXPERIMENT="${R2_EXPERIMENT:-$(basename "$RUN_DIR")}"
export R2_REGIME="${R2_REGIME:-$REGIME}"
EXPN="exp42-${REGIME}${EXPN_SUFFIX:-}"
OUT="${RUN_DIR}/${REGIME}"
WEIGHTS="$OUT/weights"
mkdir -p "$WEIGHTS"
# main_ppo (incl. [comm_eff][weight_traj], val, response_length) -> *_internal.log
# (matches the plan's grep). Launcher banner + `set -x` resolved command (ground
# truth for resolved_params) -> driver.log.
INTERNAL_LOG="$OUT/train_${REGIME}_internal.log"

echo "=== EXP-42/43 $REGIME: comm_eff.enabled=$COMM_EFF_ENABLED weight_traj=FULL(all-params) per_tick=$PER_TICK dump_dtype=$FULL_DTYPE every_steps=$FULL_EVERY r2_enabled=$R2_ENABLED r2_async=$R2_ASYNC r2_workers=$R2_WORKERS r2_flush_timeout=$R2_FLUSH_TIMEOUT r2_key=verl-research/$R2_EXPERIMENT/$R2_REGIME/weights exp=$EXPN out_dir=$WEIGHTS ==="
LOG="$INTERNAL_LOG" \
ALLOW_SINGLE_GPU=1 \
ROLLOUT_TP=1 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_TRAINING_STEPS=80 \
TEST_FREQ=40 \
USE_DYNAMIC_BSZ=True \
VAL_BEFORE_TRAIN=False \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
EXPERIMENT_NAME="$EXPN" \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.per_tick=$PER_TICK \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.dump_dtype=$FULL_DTYPE \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.every_steps=$FULL_EVERY \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_enabled=$R2_ENABLED \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_async=$R2_ASYNC \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_flush_every_steps=$R2_FLUSH_EVERY \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_upload_workers=$R2_WORKERS \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_max_staged_gb=$R2_MAX_STAGED_GB \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.r2_flush_timeout_s=$R2_FLUSH_TIMEOUT \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.out_dir="$WEIGHTS" \
  > "$OUT/driver.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) done $REGIME rc=$RC" > "$OUT/done.flag"
echo "=== $REGIME finished rc=$RC ; internal_log=$INTERNAL_LOG ==="
exit $RC
