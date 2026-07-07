#!/usr/bin/env bash
# =============================================================================
# bigmath_signed_ema_hpt.sh
#
# HYPERPARAMETER-TUNING ABLATION for the two-circuit comm-eff GRPO method on the
# HARD dataset (Big-Math), at HIGH anchor latency (cadence/delay_K = 20/20),
# WITHOUT weight projection (look-ahead anchor OFF). Sweeps the two signed_ema
# merger knobs — alpha (sign-correction weight) and beta_anc (the M EMA decay).
#
# This is a SEQUENTIAL driver: it preps Big-Math once, sets a single WandB run
# GROUP, then runs each grid cell one-after-another by calling the canonical
# generic comm-eff launcher with per-cell env overrides. It is IDEMPOTENT /
# RESUMABLE (a cell whose runs/<name>/done.flag already exists is skipped).
#
# ---------------------------------------------------------------------------
# THE TWO CIRCUITS (what we are tuning)
#   * Fast circuit:   the normal actor train fwd/bwd, its pipeline-boundary
#                     activations compressed by PowerSGD (rank r=77). Produces
#                     the fast, biased+noisy gradient G_noisy for each matrix.
#   * Anchor circuit: an uncompressed, no-optimizer clone pass from a delay_K-
#                     stale weight snapshot, fired every `cadence` ticks. Its
#                     raw gradient G_anchor is EMA'd into M, and (owns_q) it is
#                     the only thing that updates the PowerSGD basis Q.
#
# THE MERGER (signed_ema) rewrites the fast gradient before optimizer.step():
#       G_corr = alpha * G_noisy  +  (1 - alpha) * |G_noisy| * sign(M)
#   -> MAGNITUDE from the fast compressed grad; SIGN from the (stale but
#      UNCOMPRESSED) anchor EMA M. Compression corrupts small-coord signs; the
#      anchor gives a cleaner sign even though it lags.
#
#   alpha  (COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA), in [0,1]:
#       1.0 -> no correction (G_corr == G_noisy; anchor still builds Q)
#       0.0 -> sign fully dictated by the anchor (max correction)
#       0.25 -> incumbent (25% raw grad + 75% anchor-signed)
#
#   beta_anc (COMM_EFF_SPECTRAL_BETA_ANC), in [0,1] — the EMA that BUILDS M,
#   applied at each anchor FIRE:   M <- beta_anc*M + (1-beta_anc)*G_anchor
#       0.0 -> M = the newest fire's anchor grad (freshest, no memory)
#       0.9 -> M = a long smooth average of past fires (low variance, stalest)
#       0.5 -> incumbent
#
# CADENCE ARITHMETIC (why the run length matters):
#   cadence/delay_K are in OPTIMIZER TICKS. train_batch/mini_batch = 128/64 =>
#   2 ticks/step. 75 steps = 150 ticks. At cadence=20 the anchor fires 7 times
#   (ticks 20,40,...,140) => M is refreshed only 7x and beta_anc blends across
#   those 7 fires. delay_K=20 ticks = ~10 steps of weight staleness. M is cold
#   (merger inert) until the first fire (~step 10), so cells only diverge after
#   ~step 10. This is the "k-collapse" regime (collapse became visible ~step 61
#   in the earlier 100-step study) — 75 steps captures its ONSET.
#
# JUDGED ON (no validation — pure training-curve check, all in one WandB group):
#   critic/rewards/mean (or critic/score/mean), actor/pg_loss, actor/grad_norm
#   (stability), response_length, plus the comm-eff counters bytes_ratio /
#   anchor_backwards / merger_coldM_fallbacks. GRPO is critic-free; "critic"
#   here means the reward/advantage/loss training curve.
#
# ---------------------------------------------------------------------------
# USAGE (on the box, from the verl repo root, inside tmux):
#   tmux new -s hpt
#   bash research/scripts/bigmath_signed_ema_hpt.sh              # run the whole grid
#   DRY_RUN=1 bash research/scripts/bigmath_signed_ema_hpt.sh    # print the grid, do nothing
#   ONLY_CELL=3 bash research/scripts/bigmath_signed_ema_hpt.sh  # run just cell 3
#   START_CELL=4 bash research/scripts/bigmath_signed_ema_hpt.sh # resume from cell 4
#
# OOM fallback (operator directive — shrink response before escalating GPUs):
#   PPO_MAX_TOKEN_LEN_PER_GPU=9216 bash ...     # first lever
#   MAX_RESPONSE_LENGTH=2048 PPO_MAX_TOKEN_LEN_PER_GPU=9216 bash ...   # then this
#
# Prereqs on the box (handled by the verl-research template):
#   1. /workspace/verl checked out (shamanez/verl @ vast-ai-workload)
#   2. verl pip-installed -e .
#   3. ~/.config/verl-research/secrets.env with HF_TOKEN + WANDB_API_KEY
# =============================================================================
set -uo pipefail   # NOT -e: a single cell's benign atexit rc=1 must not abort the grid.

# Activate the box's Python env. Prefer the docker-image-matched `run-verl` venv
# (torch 2.11 / vllm 0.20.2 / transformers 5.3.0 / flash-attn 2.8.3 — built by
# research/scripts/setup_run_verl_env.sh); fall back to /venv/main, then to the
# verl template (where python3 is already correct).
if [[ -f /workspace/venvs/run-verl/bin/activate ]]; then
  source /workspace/venvs/run-verl/bin/activate
elif [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi

# ---- editable knobs ---------------------------------------------------------
VERL_ROOT="${VERL_ROOT:-/workspace/verl}"
DATA_DIR="${DATA_DIR:-/root/data/bigmath}"
LAUNCHER="examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh"

# One WandB run GROUP for the whole sweep (verl leaves wandb group unset, so the
# WANDB_RUN_GROUP env var is honored). Project/entity match the team account.
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-hyperparam_tuning_bigmath_signed_ema_c20d20}"
export WANDB_ENTITY="${WANDB_ENTITY:-shamanework-pl}"
PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"

# Fast, no-validation, curve-only 75-step surface — mirrors the proven EXP-58
# Big-Math 1xH200 surface (resp 4096, dynamic bsz, TP1xN8, mem_util 0.45).
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-75}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"   # anchor clone fits; OOM fallback: 9216
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"

# Fixed comm-eff substrate for EVERY cell (LOCKED; only alpha/beta vary):
#   two-circuit ON, PowerSGD r=77, anchor owns Q, cadence/delay_K=20/20,
#   paired replay, cpu offload for snapshots + M, NO weight projection
#   (lookahead_* left at their OFF defaults), signed_ema merger.
ANCHOR_CADENCE="${ANCHOR_CADENCE:-20}"
ANCHOR_DELAY_K="${ANCHOR_DELAY_K:-20}"

# ---- the grid: "alpha beta_anc" — cross/plus design centered at (0.25, 0.50) --
# Ordered so the highest-signal comparisons land first (incumbent, then the two
# alpha extremes) — you can early-kill after any cell.
GRID=(
  "0.25 0.50"   # 1  incumbent center (GSM8K-tuned) — reference + pipeline/mem canary
  "1.00 0.50"   # 2  merger OFF: raw PowerSGD + stale-anchor-owns-Q, NO sign correction (k-collapse floor)
  "0.00 0.50"   # 3  pure sign-merger: sign FULLY from anchor (max correction)
  "0.50 0.50"   # 4  alpha sweep — interior
  "0.75 0.50"   # 5  alpha sweep — light-touch correction
  "0.25 0.00"   # 6  beta sweep — freshest M (no EMA memory)
  "0.25 0.90"   # 7  beta sweep — smoothest / stalest M
)

# ---- helpers ----------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
ONLY_CELL="${ONLY_CELL:-0}"
START_CELL="${START_CELL:-1}"
FORCE="${FORCE:-0}"

cd "$VERL_ROOT"
SUMMARY="$VERL_ROOT/runs/hpt_bigmath_summary.tsv"
mkdir -p "$(dirname "$SUMMARY")"
[[ -f "$SUMMARY" ]] || printf 'cell\talpha\tbeta_anc\texp_name\tstatus\tlast_step\n' > "$SUMMARY"

banner() {
  echo ""
  echo "============================================================"
  echo "  Big-Math signed_ema HPT  |  group=$WANDB_RUN_GROUP"
  echo "  surface: resp=$MAX_RESPONSE_LENGTH dyn_bsz=True TP=${ROLLOUT_TP}xN=${ROLLOUT_N} mem=$ROLLOUT_GPU_MEM_UTIL max_tok=$PPO_MAX_TOKEN_LEN_PER_GPU steps=$TOTAL_TRAINING_STEPS (NO val, NO ckpt)"
  echo "  anchor: cadence=$ANCHOR_CADENCE delay_K=$ANCHOR_DELAY_K owns_q=true powersgd r=77 | weight projection OFF"
  echo "  grid (alpha beta_anc):"
  local n=0
  for c in "${GRID[@]}"; do n=$((n+1)); printf "    %d) %s\n" "$n" "$c"; done
  echo "============================================================"
}

run_cell() {
  local ALPHA="$1" BETA="$2" EXP="$3"
  # Isolated subshell so per-cell exports never leak between cells.
  (
    export COMM_EFF_ENABLED=true
    export COMM_EFF_COMPRESSION_TYPE=powersgd
    export COMM_EFF_POWERSGD_RANK=77
    export COMM_EFF_ANCHOR_ENABLED=true
    export COMM_EFF_ANCHOR_OWNS_Q=true
    export COMM_EFF_ANCHOR_CADENCE="$ANCHOR_CADENCE"
    export COMM_EFF_ANCHOR_DELAY_K="$ANCHOR_DELAY_K"
    export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH=true
    export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE=cpu
    export COMM_EFF_CLEAN_CADENCE=0
    export COMM_EFF_SPECTRAL_ENABLED=true
    export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
    export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="$ALPHA"
    export COMM_EFF_SPECTRAL_BETA_ANC="$BETA"
    export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
    export COMM_EFF_SPECTRAL_MAX_TARGETS=-1

    export DATA_DIR MAX_RESPONSE_LENGTH MAX_PROMPT_LENGTH USE_DYNAMIC_BSZ=True
    export PPO_MAX_TOKEN_LEN_PER_GPU ROLLOUT_TP ROLLOUT_N ROLLOUT_GPU_MEM_UTIL
    export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ACTOR_LR=1e-6
    export TOTAL_TRAINING_STEPS TOTAL_EPOCHS=1
    export VAL_BEFORE_TRAIN=False TEST_FREQ=100000 SAVE_FREQ=100000
    export PROJECT_NAME EXPERIMENT_NAME="$EXP"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    # diagnostics=false: production speed knob (aggregate WandB counters kept).
    bash "$LAUNCHER" \
      actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false
  )
}

# ---- run --------------------------------------------------------------------
banner
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> nothing launched."; exit 0; fi

# Big-Math prep (idempotent — skipped if both parquets already exist).
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== prepping Big-Math -> $DATA_DIR (train-cap 0 = full 123,602 rows) ==="
  python3 research/scripts/bigmath_dapo.py --local_save_dir "$DATA_DIR" \
    --train-cap 0 --val-size 500 --seed 42
else
  echo "=== Big-Math parquets present in $DATA_DIR (skipping prep) ==="
fi

i=0
for cell in "${GRID[@]}"; do
  i=$((i+1))
  [[ "$ONLY_CELL" != "0" && "$ONLY_CELL" != "$i" ]] && continue
  [[ "$i" -lt "$START_CELL" ]] && continue
  read -r ALPHA BETA <<< "$cell"
  EXP="hpt_bm_a${ALPHA}_b${BETA}"
  DONE_FLAG="$VERL_ROOT/runs/$EXP/done.flag"
  LOG="$VERL_ROOT/runs/$EXP/train.log"

  if [[ "$FORCE" != "1" && -f "$DONE_FLAG" ]]; then
    echo ">>> CELL $i/${#GRID[@]} $EXP already has done.flag — skipping (FORCE=1 to rerun)."
    continue
  fi

  echo ""
  echo "############################################################"
  echo "# CELL $i/${#GRID[@]}: alpha=$ALPHA beta_anc=$BETA  ($EXP)"
  echo "# $(date -u +%FT%TZ)"
  echo "############################################################"
  run_cell "$ALPHA" "$BETA" "$EXP" \
    || echo "WARN: cell $i launcher returned nonzero (often benign atexit rc=1 AFTER success — gating on done.flag below)."

  # Status gate: done.flag, NOT rc (benign atexit noise can rc=1 post-success).
  if [[ -f "$DONE_FLAG" ]]; then STATUS=done; else STATUS=INCOMPLETE; fi
  LAST_STEP=$(grep -oaE "'?global_step'?[:=][0-9]+|step[ :_-]?[0-9]+" "$LOG" 2>/dev/null \
              | grep -oE "[0-9]+" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$ALPHA" "$BETA" "$EXP" "$STATUS" "${LAST_STEP:-NA}" >> "$SUMMARY"
  echo ">>> CELL $i $EXP: status=$STATUS last_step=${LAST_STEP:-NA}"
done

echo ""
echo "=== HPT grid finished at $(date -u +%FT%TZ) ==="
echo "=== summary: $SUMMARY ==="
cat "$SUMMARY"
