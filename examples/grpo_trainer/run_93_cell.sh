#!/usr/bin/env bash
# run_93_cell.sh
# One cell of the issue #93 long-horizon stability run matrix, launched with
# the #90 protocol (run_prf_exactk_600.sh): Qwen2.5-Math-1.5B on MATH,
# batch 128 / mini 128 (one on-policy optimizer tick per step), 1024/2048,
# pp=8, LR 1e-6, ONE H200. WandB project derives from
# WANDB_RUN_GROUP=93-long-horizon-stability; each cell's run is <arm>-<slug>.
#
# Usage:  ARM=<cell> bash run_93_cell.sh     (or: bash run_93_cell.sh <cell>)
#
# Cells (issue #93 sections 6 and 8):
#   a1  sr_quant bits=1 block=32 rounding=sr           (120 steps, val off)
#   a2  sr_quant bits=1 block=32 rounding=rn           (bias control)
#   a3  sr_quant bits=2 block=32 subset_k=493          (byte-parity hybrid)
#   a4  prf_mask exact-k + CVC CE mode lambda=0.003    (make-it-settle arm)
#   a5  frlr rank=48 k=28 + decoupled token-IS 2.0     (coherent+corrected)
#   b1  CODEC_ARM=<a1..a5> + dense-view probe(25) + KL controller (200 steps)
#   c   CODEC_ARM=<a1..a5> + probe/controller + 600 steps, val 0/150/300/450/
#       600, SAVE_FREQ=100, R2 checkpoint sink on
#
# b1 and c REQUIRE COMM_EFF_PROBE_KL_TARGET_TABLE in the env (the controller
# setpoint curve baked from the finished 90-dense-600 reference-KL history).
# DRY_RUN=1 resolves + echoes the cell config and exits before any bring-up.
#
# Rounds A/B: validation OFF (VAL_BEFORE_TRAIN=False, TEST_FREQ=-1), no
# checkpoints (SAVE_FREQ=-1); the free training reward slope is the guard.
# Run inside tmux. The engine tees training to $LOG (the heartbeat log).
set -uo pipefail

fatal() { echo "FATAL: $*" >&2; exit 1; }

# -------- knobs you may edit --------
ARM="${ARM:-${1:-}}"
BRANCH="${BRANCH:-93-mismatch-control-kit}"     # all #93 program code lands here
REPO="${REPO:-https://github.com/shamanez/verl.git}"
WORK="${WORK:-/workspace}"
GPU_MEM="${GPU_MEM:-0.72}"                      # the #90 rollout KV-cache setting
RUN_GROUP="${WANDB_RUN_GROUP:-93-long-horizon-stability}"
# ------------------------------------

[[ -n "$ARM" ]] || fatal "no ARM given. Usage: ARM=<a1|a2|a3|a4|a5|b1|c> bash run_93_cell.sh"

# ---------------------------------------------------------------------------
# 1. Resolve the arm BEFORE any bring-up: fail loud on an unknown ARM, echo
#    the full resolved config, and let DRY_RUN=1 stop here (launcher lint).
# ---------------------------------------------------------------------------

# Codec configs (the a-cells; b1/c reuse them via CODEC_ARM). Every arm is the
# #90 comm-eff surface with only the codec knobs changed; the anchor circuit
# stays at its defaults (issue #93 4.8: anchor unchanged in phase 1).
apply_codec_arm() {
  local codec="$1"
  export COMM_EFF_ENABLED=true
  # No boundary codec carries a PowerSGD basis Q in this matrix.
  export COMM_EFF_ANCHOR_OWNS_Q=false
  export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false
  export COMM_EFF_MASK_RECOMPUTE=true
  export COMM_EFF_MASK_REFERENCE=true
  export COMM_EFF_MASK_PP_SIZE=8
  case "$codec" in
    a1|a2)
      export COMM_EFF_COMPRESSION_TYPE=sr_quant
      export COMM_EFF_QUANT_BITS=1
      export COMM_EFF_QUANT_BLOCK_SIZE=32
      export COMM_EFF_QUANT_SUBSET_K=0
      if [[ "$codec" == "a2" ]]; then
        export COMM_EFF_QUANT_ROUNDING=rn      # biased round-to-nearest control
        CODEC_SLUG="srq-b1-rn"
      else
        export COMM_EFF_QUANT_ROUNDING=sr
        CODEC_SLUG="srq-b1-sr"
      fi
      ;;
    a3)
      # Byte-parity hybrid (issue #93 4.3): 2-bit SR on a PRF-fresh exact-493
      # subset, H/k rescale. 493*2 + 493*16/32 = 1232.5 -> 1233 bits/token/
      # boundary vs the prf exact-k incumbent's 77*16 = 1232.
      export COMM_EFF_COMPRESSION_TYPE=sr_quant
      export COMM_EFF_QUANT_BITS=2
      export COMM_EFF_QUANT_BLOCK_SIZE=32
      export COMM_EFF_QUANT_ROUNDING=sr
      export COMM_EFF_QUANT_SUBSET_K=493
      CODEC_SLUG="srq-parity-k493"
      ;;
    a4)
      # The #89/#90 incumbent PRF exact-k codec + CVC CE mode (issue #93 4.7a).
      export COMM_EFF_COMPRESSION_TYPE=prf_mask
      export COMM_EFF_MASK_ENABLED=true
      export COMM_EFF_MASK_P=0.95
      export COMM_EFF_MASK_RESCALE_MODE=constant
      export COMM_EFF_MASK_EXACT_K=true
      export COMM_EFF_CVC_LAMBDA=0.003
      export COMM_EFF_CVC_WARMUP_STEPS=20
      CODEC_SLUG="prf-exactk-cvc-ce"
      ;;
    a5)
      # FRLR r=48 k=28 + decoupled token-IS (issue #93 4.4); token-IS is
      # exclusive to this arm (measured dead on the PRF/sr_quant views).
      export COMM_EFF_COMPRESSION_TYPE=prf_mask
      export COMM_EFF_MASK_ENABLED=true
      export COMM_EFF_MASK_FRLR=true
      export COMM_EFF_MASK_FRLR_RANK=48
      export COMM_EFF_MASK_FRLR_K=28
      export COMM_EFF_MASK_RESCALE=false
      export COMM_EFF_MASK_RESCALE_MODE=auto   # frlr does its own norm matching
      export ROLLOUT_IS=token
      export ROLLOUT_IS_THRESHOLD=2.0
      CODEC_SLUG="frlr-r48k28-tis"
      ;;
    *)
      fatal "unknown CODEC_ARM '$codec' (a1|a2|a3|a4|a5)"
      ;;
  esac
}

# Control plane for b1/c: probe + adaptive KL coefficient (issue #93 I3). The
# setpoint table is REQUIRED from the env; running the controller floor-only
# would silently change the round-B/C science.
apply_control_plane() {
  [[ -n "${COMM_EFF_PROBE_KL_TARGET_TABLE:-}" ]] \
    || fatal "$ARM requires COMM_EFF_PROBE_KL_TARGET_TABLE in the env ('step:value,...' baked from the 90-dense-600 reference-KL curve)"
  export COMM_EFF_PROBE_KL_TARGET_TABLE
  # Parameterized 2026-07-25, default unchanged. At 200 steps a cadence of 25
  # yields only 8 probes, which leaves the window=4 brake detector a single
  # evaluable point; round B should pass COMM_EFF_PROBE_EVERY=5 for 40 probes.
  export COMM_EFF_PROBE_EVERY="${COMM_EFF_PROBE_EVERY:-25}"
  export COMM_EFF_PROBE_CTRL_ENABLED=true
}

case "$ARM" in
  a1|a2|a3|a4|a5)
    apply_codec_arm "$ARM"
    TOTAL_STEPS="${TOTAL_STEPS:-120}"
    # Default -1 preserves the registered "validation OFF for all gate cells".
    # Passing TEST_FREQ=$TOTAL_STEPS gives a terminal-step val that cannot
    # perturb training, since no training step follows it. That is an operator
    # decision (it amends the matrix and costs about 0.2 GPU-h), not a default.
    TEST_FREQ="${TEST_FREQ:--1}"
    VAL_BEFORE_TRAIN=False
    # Default -1 preserves the registered "no checkpoints in rounds A/B" disk
    # lesson. Round A saving nothing meant none of its five arms can ever be
    # re-analysed, so a probe cell that wants post-hoc geometry must opt in.
    SAVE_FREQ="${SAVE_FREQ:--1}"
    EXPERIMENT_NAME="${ARM}-${CODEC_SLUG}"
    ;;
  b1)
    [[ -n "${CODEC_ARM:-}" ]] || fatal "ARM=b1 requires CODEC_ARM=<a1|a2|a3|a4|a5> (the round-A winner codec)"
    apply_codec_arm "$CODEC_ARM"
    apply_control_plane
    TOTAL_STEPS="${TOTAL_STEPS:-200}"
    TEST_FREQ=-1
    VAL_BEFORE_TRAIN=False
    SAVE_FREQ=-1
    EXPERIMENT_NAME="b1-${CODEC_ARM}-${CODEC_SLUG}-ctrl"
    ;;
  c)
    [[ -n "${CODEC_ARM:-}" ]] || fatal "ARM=c requires CODEC_ARM=<a1|a2|a3|a4|a5> (the round-B winner codec)"
    apply_codec_arm "$CODEC_ARM"
    apply_control_plane
    TOTAL_STEPS="${TOTAL_STEPS:-600}"
    TEST_FREQ="${TEST_FREQ:-300}"        # val at 0/300/600 (operator, 2026-07-25); override via TEST_FREQ
    VAL_BEFORE_TRAIN=True
    SAVE_FREQ=100
    # R2 checkpoint sink, the #90 contract: CKPT_R2_ENABLED gates the trainer
    # sink; creds + R2_BUCKET (hard-guarded to shamane-pluralis in r2_sink.py)
    # come from the secrets file; EXPERIMENT/REGIME name the R2 prefix.
    export CKPT_R2_ENABLED=true
    EXPERIMENT_NAME="c-${CODEC_ARM}-${CODEC_SLUG}-val600"
    export R2_EXPERIMENT="${R2_EXPERIMENT:-$RUN_GROUP}"
    export R2_REGIME="${R2_REGIME:-$EXPERIMENT_NAME}"
    ;;
  *)
    fatal "unknown ARM '$ARM' (a1|a2|a3|a4|a5|b1|c)"
    ;;
esac

export TOTAL_STEPS TEST_FREQ VAL_BEFORE_TRAIN SAVE_FREQ EXPERIMENT_NAME

cat <<EOF
=== resolved #93 cell config: $ARM ===
  run name:            $EXPERIMENT_NAME
  wandb group/project: $RUN_GROUP
  codec:               ${COMM_EFF_COMPRESSION_TYPE}
  sr_quant:            bits=${COMM_EFF_QUANT_BITS:-<default>} block=${COMM_EFF_QUANT_BLOCK_SIZE:-<default>} rounding=${COMM_EFF_QUANT_ROUNDING:-<default>} subset_k=${COMM_EFF_QUANT_SUBSET_K:-0}
  prf_mask:            enabled=${COMM_EFF_MASK_ENABLED:-false} p=${COMM_EFF_MASK_P:-<default>} rescale_mode=${COMM_EFF_MASK_RESCALE_MODE:-<default>} exact_k=${COMM_EFF_MASK_EXACT_K:-false}
  frlr:                ${COMM_EFF_MASK_FRLR:-false} rank=${COMM_EFF_MASK_FRLR_RANK:-<default>} k=${COMM_EFF_MASK_FRLR_K:-<default>}
  cvc:                 ce_lambda=${COMM_EFF_CVC_LAMBDA:-0.0} warmup=${COMM_EFF_CVC_WARMUP_STEPS:-20}
  rollout_is:          ${ROLLOUT_IS:-null} threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
  probe:               every=${COMM_EFF_PROBE_EVERY:-0} ctrl=${COMM_EFF_PROBE_CTRL_ENABLED:-false} table=[${COMM_EFF_PROBE_KL_TARGET_TABLE:-<unset>}]
  schedule:            steps=$TOTAL_STEPS test_freq=$TEST_FREQ val_before_train=$VAL_BEFORE_TRAIN save_freq=$SAVE_FREQ
  r2 ckpt sink:        ${CKPT_R2_ENABLED:-false} experiment=${R2_EXPERIMENT:-<n/a>} regime=${R2_REGIME:-<n/a>}
  geometry:            batch 128 / mini 128, prompt 1024 / response 2048, pp 8, LR 1e-6, gpu_mem $GPU_MEM
=== end resolved config ===
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: resolved config only, no bring-up."
  exit 0
fi

# R2 preflight for cell c: fail before GPU spend, not at the step-100 save.
if [[ "$ARM" == "c" ]]; then
  command -v aws >/dev/null 2>&1 || fatal "cell c needs the aws CLI for the R2 checkpoint sink (pip install awscli)"
  SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
  R2_PRECHECK="$(bash -c "source '$SECRETS_FILE' 2>/dev/null; \
    [[ \"\${R2_BUCKET:-}\" == shamane-pluralis && -n \"\${R2_ACCESS_KEY_ID:-}\" && -n \"\${R2_SECRET_ACCESS_KEY:-}\" \
       && ( -n \"\${R2_ENDPOINT:-}\" || -n \"\${R2_ACCOUNT_ID:-}\" ) ]] && echo ok")" || true
  [[ "$R2_PRECHECK" == "ok" ]] \
    || fatal "cell c needs R2 creds in $SECRETS_FILE (R2_BUCKET=shamane-pluralis + R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY + R2_ENDPOINT|R2_ACCOUNT_ID)"
fi

# ---------------------------------------------------------------------------
# 2. Bring-up (the #90 pattern): checkout, editable install, money gate, data.
# ---------------------------------------------------------------------------
RUN_DIR="$WORK/runs/$EXPERIMENT_NAME"
mkdir -p "$RUN_DIR"
cd "$WORK"

if [[ -d verl/.git ]] \
   && (cd verl && git remote set-url origin "$REPO" \
       && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
       && git reset --hard FETCH_HEAD); then
  echo "=== reused checkout, reset to origin/$BRANCH ==="
else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl
fi
cd "$WORK/verl"
echo "=== verl HEAD: $(git rev-parse HEAD) ==="

python3 -c "import verl" 2>/dev/null || {
  if command -v uv >/dev/null 2>&1; then uv pip install --no-deps -e . ; else python3 -m pip install --no-deps -e . ; fi
}
python3 -c "import verl" || fatal "verl import failed after install"

# Money gate: prove every #93 lever this matrix uses is in this checkout.
python3 - <<'PY' || fatal "#93 levers absent from checkout (wrong branch?)"
import inspect
from verl.workers.comm_eff.activation_mask import prf_token_mask
from verl.workers.comm_eff.activation_quant import sr_quantize
from verl.workers.config.comm_eff import CommEffDCConfig, CommEffProbeConfig  # noqa: F401

assert "exact_k" in inspect.signature(prf_token_mask).parameters, "prf_token_mask missing exact_k"
assert "subset_k" in inspect.signature(sr_quantize).parameters, "sr_quantize missing subset_k (I5)"
print("OK: #93 levers present (exact_k, sr_quant subset_k, probe, dc)")
PY

export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || fatal "MATH parquet unavailable in $DATA_DIR"

# ---------------------------------------------------------------------------
# 3. Patched launcher copy (the #90 geometry): only three scalars change.
# ---------------------------------------------------------------------------
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_93_cell.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"
chmod +x "$PATCHED"
# fail loud if the base launcher shape drifted and a substitution missed
grep -q '^export TRAIN_BATCH_SIZE=128$'    "$PATCHED" || fatal "batch patch missed"
grep -q '^export PPO_MINI_BATCH_SIZE=128$' "$PATCHED" || fatal "mini-batch patch missed"
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || fatal "response patch missed"

# ---------------------------------------------------------------------------
# 4. Run controls + WandB naming, then launch.
# ---------------------------------------------------------------------------
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export TOTAL_TRAINING_STEPS="$TOTAL_STEPS"
export TOTAL_EPOCHS=20                    # >=11 needed so 600 steps is the stop at batch 128
export TEST_FREQ SAVE_FREQ VAL_BEFORE_TRAIN EXPERIMENT_NAME
# The engine derives PROJECT_NAME from WANDB_RUN_GROUP: project
# 93-long-horizon-stability, run $EXPERIMENT_NAME.
export WANDB_RUN_GROUP="$RUN_GROUP"
export PROJECT_NAME="$RUN_GROUP"
export LOG="$RUN_DIR/train.log"
# WandB: use it if a key is present, else fall back to offline (no crash).
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== launching #93 cell $ARM as $RUN_GROUP/$EXPERIMENT_NAME: $TOTAL_STEPS steps, test_freq $TEST_FREQ, save_freq $SAVE_FREQ, gpu_mem $GPU_MEM ==="
exec bash "$PATCHED"
