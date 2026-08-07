#!/usr/bin/env bash
# run_qwen3_4b_4k_500_fsdp.sh
#
# ONE-COMMAND bring-up and launch of ONE ARM of the Qwen3-4B-Base 4096-context
# 500-step MATH GRPO pair on 4x H200.
#
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh commeff   # arm A (run first)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh dense     # arm B (control)
#
# The two arms are byte-identical except for the master compression switch, so
# the comparison isolates the codec and nothing else. Everything the box needs is
# done here: checkout, install, awscli bootstrap, R2 bucket repair, MATH parquet,
# CPU money gates, launcher patch, launch. Nothing is authored on the box.
#
# Surface, and the deltas from the project default:
#   model                Qwen2.5-Math-1.5B -> Qwen3-4B-Base   (36 layers, hidden 2560)
#   context              1024/3072 UNCHANGED                  (4096 total, the reference protocol)
#   train / mini batch   512/256 -> 128/128                   (one optimizer tick per generation)
#   steps                100 -> 500, val at 0/100/200/.../500
#   checkpoints          off -> every 100 steps, mirrored to R2, KEPT locally for eval
#   codec (arm A)        UNCHANGED: prf_mask, p=0.95, exact-k, constant rescale
#   codec (arm B)        COMM_EFF_ENABLED=false               (the engine's dense path)
#
# WHY 128/128 rather than the 512/256 in CLAUDE.md: 128/128 is the surface the
# 600-step horizon evidence sits on (issue #90's PRF-vs-dense pair and every #93
# arm). It gives ONE optimizer tick per global step, so anchor cadence 20 ticks
# reads directly as "every 20 global steps", and it is what makes 2 x 500 steps
# affordable on a single 4-GPU box. Set TRAIN_BATCH_SIZE/PPO_MINI_BATCH_SIZE to
# 512/256 to train on the CLAUDE.md surface instead; the launcher patches either.
#
# At hidden 2560 exact-k keeps exactly 128 of 2560 coordinates per token per
# boundary (2048 bits/token/boundary against 40960 dense, i.e. 5.0 percent of the
# wire). With 36 layers over 8 pipeline stages the boundaries are decoder layers
# [4, 9, 14, 19, 23, 27, 31]. Both facts are asserted below before a GPU is touched.
#
# Run inside tmux. The engine redirects training to $LOG, which is the heartbeat
# log the harness registers as run.json's remote_log.
set -uo pipefail

# ---------------------------- arm ------------------------------------------
ARM="${1:-commeff}"
case "$ARM" in
  commeff|dense) ;;
  *) echo "FATAL: unknown arm '$ARM' (commeff|dense)" >&2; exit 1 ;;
esac
shift || true

# ---------------------------- knobs ----------------------------------------
RUN_ID="${RUN_ID:-qwen3-4b-4k-500}"
ARM_NAME="${ARM_NAME:-qwen3-4b-4k-${ARM}-500}"
BRANCH="${BRANCH:-exp/qwen3-4b-4k-500}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
WORK="${WORK:-/workspace}"

# Money gate. There are TWO independent `${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}`
# defaults on the path to Hydra (run_qwen25_math_1p5b_rank1_relex_fsdp.sh and
# vast_comm_eff_engine_grpo.sh), so an unset MODEL_PATH does not fail: it
# silently trains the 1.5B model for five hundred steps. Set it explicitly here.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
EXPECT_HIDDEN="${EXPECT_HIDDEN:-2560}"
EXPECT_LAYERS="${EXPECT_LAYERS:-36}"

# Context and batch. These four are BARE `export NAME=value` lines in the base
# launcher (no ${VAR:-default}), so exporting them here would be overwritten;
# they have to be patched into a generated copy. See section 7.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Box gates.
EXPECT_GPUS="${EXPECT_GPUS:-4}"
MIN_RAM_GIB="${MIN_RAM_GIB:-320}"      # anchor keeps a CPU clone + CPU snapshots + the fp32 EMA M, per rank
MIN_DISK_GIB="${MIN_DISK_GIB:-900}"    # 10 kept 4B checkpoints across both arms + merged eval models
# SKIP_BOX_GATES=1 downgrades the RAM and disk gates to loud warnings.
SKIP_BOX_GATES="${SKIP_BOX_GATES:-0}"

# Checkpoints are KEPT locally (the same box runs the OOD eval afterwards, and
# re-pulling ~250 GB per arm over a 2-8 MB/s uplink would dominate the schedule).
# Set CKPT_R2_DELETE_LOCAL=true on a small-disk box and let the eval re-pull.
CKPT_R2_DELETE_LOCAL="${CKPT_R2_DELETE_LOCAL:-false}"
# ---------------------------------------------------------------------------

RUN_DIR="$WORK/runs/$ARM_NAME"
mkdir -p "$RUN_DIR" || { echo "FATAL: cannot create $RUN_DIR" >&2; exit 1; }
cd "$WORK" || { echo "FATAL: cannot cd $WORK" >&2; exit 1; }

echo "=== $ARM_NAME: Qwen3-4B-Base / arm=$ARM / 4096 ctx / 500 steps ==="

# ---------------------------------------------------------------------------
# 1. Box preflight. Everything that makes this run impossible, checked before
#    the checkout so a bad box fails in seconds instead of after a model pull.
# ---------------------------------------------------------------------------
box_gate_fail() {
  # RAM/disk are advisory under SKIP_BOX_GATES=1; everything else is fatal.
  if [[ "$SKIP_BOX_GATES" == "1" && "${2:-hard}" == "soft" ]]; then
    echo "WARN: $1" >&2
    echo "WARN: continuing anyway because SKIP_BOX_GATES=1." >&2
    return 0
  fi
  echo "FATAL: $1" >&2
  [[ "${2:-hard}" == "soft" ]] && echo "       (set SKIP_BOX_GATES=1 to downgrade this to a warning)" >&2
  exit 1
}

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
echo "--- preflight GPUs:  detected=${GPU_COUNT:-0} required=$EXPECT_GPUS"
[[ "$GPU_COUNT" == "$EXPECT_GPUS" ]] \
  || box_gate_fail "expected $EXPECT_GPUS GPUs, detected ${GPU_COUNT:-0} (set EXPECT_GPUS to override)"

RAM_GIB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
echo "--- preflight RAM:   ${RAM_GIB:-unknown} GiB required=$MIN_RAM_GIB GiB"
if [[ "$RAM_GIB" =~ ^[0-9]+$ ]]; then
  (( RAM_GIB >= MIN_RAM_GIB )) \
    || box_gate_fail "host RAM ${RAM_GIB} GiB is below the ${MIN_RAM_GIB} GiB this run needs" soft
else
  box_gate_fail "could not read host RAM from 'free -g'" soft
fi

DISK_GIB="$(df -Pk "$WORK" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')"
echo "--- preflight disk:  ${DISK_GIB:-unknown} GiB free on $WORK required=$MIN_DISK_GIB GiB"
if [[ "$DISK_GIB" =~ ^[0-9]+$ ]]; then
  (( DISK_GIB >= MIN_DISK_GIB )) \
    || box_gate_fail "only ${DISK_GIB} GiB free on $WORK, need ${MIN_DISK_GIB} GiB to KEEP both arms' checkpoints (or set CKPT_R2_DELETE_LOCAL=true and re-pull for eval)" soft
else
  box_gate_fail "could not read free disk for $WORK" soft
fi

# ---------------------------------------------------------------------------
# 2. awscli bootstrap. The R2 sink shells out to `aws`, and a MISSING binary does
#    not skip the upload, it RAISES out of _save_checkpoint and kills the run at
#    the first save. That cost 49 H200 steps on issue #95 even though it was
#    already a known failure. Install it HERE, on a laptop-visible line.
#    v2's self-contained zip bundles its own Python, so it cannot perturb the
#    pinned torch/vllm stack the way `pip install awscli` can.
# ---------------------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
  echo "=== installing awscli v2 (required by the R2 checkpoint sink) ==="
  ( cd /tmp \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip -qo awscliv2.zip \
    && ./aws/install --update ) >/tmp/awscli_install.log 2>&1 \
    || { echo "FATAL: awscli v2 install failed; see /tmp/awscli_install.log" >&2; exit 1; }
fi
command -v aws >/dev/null 2>&1 || box_gate_fail "'aws' still not on PATH after install: the R2 checkpoint mirror cannot run"
echo "--- preflight aws:   $(command -v aws) ($(aws --version 2>&1 | head -1))"

# A 256 MB part size is not cosmetic. At the 8 MB default an 11.5 GB optimizer
# shard becomes ~1437 parts and R2 rejects the CompleteMultipartUpload with
# InvalidPart, silently losing exactly the big files.
aws configure set default.s3.multipart_chunksize 256MB || true
aws configure set default.s3.multipart_threshold 256MB || true

# ---------------------------------------------------------------------------
# 3. R2 bucket repair, IN THE SECRETS FILE. The engine re-sources
#    ~/.config/verl-research/secrets.env at runtime, which CLOBBERS any
#    `export R2_BUCKET=` a wrapper sets. The box's secrets file often ships
#    R2_BUCKET set to the PREFIX (autonomous-harness-rlvr-compression) rather
#    than the bucket, and r2_sink.py hard-refuses anything but shamane-pluralis.
#    So the correction has to land in the file itself.
# ---------------------------------------------------------------------------
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
[[ -r "$SECRETS_FILE" ]] || box_gate_fail "$SECRETS_FILE not found; the engine needs HF_TOKEN + WANDB_API_KEY and the sink needs the R2 keys"
if grep -qE '^[[:space:]]*(export[[:space:]]+)?R2_BUCKET=' "$SECRETS_FILE"; then
  if ! grep -qE '^[[:space:]]*(export[[:space:]]+)?R2_BUCKET=["'"'"']?shamane-pluralis["'"'"']?[[:space:]]*$' "$SECRETS_FILE"; then
    cp -a "$SECRETS_FILE" "$SECRETS_FILE.bak.$(date +%s)"
    sed -i.tmp -E 's|^([[:space:]]*(export[[:space:]]+)?)R2_BUCKET=.*$|\1R2_BUCKET=shamane-pluralis|' "$SECRETS_FILE"
    rm -f "$SECRETS_FILE.tmp"
    echo "--- repaired R2_BUCKET in $SECRETS_FILE -> shamane-pluralis (backup kept)"
  fi
else
  printf 'export R2_BUCKET=shamane-pluralis\n' >> "$SECRETS_FILE"
  echo "--- appended R2_BUCKET=shamane-pluralis to $SECRETS_FILE"
fi
grep -qE '^[[:space:]]*(export[[:space:]]+)?R2_BUCKET=["'"'"']?shamane-pluralis' "$SECRETS_FILE" \
  || box_gate_fail "R2_BUCKET repair did not stick in $SECRETS_FILE"

# Round-trip the credentials NOW, not an hour into training. Never print a value.
( set +u; set -a; . "$SECRETS_FILE"; set +a
  : "${R2_ENDPOINT:?R2_ENDPOINT missing from the secrets file}"
  aws s3 ls "s3://shamane-pluralis/" --endpoint-url "$R2_ENDPOINT" >/dev/null 2>&1 ) \
  || box_gate_fail "R2 credential check failed (aws s3 ls s3://shamane-pluralis/ did not succeed)"
echo "--- preflight R2:    credentials OK, bucket shamane-pluralis reachable"

# ---------------------------------------------------------------------------
# 4. Obtain the verl checkout (fast path reuse, else shallow clone).
#    ORDERING IS LOAD-BEARING: every reset/checkout happens HERE, and the money
#    gate in section 6 runs on the FINAL tree. run_layer_rotation_300.sh gated
#    first and then hard-reset to a different branch, and burned 16 GPU-hours
#    running code the gate never saw. Nothing below this section touches git.
# ---------------------------------------------------------------------------
if [[ -d verl/.git ]] \
   && (cd verl && git remote set-url origin "$REPO" \
       && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
       && git reset --hard FETCH_HEAD); then
  echo "=== reused checkout, reset to origin/$BRANCH ==="
else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl \
    || { echo "FATAL: could not clone $REPO @ $BRANCH" >&2; exit 1; }
fi
cd "$WORK/verl" || { echo "FATAL: cannot cd $WORK/verl" >&2; exit 1; }
echo "=== verl HEAD: $(git rev-parse HEAD) ($BRANCH) ==="

# ---------------------------------------------------------------------------
# 5. Editable install (no deps) if verl is not importable yet.
# ---------------------------------------------------------------------------
python3 -c "import verl" 2>/dev/null || {
  if command -v uv >/dev/null 2>&1; then uv pip install --no-deps -e . ; else python3 -m pip install --no-deps -e . ; fi
}
python3 -c "import verl" || { echo "FATAL: verl import failed after install" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 6. Money gate, on the FINAL checkout. Four claims, all cheap and all CPU:
#      a) the exact-k lever exists at all;
#      b) at Qwen3-4B-Base's hidden 2560 it keeps exactly 128 coordinates per
#         token (the 5 percent budget), so the wire budget is what we think;
#      c) 36 layers over 8 stages gives the boundary set we costed;
#      d) the anchor clone inherits gradient checkpointing. Without that fix the
#         anchor's dense replay forward stores every activation: measured
#         +62.9 GiB at the first fire on a 1.5B model (run 90, step 20). A stale
#         checkout silently OOMs, so refuse to start on one.
#    The model's own config is checked against (b)/(c) when the hub is reachable,
#    so a renamed or re-shaped checkpoint cannot slip through.
# ---------------------------------------------------------------------------
MODEL_PATH="$MODEL_PATH" EXPECT_HIDDEN="$EXPECT_HIDDEN" EXPECT_LAYERS="$EXPECT_LAYERS" \
python3 - <<'PY' || { echo "FATAL: money gate FAILED — this checkout must not be launched" >&2; exit 1; }
import inspect
import os

import torch

from verl.workers.comm_eff.activation_mask import prf_token_mask
from verl.workers.comm_eff.anchor import build_anchor_module
from verl.workers.comm_eff.boundary import decoder_boundary_indices

H = int(os.environ["EXPECT_HIDDEN"])
L = int(os.environ["EXPECT_LAYERS"])
MODEL = os.environ["MODEL_PATH"]

assert "exact_k" in inspect.signature(prf_token_mask).parameters, "prf_token_mask is missing exact_k"

mask = prf_token_mask(
    sample_ids=torch.zeros(1, dtype=torch.int64),
    position_ids=torch.zeros(1, dtype=torch.int64),
    layer_idx=0,
    global_step=1,
    base_seed=0,
    hidden_size=H,
    p=0.95,
    device=torch.device("cpu"),
    dtype=torch.float32,
    exact_k=True,
)
kept = int((mask != 0).sum().item())
assert kept == 128, f"exact-k at hidden {H} kept {kept} coordinates, expected 128"

anchor_src = inspect.getsource(build_anchor_module)
assert "gradient_checkpointing_enable" in anchor_src, (
    "build_anchor_module does NOT enable gradient checkpointing on the anchor clone. "
    "This checkout predates that fix; the first anchor fire will OOM. Refusing to launch."
)

boundaries = decoder_boundary_indices(L, 8)
assert boundaries == [4, 9, 14, 19, 23, 27, 31], f"unexpected boundary set {boundaries}"

# The shape claims above are only meaningful if the model really has them.
try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=False)
except Exception as exc:  # offline / hub hiccup: the gate stays advisory here
    print(f"WARN: could not read {MODEL} config ({type(exc).__name__}); shape claims are UNVERIFIED")
else:
    assert cfg.hidden_size == H, f"{MODEL} hidden_size is {cfg.hidden_size}, gate assumed {H}"
    assert cfg.num_hidden_layers == L, f"{MODEL} has {cfg.num_hidden_layers} layers, gate assumed {L}"
    print(f"OK: {MODEL} config confirms hidden {cfg.hidden_size}, {cfg.num_hidden_layers} layers")

print(f"OK: exact-k keeps {kept}/{H} coords per token per boundary "
      f"({kept * 16} bits/token/boundary vs {H * 16} dense = {100.0 * kept / H:.1f} percent)")
print("OK: anchor clone inherits gradient checkpointing")
print(f"OK: {L} layers over 8 stages -> boundaries {boundaries}")
PY

# ---------------------------------------------------------------------------
# 7. MATH parquet ($HOME/data/math), prepared with the canonical research prep.
# ---------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: MATH parquet unavailable in $DATA_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 8. Patched launcher copy. Only the four bare scalars change. Each one is
#    checked against its EXPECTED CURRENT value in the base first (so a drifted
#    base fails loud instead of being silently un-patched) and asserted again in
#    the generated copy afterwards.
# ---------------------------------------------------------------------------
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_qwen3_4b_4k_500.${ARM}.gen.sh"
[[ -f "$BASE" ]] || { echo "FATAL: base launcher $BASE not found" >&2; exit 1; }
cp "$BASE" "$PATCHED" || { echo "FATAL: could not copy $BASE" >&2; exit 1; }

# name | value the base is expected to hold today | value this run needs
PATCH_SPEC=(
  "MAX_PROMPT_LENGTH|1024|$MAX_PROMPT_LENGTH"
  "MAX_RESPONSE_LENGTH|3072|$MAX_RESPONSE_LENGTH"
  "TRAIN_BATCH_SIZE|512|$TRAIN_BATCH_SIZE"
  "PPO_MINI_BATCH_SIZE|256|$PPO_MINI_BATCH_SIZE"
)
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom pto <<< "$spec"
  grep -q "^export ${pname}=${pfrom}\$" "$BASE" || {
    echo "FATAL: base launcher shape drifted — expected 'export ${pname}=${pfrom}' in $BASE" >&2
    echo "       Re-derive this launcher's patch table before running." >&2
    exit 1
  }
  sed -i.bak -e "s/^export ${pname}=${pfrom}\$/export ${pname}=${pto}/" "$PATCHED" \
    || { echo "FATAL: sed failed for ${pname}" >&2; exit 1; }
  rm -f "$PATCHED.bak"
  grep -q "^export ${pname}=${pto}\$" "$PATCHED" \
    || { echo "FATAL: ${pname} patch missed (wanted ${pto})" >&2; exit 1; }
  echo "--- patched ${pname}: ${pfrom} -> ${pto}"
done
chmod +x "$PATCHED"

TOTAL_CTX=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
(( TOTAL_CTX == MAX_MODEL_LEN )) \
  || { echo "FATAL: prompt+response ($TOTAL_CTX) != MAX_MODEL_LEN ($MAX_MODEL_LEN)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 9. Model + codec + run controls.
#
#    ARM A (commeff): the project default codec block, IDENTICAL to
#    run_prf_exactk_600.sh and to the 8B launcher: prf_mask, p=0.95, exact-k,
#    constant rescale, masking the train forward + the old-logprob recompute +
#    the reference forward, 8 pipeline stages, no anchor-owned Q.
#
#    ARM B (dense): the master switch OFF. That is the ONLY science delta. The
#    engine's dense path leaves the anchor, the RELEX projector and the signed
#    EMA inert, so this is a clean uncompressed control on the same surface.
# ---------------------------------------------------------------------------
export MODEL_PATH

if [[ "$ARM" == "commeff" ]]; then
  export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
  export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-prf_mask}"
  export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
  export COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"
  export COMM_EFF_MASK_RESCALE_MODE="${COMM_EFF_MASK_RESCALE_MODE:-constant}"
  export COMM_EFF_MASK_EXACT_K="${COMM_EFF_MASK_EXACT_K:-true}"
  export COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"
  export COMM_EFF_MASK_REFERENCE="${COMM_EFF_MASK_REFERENCE:-true}"
  export COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"
  export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-false}"        # prf_mask has no basis Q to own
  export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"
else
  export COMM_EFF_ENABLED=false
fi

# Run schedule. 500 steps at batch 128 needs ~9 MATH epochs, so the step cap is
# the stopping condition; 20 leaves margin after prompt filtering.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export TEST_FREQ="${TEST_FREQ:-100}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export SAVE_FREQ="${SAVE_FREQ:-100}"

# Checkpoints -> R2. R2_EXPERIMENT/R2_REGIME are the two path segments; neither
# is in the secret-seeding allowlist and no other script sets them, so an unset
# pair silently lands the whole run under EXP-unknown/regime/.
#   s3://shamane-pluralis/autonomous-harness-rlvr-compression/$RUN_ID/$ARM_NAME/checkpoints/global_step_<N>/actor/
export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-true}"
export CKPT_R2_ASYNC="${CKPT_R2_ASYNC:-true}"
export CKPT_R2_DELETE_LOCAL
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-120}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-4}"
export R2_EXPERIMENT="${R2_EXPERIMENT:-$RUN_ID}"
export R2_REGIME="${R2_REGIME:-$ARM_NAME}"

# Token budgets. The actor budget stays at the anchor-aware 18432. The log-prob
# paths default to 36864, which was sized for a 1.5B model on this same 4k
# protocol; 4B is 2.1x the per-token activation footprint, so cut them to 24576.
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}"
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}"
export ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
# 4B leaves far more room than 8B did, but the anchor's unsharded clone is the
# peak and it does not shard, so hold vLLM below the 0.72 the 8B run used.
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.60}"
# ROLLOUT_TP is NOT set here on purpose: it is a bare `export ROLLOUT_TP=1` in
# the base launcher, so anything exported from here is discarded. TP=1 is what
# we want anyway (4B fits one GPU, so the four ranks are four independent vLLM
# replicas and generation scales with the GPU count). Change it in the patch
# table above, not here.

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$ARM_NAME}"
export PROJECT_NAME="${PROJECT_NAME:-$RUN_ID}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"
export LOG="${LOG:-$RUN_DIR/train.log}"
# WandB: use it if a key is present, else fall back to offline (no crash).
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

# The engine mkdir -p's only dirname($LOG), then at the very end touches
# $VERL_ROOT/runs/$EXPERIMENT_NAME/done.flag, a path nothing created. Because
# $LOG lives outside the repo here, that touch fails under `set -e` and a
# CLEAN run exits non-zero with no done.flag. Create the directory up front so
# completion is recorded and the exit status means what it says.
mkdir -p "$WORK/verl/runs/$EXPERIMENT_NAME"

# SMOKE=1 is the cheap look before the 500-step commitment: 25 steps, no
# validation, no checkpoints, nothing written to R2. It runs PAST the first
# anchor fire at step 20 on purpose. Read three things off it.
#   timing_s/step         prices the run (ignore step 1, always an outlier)
#   max_memory_allocated  the anchor's replay clone does not shard, so its peak
#                         is set by the token budget rather than the GPU count,
#                         and it only appears at the first fire on step 20
#   response_length/mean  and clip_ratio: Qwen3 base models are wordier than the
#                         Qwen2.5-Math model this recipe was tuned on, and a
#                         response distribution already pressed against the cap
#                         is the truncation feedback loop that ended the last
#                         long-context attempt. The reference 1.5B run sat at a
#                         574-token mean with 2 percent clipped and FELL over
#                         time. Anything climbing toward 3072 means stop.
if [[ "${SMOKE:-0}" == "1" ]]; then
  export TOTAL_TRAINING_STEPS="${SMOKE_STEPS:-25}"
  export TEST_FREQ=-1
  export VAL_BEFORE_TRAIN=False
  export SAVE_FREQ=-1
  export CKPT_R2_ENABLED=false
  export EXPERIMENT_NAME="${ARM_NAME}-smoke"
  export LOG="$RUN_DIR/smoke.log"
  mkdir -p "$WORK/verl/runs/$EXPERIMENT_NAME"
  echo "=== SMOKE MODE: $TOTAL_TRAINING_STEPS steps, no val, no checkpoints, no R2 ==="
fi

# ---------------------------------------------------------------------------
# 10. Hydra overrides this launcher owns.
#
#     max_model_len is REQUIRED, not cosmetic. The engine does not emit it, so it
#     falls back to rollout.yaml's null and vLLM then uses the checkpoint's
#     max_position_embeddings, which is 32768 for Qwen3-4B-Base: eight times the
#     context we are paying for, and a KV cache sized for it. max_num_batched_tokens
#     must stay >= max_model_len because enable_chunked_prefill=False is hardcoded
#     in the engine; 8192 is the schema default and fits two full sequences.
#     Both keys already exist in the schema, so neither takes a `+` prefix.
#
#     "$@" goes LAST so a caller's override of the same key wins (Hydra applies
#     CLI overrides in order).
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES=(
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
)

if [[ "$ARM" == "commeff" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE"
else
  CODEC_LINE="DENSE CONTROL (comm_eff master switch off)"
fi

cat <<EOF
=== launching $ARM_NAME ===
    arm              $ARM
    model            $MODEL_PATH
    context          $MAX_PROMPT_LENGTH prompt + $MAX_RESPONSE_LENGTH response = $TOTAL_CTX
    batch            train $TRAIN_BATCH_SIZE / mini $PPO_MINI_BATCH_SIZE (one optimizer tick per step)
    codec            $CODEC_LINE
    schedule         $TOTAL_TRAINING_STEPS steps, test_freq $TEST_FREQ, save_freq $SAVE_FREQ
    r2               enabled=$CKPT_R2_ENABLED delete_local=$CKPT_R2_DELETE_LOCAL
                     s3://shamane-pluralis/autonomous-harness-rlvr-compression/$R2_EXPERIMENT/$R2_REGIME/checkpoints/
    rollout          gpu_mem $ROLLOUT_GPU_MEM_UTIL, tp 1 (pinned in the base launcher), max_model_len $MAX_MODEL_LEN
    local ckpts      $WORK/verl/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/global_step_<N>/actor
    log              $LOG
    hydra            ${HYDRA_OVERRIDES[*]} $*
EOF

exec bash "$PATCHED" "${HYDRA_OVERRIDES[@]}" "$@"
