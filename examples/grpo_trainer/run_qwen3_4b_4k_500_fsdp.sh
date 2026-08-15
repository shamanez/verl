#!/usr/bin/env bash
# run_qwen3_4b_4k_500_fsdp.sh
#
# ONE-COMMAND bring-up and launch of ONE ARM of the Qwen3-4B-Base 4096-context
# 500-step MATH GRPO pair on 4x H200.
#
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh commeff   # arm A (run first)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh dense     # arm B (control)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh optreset  # arm C (arm A + anchor-sourced optimizer reset)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh freshm    # arm D (arm A, sign correction ONLY on fire ticks)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh nosign    # arm E (arm A with the signed-EMA merger OFF)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh powersgdq # arm F (arm D, codec swapped to PowerSGD w/ anchor-owned Q)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh delayedef       # arm G (arm A, merger swapped to delayed_ef, stale delta reused every tick)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh delayedef-fresh # arm H (arm G, delta applied on fire ticks ONLY)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh blend           # arm I (arm G, merger swapped to the convex value blend, eta=0.3)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh delayedef-cad10 # arm J (arm H, spectral.cadence 20 -> 10: ONE stale delta per interval)
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh delayedef-anneal # arm K (arm G + decay=0.75: held delta ANNEALED by decay^age each tick)
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
#   codec (arm D)        arm A, spectral.cadence 1 -> 20       (fresh M only, no stale reuse)
#   codec (arm E)        arm A, signed_ema_alpha 0.25 -> 1.0   (merger becomes identity)
#   codec (arm F)        arm D, prf_mask -> powersgd r=128      (anchor-owned Q, same 5.0% wire)
#   codec (arm G)        arm A, merger signed_ema -> delayed_ef (lambda=1, beta_anc=0; stale delta every tick)
#   codec (arm H)        arm G, spectral.cadence 1 -> 20        (delta applied on fire ticks only)
#   codec (arm I)        arm A, merger signed_ema -> blend      (eta=0.3, beta_anc=0; convex, no held residual)
#   codec (arm J)        arm H, spectral.cadence 20 -> 10       (interior stale-dose point: held delta re-applied ONCE per interval)
#   codec (arm K)        arm G, delayed_ef_decay 1.0 -> 0.75    (held delta annealed: weight lambda*decay^age, interval impulse ~4 vs arm G's 20)
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
  commeff|dense|freshm|nosign|powersgdq|delayedef|delayedef-fresh|delayedef-cad10|delayedef-anneal|blend) ;;
  # The optreset arm names itself off the commeff arm it extends (cadence 50),
  # so its WandB run, R2 regime, log dir and done.flag dir all carry the delta.
  optreset) ARM_NAME="${ARM_NAME:-qwen3-4b-4k-commeff-optreset50-500}" ;;
  *) echo "FATAL: unknown arm '$ARM' (commeff|dense|optreset|freshm|nosign|powersgdq|delayedef|delayedef-fresh|delayedef-cad10|delayedef-anneal|blend)" >&2; exit 1 ;;
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
  command -v unzip >/dev/null 2>&1 || apt-get install -y -qq unzip >/dev/null 2>&1 || true
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
# The R2_* names must be mapped onto the AWS_* names aws actually reads; the
# r2_sink does the same mapping internally at upload time.
( set +u; set -a; . "$SECRETS_FILE"; set +a
  : "${R2_ENDPOINT:?R2_ENDPOINT missing from the secrets file}"
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
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
#
#    ARM D (freshm): arm A with ONE number changed, spectral.cadence 1 -> 20,
#    locked to anchor.cadence. THE STALE M IS NEVER REUSED: the correction is
#    applied on the fire ticks only, where M was refreshed earlier in the SAME
#    train_batch, and is skipped on the nineteen ticks in between.
#
#    Two cadences are involved and they are not the same knob:
#      anchor.cadence=20    REFRESHES M (dense replay -> EMA), on ticks 20,40,...
#      spectral.cadence=1   APPLIES sign(M) to the gradient, on EVERY tick
#    Run A's own counters show the consequence: anchor_replay_fires=10 against
#    spectral_corrections=72038, which factors as 398 floating params x 181
#    ticks, and 181 = its 200 steps minus the 19 warmup ticks before M was ready.
#    So 181 of 200 steps took the correction and only 10 of those read a freshly
#    refreshed M. Stale applications outnumbered fresh ones eighteen to one.
#    Setting spectral.cadence=anchor.cadence collapses 181 down to 10, each one
#    fresh, which is the hypothesis this arm tests: that it is the REUSE of a
#    frozen M between fires that does the damage, not the correction itself.
#
#    Ordering is what makes this work, and it is a fact of the engine rather than
#    an assumption: BaseEngine.train_batch calls the anchor refresh at the TOP,
#    then the compressed fwd/bwd, then the grad correction, then optimizer_step
#    (verl/workers/engine/base.py). Both hooks advance their counter on every
#    train_batch, so with equal cadences they fire on exactly the same ticks and
#    the correction always reads an M refreshed moments earlier in that tick.
#    The config validator independently requires anchor.cadence % spectral.cadence
#    == 0, and 20 % 20 == 0.
#
#    Everything else is arm A untouched: alpha stays 0.25, beta_anc 0.25, the
#    codec, the anchor, the RELEX projection, the batch shape and the schedule.
#    No optimizer state is swapped or reset (that is arm C).
#
#    ARM E (nosign): arm A with spectral.signed_ema_alpha 0.25 -> 1.0. The merger
#    computes
#        G_corr = alpha*G + (1-alpha)*|G|*sign(M)
#    so alpha=1.0 returns G bit-for-bit and M never reaches the optimizer at all.
#    This is the ZERO point of the same dose axis arm D sits in the middle of:
#    481 applications (arm A) -> 25 fresh ones (arm D) -> none (arm E) over 500
#    steps. It is also the endpoint of the alpha axis #84 swept (0.25 best, 0.5
#    worse).
#
#    Why alpha=1.0 and NOT spectral.enabled=false for arm E: the config validator
#    requires spectral.enabled=true whenever anchor.lookahead_mode=rank1_relex
#    (see CommEffConfig.__post_init__), and with spectral off the engine's anchor
#    hook returns before it snapshots anything, so the whole anchor circuit would
#    disappear too. That is three deltas, not one. alpha=1.0 keeps the anchor
#    firing, the RELEX projection running and M accumulating exactly as in arm A,
#    and changes only whether M is allowed to rewrite gradient signs.
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
elif [[ "$ARM" == "optreset" ]]; then
  # ARM C: the commeff codec block VERBATIM (kept a separate branch so arm A
  # stays byte-identical to its reference launchers), plus the anchor-sourced
  # optimizer-state reset: every 50 optimizer ticks the fast AdamW moments are
  # overwritten with the anchor's clean-replay moments (norm-matched).
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
  export COMM_EFF_OPT_RESET_ENABLED="${COMM_EFF_OPT_RESET_ENABLED:-true}"
  export COMM_EFF_OPT_RESET_CADENCE="${COMM_EFF_OPT_RESET_CADENCE:-50}"
  export COMM_EFF_OPT_RESET_MODE="${COMM_EFF_OPT_RESET_MODE:-anchor_moments}"
  export COMM_EFF_OPT_RESET_B1="${COMM_EFF_OPT_RESET_B1:-0.8}"
  export COMM_EFF_OPT_RESET_B2="${COMM_EFF_OPT_RESET_B2:-0.95}"
  export COMM_EFF_OPT_RESET_SCALE_MATCH="${COMM_EFF_OPT_RESET_SCALE_MATCH:-true}"
elif [[ "$ARM" == "freshm" ]]; then
  # ARM D: the commeff codec block VERBATIM (kept a separate branch so arm A
  # stays byte-identical to its reference launchers), plus the one number that
  # stops the frozen M being reused between anchor fires.
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
  # THE ONE DELTA against arm A. The anchor cadence is pinned to its arm-A value
  # here (rather than left to the base launcher) purely so the merger cadence can
  # be DERIVED from it: the two must be equal or a correction lands on a tick
  # whose M is stale again, which is the thing this arm removes.
  export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
  export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-$COMM_EFF_ANCHOR_CADENCE}"
  if [[ "$COMM_EFF_SPECTRAL_CADENCE" != "$COMM_EFF_ANCHOR_CADENCE" ]]; then
    echo "FATAL: freshm requires spectral cadence == anchor cadence, got" >&2
    echo "       COMM_EFF_SPECTRAL_CADENCE=$COMM_EFF_SPECTRAL_CADENCE vs" >&2
    echo "       COMM_EFF_ANCHOR_CADENCE=$COMM_EFF_ANCHOR_CADENCE. Unequal cadences put" >&2
    echo "       corrections back on ticks where M is stale, which is arm A's behaviour." >&2
    exit 1
  fi
  # A cadence the anchor cadence is not a multiple of is rejected by the config
  # validator anyway; catch it here with a message that names the arm.
  if (( COMM_EFF_ANCHOR_CADENCE % COMM_EFF_SPECTRAL_CADENCE != 0 )); then
    echo "FATAL: anchor.cadence must be divisible by spectral.cadence" >&2
    exit 1
  fi
  # alpha=1.0 would silently make this the nosign arm under the freshm name.
  if [[ "${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}" == "1.0" ]]; then
    echo "FATAL: freshm keeps signed_ema_alpha at its arm-A value; alpha=1.0 disables the" >&2
    echo "       merger entirely, which is the 'nosign' arm. Run that arm instead." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_SPECTRAL_ENABLED:-true}" != "true" ]]; then
    echo "FATAL: freshm requires COMM_EFF_SPECTRAL_ENABLED=true (the merger still runs," >&2
    echo "       just only on fire ticks). spectral.enabled=false is rejected by the" >&2
    echo "       validator under rank1_relex and deletes the anchor circuit as well." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_OPT_RESET_ENABLED:-false}" != "false" ]]; then
    echo "FATAL: freshm requires COMM_EFF_OPT_RESET_ENABLED=false (no optimizer-state swap)." >&2
    echo "       Run the 'optreset' arm if that is what is wanted." >&2
    exit 1
  fi
elif [[ "$ARM" == "powersgdq" ]]; then
  # ARM F: arm D's circuit with the CODEC swapped from the PRF mask back to
  # PowerSGD with an anchor-owned basis Q. The fresh-M merger stays exactly as
  # arm D has it (spectral.cadence == anchor.cadence), so the only change from
  # the freshm run is how the boundary activations are compressed.
  #
  # How Q moves, and why this is the "normal" PowerSGD arrangement:
  #   sketch   V += M^T (M Q)      accumulated under no_grad on the anchor's
  #                                stale-weight forward (NOT the fast path)
  #   update   Q <- orth(V)        fp32 QR, then staged and broadcast to every
  #                                DP rank, published after the PPO minibatches
  # That is block power iteration on the activation Gram matrix M^T M, whose
  # fixed point is the top-r right singular subspace. With owns_q=true the fast
  # path is a READ-ONLY consumer: its sketch accumulation and its end-of-step
  # orth(V) are both gated off, so Q advances ONLY when the anchor fires. The
  # basis therefore rides the slow circuit, which is the whole point when the
  # stage boundary crosses the open internet.
  export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
  export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-powersgd}"
  export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"          # the anchor is the ONLY Q writer
  export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-true}"
  export COMM_EFF_POWERSGD_SYNC_BASIS="${COMM_EFF_POWERSGD_SYNC_BASIS:-true}"
  export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE="${COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE:-true}"
  export COMM_EFF_POWERSGD_COMPRESS_REFERENCE="${COMM_EFF_POWERSGD_COMPRESS_REFERENCE:-true}"
  export COMM_EFF_POWERSGD_WARM_START="${COMM_EFF_POWERSGD_WARM_START:-true}"
  export COMM_EFF_POWERSGD_QR_DTYPE="${COMM_EFF_POWERSGD_QR_DTYPE:-fp32}"
  export COMM_EFF_POWERSGD_PP_SIZE="${COMM_EFF_POWERSGD_PP_SIZE:-8}"
  # RANK = 128, NOT the 77 that is this project's default. 77 was chosen at the
  # 1.5B model's hidden 1536, where it is 5.0 percent of the wire. Qwen3-4B has
  # hidden 2560, and both codecs report their cost in the SAME unit, coordinates
  # per token per boundary:
  #     PRF exact-k : (1 - p) * H = 0.05 * 2560 = 128   (logical_pp_bytes_prf)
  #     PowerSGD    : r                                  (logical_pp_bytes_powersgd_y_only)
  # so r=128 keeps the freshm run's exact budget and r=77 would silently tighten
  # it to 3.0 percent, confounding the codec change with a bandwidth change.
  # Set COMM_EFF_POWERSGD_RANK=77 to run the project-default rank instead.
  export COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-128}"
  # THE FRESH-M MERGER, IDENTICAL TO ARM D. Derived from the anchor cadence for
  # the same reason: unequal cadences put corrections back on stale-M ticks.
  export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
  export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-$COMM_EFF_ANCHOR_CADENCE}"
  if [[ "$COMM_EFF_SPECTRAL_CADENCE" != "$COMM_EFF_ANCHOR_CADENCE" ]]; then
    echo "FATAL: powersgdq inherits freshm's merger: spectral cadence must equal anchor cadence," >&2
    echo "       got $COMM_EFF_SPECTRAL_CADENCE vs $COMM_EFF_ANCHOR_CADENCE." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}" == "1.0" ]]; then
    echo "FATAL: powersgdq keeps signed_ema_alpha at its arm-A value; 1.0 disables the merger." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_ANCHOR_ENABLED:-true}" != "true" ]]; then
    echo "FATAL: powersgdq requires COMM_EFF_ANCHOR_ENABLED=true: with owns_q the anchor is the" >&2
    echo "       only thing that ever updates Q, so a disabled anchor freezes the basis at its" >&2
    echo "       random seeded bootstrap for the whole run." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_OPT_RESET_ENABLED:-false}" != "false" ]]; then
    echo "FATAL: powersgdq requires COMM_EFF_OPT_RESET_ENABLED=false (no optimizer-state swap)." >&2
    exit 1
  fi
elif [[ "$ARM" == "nosign" ]]; then
  # ARM E: the commeff codec block VERBATIM (kept a separate branch so arm A
  # stays byte-identical to its reference launchers), plus the one number that
  # turns the signed-EMA merger into an identity.
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
  # THE ONE DELTA against arm A. Everything above and every anchor/RELEX default
  # in the base launcher is untouched.
  export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-1.0}"
  # Guard the two knobs whose flipping would silently make this a different
  # experiment: the merger has to stay wired in (validator + anchor circuit),
  # and the optimizer-state swap has to stay off (that is arm C, not this).
  if [[ "${COMM_EFF_SPECTRAL_ENABLED:-true}" != "true" ]]; then
    echo "FATAL: nosign requires COMM_EFF_SPECTRAL_ENABLED=true. alpha=1.0 is how the" >&2
    echo "       merger is disabled here; spectral.enabled=false additionally deletes the" >&2
    echo "       anchor circuit and is rejected by the validator under rank1_relex." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_OPT_RESET_ENABLED:-false}" != "false" ]]; then
    echo "FATAL: nosign requires COMM_EFF_OPT_RESET_ENABLED=false (no optimizer-state swap)." >&2
    echo "       Run the 'optreset' arm if that is what is wanted." >&2
    exit 1
  fi
elif [[ "$ARM" == "delayedef" || "$ARM" == "delayedef-fresh" || "$ARM" == "delayedef-cad10" || "$ARM" == "delayedef-anneal" ]]; then
  # ARM G (delayedef): the commeff codec block VERBATIM (kept a separate branch
  # so arm A stays byte-identical to its reference launchers), with the MERGER
  # swapped from signed_ema to delayed_ef. Everything else in arm A is untouched:
  # PRF exact-k p=0.95 codec, anchor cadence/delay 20/20, rank-1 RELEX W2
  # strength 1, and (for arm G) spectral.cadence=1 so the held residual is
  # re-applied on every one of the 19 ticks between fires, the exact analog of
  # arm A's stale-M reuse. The merger becomes
  #     delta      = M - G_comp        refreshed ONCE per anchor fire
  #     G_corr(t)  = G_comp(t) + lambda * delta
  # with lambda=1 and beta_anc=0 (M is the raw fresh anchor gradient, no EMA
  # history), so ON the fire tick G_corr = G_anchor exactly: the anchor's dense
  # gradient (computed at the RELEX-projected weights on the SAME batch as the
  # fast gradient) replaces the compressed one outright. signed_ema_alpha is
  # UNREAD in this mode.
  #
  # ARM H (delayedef-fresh): arm G with spectral.cadence 1 -> 20 locked to the
  # anchor cadence, the same one-number change freshm made to arm A. The held
  # delta is then never re-applied stale: every 20th tick the gradient IS the
  # anchor gradient, all other ticks are pure G_comp.
  #
  # ARM J (delayedef-cad10): arm H with spectral.cadence 20 -> 10, the INTERIOR
  # point of the stale-dose response curve. Corrections run on ticks 10,20,30,...:
  # even correction ticks coincide with anchor fires (delta refreshes, so
  # G_corr = G_anchor exactly, as in arm H), odd ones (30, 50, ...) re-apply the
  # HELD delta exactly once, 10 ticks stale. Dose per 20-tick interval: arm G 19,
  # arm J 1, arm H 0. Arm G (dose 19) collapsed via Mode C, arm H (dose 0)
  # finished 500 clean at +7.02pt; arm J prices dose 1. Tick 10 is a cold-M
  # no-op by construction (spectral_filter returns G_comp untouched before the
  # first fire). Runtime fingerprint: delayed_ef_held : delayed_ef_refreshed
  # must read 1:1 (arm G was 19:1, arm H 0:1), and mid-interval correction
  # ticks show grad_norm ~1.4x neighbors (delta's norm is dominated by the
  # anti-correlated tick-20 compressed noise; expected, not a fault).
  # The anchor circuit itself is NOT a knob on this surface: cadence and delay
  # model the slow network path (CLAUDE.md), so both are pinned to 20 below.
  #
  # ARM K (delayedef-anneal): arm G with ONE new number, delayed_ef_decay
  # 1.0 -> 0.75. The held delta is re-applied on every tick between fires (the
  # arm-G schedule, spectral.cadence=1) but its weight ANNEALS geometrically:
  #     G_corr(t) = G_comp(t) + lambda * decay^age * delta
  # with age = ticks since the fire (0 on the fire tick, so the fire tick still
  # returns G_anchor exactly). Interval impulse Sum decay^a ~ 4 units (1 fresh
  # + ~3 decaying stale, concentrated at ages 1-4 where delta is most valid)
  # against arm G's 20 undecayed units. Under the directional-persistence
  # reading of the collapses this kills the standing direction geometrically
  # instead of letting it stand 19 ticks; it is the direct attempt to buy arm
  # G's early speed (+5.8pt val@100 over arm H) without arm G's death. The
  # decay endpoints are identities of existing arms (1.0 = arm G bitwise,
  # 0.0 = arm H's fire-only dose), so d=0.75 is a true interior point of the
  # same one-parameter family.
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
  # THE MERGER SWAP. beta_anc=0 is part of it: delayed_ef consumes M as "the
  # latest fire's raw dense anchor gradient", and any EMA history in M would
  # smear older fires into the residual.
  export COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-delayed_ef}"
  export COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA="${COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA:-1.0}"
  export COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.0}"
  if [[ "$ARM" == "delayedef-fresh" ]]; then
    # THE ONE DELTA against arm G, identical in mechanism to freshm vs arm A.
    export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
    export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-$COMM_EFF_ANCHOR_CADENCE}"
    if [[ "$COMM_EFF_SPECTRAL_CADENCE" != "$COMM_EFF_ANCHOR_CADENCE" ]]; then
      echo "FATAL: delayedef-fresh requires spectral cadence == anchor cadence, got" >&2
      echo "       COMM_EFF_SPECTRAL_CADENCE=$COMM_EFF_SPECTRAL_CADENCE vs" >&2
      echo "       COMM_EFF_ANCHOR_CADENCE=$COMM_EFF_ANCHOR_CADENCE. Unequal cadences put" >&2
      echo "       the residual back on ticks where delta is stale, which is arm G." >&2
      exit 1
    fi
  elif [[ "$ARM" == "delayedef-cad10" ]]; then
    # THE ONE DELTA against arm H. The pair (anchor=20, spectral=10) is the
    # experiment; any other pair under this arm name is a different experiment
    # smuggled in via env, so it is fixed, not defaulted.
    export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
    export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-10}"
    export COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
    if [[ "$COMM_EFF_SPECTRAL_CADENCE" != "10" || "$COMM_EFF_ANCHOR_CADENCE" != "20" ]]; then
      echo "FATAL: delayedef-cad10 is the fixed interior dose point spectral.cadence=10" >&2
      echo "       under anchor.cadence=20 (ONE stale delta re-application per interval)." >&2
      echo "       Got spectral=$COMM_EFF_SPECTRAL_CADENCE anchor=$COMM_EFF_ANCHOR_CADENCE." >&2
      echo "       A different ratio is a different dose; add a new arm for it." >&2
      exit 1
    fi
    if [[ "$COMM_EFF_ANCHOR_DELAY_K" != "20" ]]; then
      echo "FATAL: the anchor delay models the slow network path and is NOT tunable" >&2
      echo "       (CLAUDE.md). delayedef-cad10 requires COMM_EFF_ANCHOR_DELAY_K=20," >&2
      echo "       got $COMM_EFF_ANCHOR_DELAY_K." >&2
      exit 1
    fi
  elif [[ "$ARM" == "delayedef-anneal" ]]; then
    # THE ONE DELTA against arm G: decay 1.0 -> 0.75, on arm G's every-tick
    # correction schedule. The decay value is the experiment; 1.0 and 0.0 are
    # identities of arms G and H respectively, and any other value is a
    # different interior point. Pin it like cad10 pins its cadence pair.
    export COMM_EFF_SPECTRAL_DELAYED_EF_DECAY="${COMM_EFF_SPECTRAL_DELAYED_EF_DECAY:-0.75}"
    if [[ "$COMM_EFF_SPECTRAL_DELAYED_EF_DECAY" != "0.75" ]]; then
      echo "FATAL: delayedef-anneal is the fixed interior point delayed_ef_decay=0.75" >&2
      echo "       (decay=1.0 is arm G bitwise, decay=0.0 is arm H's fire-only dose)." >&2
      echo "       Got $COMM_EFF_SPECTRAL_DELAYED_EF_DECAY. A different decay is a" >&2
      echo "       different experiment; add a new arm for it." >&2
      exit 1
    fi
    if [[ "${COMM_EFF_SPECTRAL_CADENCE:-1}" != "1" ]]; then
      echo "FATAL: delayedef-anneal requires spectral.cadence=1 (annealing means the" >&2
      echo "       held delta decays across EVERY tick between fires; a coarser cadence" >&2
      echo "       is a different dose profile)." >&2
      exit 1
    fi
  else
    # Arm G must keep the base launcher's spectral.cadence=1: the stale-reuse
    # dose IS the experiment. A different cadence smuggled in via env would
    # silently run a third experiment under arm G's name.
    if [[ "${COMM_EFF_SPECTRAL_CADENCE:-1}" != "1" ]]; then
      echo "FATAL: delayedef requires spectral.cadence=1 (stale delta re-applied every tick," >&2
      echo "       the arm-A analog). For fire-tick-only application run 'delayedef-fresh'." >&2
      exit 1
    fi
  fi
  if [[ "$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA" == "0.0" || "$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA" == "0" ]]; then
    echo "FATAL: delayed_ef at lambda=0 is a bitwise identity, i.e. the 'nosign' experiment" >&2
    echo "       under a delayedef name. Run that arm instead." >&2
    exit 1
  fi
  if [[ "$ARM" != "delayedef-anneal" && "${COMM_EFF_SPECTRAL_DELAYED_EF_DECAY:-1.0}" != "1.0" ]]; then
    echo "FATAL: arm '$ARM' requires delayed_ef_decay=1.0 (constant held weight). An" >&2
    echo "       annealed residual under a non-anneal arm name is a mislabelled run;" >&2
    echo "       run 'delayedef-anneal' instead. Got ${COMM_EFF_SPECTRAL_DELAYED_EF_DECAY}." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_SPECTRAL_ENABLED:-true}" != "true" ]]; then
    echo "FATAL: delayedef arms require COMM_EFF_SPECTRAL_ENABLED=true. spectral.enabled=false" >&2
    echo "       is rejected by the validator under rank1_relex and deletes the anchor circuit." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_OPT_RESET_ENABLED:-false}" != "false" ]]; then
    echo "FATAL: delayedef arms require COMM_EFF_OPT_RESET_ENABLED=false (no optimizer-state swap)." >&2
    exit 1
  fi
elif [[ "$ARM" == "blend" ]]; then
  # ARM I (blend): the commeff codec block VERBATIM, with the MERGER swapped to
  # the norm-matched convex value blend from the pre-fork menu (EXP-30 B1):
  #     G_corr = (1-eta) * G_comp + eta * (||G_comp||/||M||) * M
  # One delta from the delayedef arm: the merge OPERATOR on the identical M
  # signal (same codec, same anchor 20/20, same RELEX W2 strength 1, same
  # beta_anc=0 fresh-per-fire M, same spectral.cadence=1). Unlike delayed_ef
  # there is NO held residual: the blend is re-formed against each tick's fresh
  # G_comp, M alone carries the staleness, and the update is convex so
  # ||G_corr|| <= ||G_comp|| (never injects energy). Unlike signed_ema it is a
  # VALUE merger: no sign transplant, real heterogeneous per-coordinate
  # magnitudes reach Adam, which breaks ingredient (i) of the verified
  # sign-railgun mechanism.
  #
  # eta=0.3 is the only eta ever paired with a VALID anchor signal on a val
  # protocol (EXP-30 B1: val@50 0.7422 vs delayed_ef 0.7528, dense 0.7839,
  # no-merge floor 0.6300, at 1.5B and anchor cadence 5). Blend at anchor
  # cadence 20/20 is untested territory. Override with
  # COMM_EFF_SPECTRAL_BLEND_ETA.
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
  # THE MERGER SWAP. beta_anc=0 for the same reason as the delayedef arms: M
  # must be the latest fire's raw dense anchor gradient (EXP-31: no extra
  # memory on the correction path, and EXP-30 B1 ran beta_anc=0).
  export COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-blend}"
  export COMM_EFF_SPECTRAL_BLEND_ETA="${COMM_EFF_SPECTRAL_BLEND_ETA:-0.3}"
  export COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.0}"
  # Arm I keeps spectral.cadence=1: the blend consumes M every tick (stale
  # between fires), the same dose surface as arms A and G. A fresh-only blend
  # would be a different arm; refuse the knob rather than run it mislabelled.
  if [[ "${COMM_EFF_SPECTRAL_CADENCE:-1}" != "1" ]]; then
    echo "FATAL: blend requires spectral.cadence=1 (M consumed every tick, the arm-A/G" >&2
    echo "       dose surface). A fire-tick-only blend is a different experiment; add it" >&2
    echo "       as its own arm rather than overriding the cadence." >&2
    exit 1
  fi
  if [[ "$COMM_EFF_SPECTRAL_BLEND_ETA" == "0.0" || "$COMM_EFF_SPECTRAL_BLEND_ETA" == "0" ]]; then
    echo "FATAL: blend at eta=0 is a bitwise identity, i.e. the 'nosign' experiment under" >&2
    echo "       a blend name. Run that arm instead." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_SPECTRAL_ENABLED:-true}" != "true" ]]; then
    echo "FATAL: blend requires COMM_EFF_SPECTRAL_ENABLED=true. spectral.enabled=false is" >&2
    echo "       rejected by the validator under rank1_relex and deletes the anchor circuit." >&2
    exit 1
  fi
  if [[ "${COMM_EFF_OPT_RESET_ENABLED:-false}" != "false" ]]; then
    echo "FATAL: blend requires COMM_EFF_OPT_RESET_ENABLED=false (no optimizer-state swap)." >&2
    exit 1
  fi
elif [[ "$ARM" == "dense" ]]; then
  # ARM B: the control. The literal quoted arm name matters here: the chain
  # script's arm-exists gate greps the launcher for "dense" before launching.
  export COMM_EFF_ENABLED=false
else
  echo "FATAL: arm '$ARM' passed the case list but has no config block." >&2
  exit 1
fi

# The anchor DELAY models the latency of the slow dense path, which the network
# sets, not the experimenter (CLAUDE.md). Every arm on this surface runs
# delay_K=20; refuse any env override rather than run a mislabelled surface.
if [[ "${COMM_EFF_ENABLED:-false}" == "true" && "${COMM_EFF_ANCHOR_DELAY_K:-20}" != "20" ]]; then
  echo "FATAL: COMM_EFF_ANCHOR_DELAY_K=$COMM_EFF_ANCHOR_DELAY_K. The anchor delay is set" >&2
  echo "       by the network model and is NOT a knob on the qwen3-4b-4k-500 surface;" >&2
  echo "       all arms run delay_K=20." >&2
  exit 1
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
# WandB: the key lives in the SECRETS FILE the engine sources later, not in this
# launcher's env, so check there before falling back to offline. Forcing offline
# here because the key is not exported yet silently downgrades the run's main
# monitoring surface (it did, once).
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qE '^(export )?WANDB_API_KEY=.+' "$SECRETS_FILE" 2>/dev/null; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

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
elif [[ "$ARM" == "freshm" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + spectral_cadence=$COMM_EFF_SPECTRAL_CADENCE == anchor_cadence=$COMM_EFF_ANCHOR_CADENCE, alpha=${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25} (sign correction on FIRE TICKS ONLY, stale M never reused, no opt-state swap)"
elif [[ "$ARM" == "powersgdq" ]]; then
  CODEC_LINE="powersgd rank=$COMM_EFF_POWERSGD_RANK ($(python3 -c "print(f'{100.0*$COMM_EFF_POWERSGD_RANK/2560:.1f}')")% of hidden 2560, matches the PRF arms' 128 coords/token) anchor_owns_q=$COMM_EFF_ANCHOR_OWNS_Q fast_q_bootstrap=$COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP sync_basis=$COMM_EFF_POWERSGD_SYNC_BASIS pp=$COMM_EFF_POWERSGD_PP_SIZE + spectral_cadence=$COMM_EFF_SPECTRAL_CADENCE == anchor_cadence=$COMM_EFF_ANCHOR_CADENCE (Q via block power iteration on the ANCHOR only; fresh-M merger; no opt-state swap)"
elif [[ "$ARM" == "nosign" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + signed_ema_alpha=$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA (merger IDENTITY, anchor/RELEX/M unchanged, no opt-state swap)"
elif [[ "$ARM" == "delayedef" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + merger=delayed_ef lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC spectral_cadence=${COMM_EFF_SPECTRAL_CADENCE:-1} (G_corr = G_comp + lambda*(M - G_comp@fire); STALE delta re-applied every tick between fires; no opt-state swap)"
elif [[ "$ARM" == "delayedef-fresh" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + merger=delayed_ef lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC spectral_cadence=$COMM_EFF_SPECTRAL_CADENCE == anchor_cadence=$COMM_EFF_ANCHOR_CADENCE (delta applied on FIRE TICKS ONLY, so every 20th tick G_corr = G_anchor exactly; no opt-state swap)"
elif [[ "$ARM" == "delayedef-cad10" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + merger=delayed_ef lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC spectral_cadence=$COMM_EFF_SPECTRAL_CADENCE anchor_cadence=$COMM_EFF_ANCHOR_CADENCE delay_K=${COMM_EFF_ANCHOR_DELAY_K:-20} (INTERIOR DOSE: fresh delta on fire ticks, held delta re-applied ONCE at +10; held:refreshed must read 1:1; no opt-state swap)"
elif [[ "$ARM" == "delayedef-anneal" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + merger=delayed_ef lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA decay=$COMM_EFF_SPECTRAL_DELAYED_EF_DECAY beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC spectral_cadence=${COMM_EFF_SPECTRAL_CADENCE:-1} (ANNEALED DOSE: G_corr = G_comp + lambda*decay^age*delta, fire tick = G_anchor exactly, interval impulse ~4 vs arm G's 20; no opt-state swap)"
elif [[ "$ARM" == "blend" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + merger=blend eta=$COMM_EFF_SPECTRAL_BLEND_ETA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC spectral_cadence=${COMM_EFF_SPECTRAL_CADENCE:-1} (G_corr = (1-eta)*G_comp + eta*(||G_comp||/||M||)*M, convex, no held residual, no sign transplant; no opt-state swap)"
elif [[ "$ARM" == "optreset" ]]; then
  CODEC_LINE="$COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE + opt_reset cadence=$COMM_EFF_OPT_RESET_CADENCE mode=$COMM_EFF_OPT_RESET_MODE b1=$COMM_EFF_OPT_RESET_B1 b2=$COMM_EFF_OPT_RESET_B2 scale_match=$COMM_EFF_OPT_RESET_SCALE_MATCH"
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
