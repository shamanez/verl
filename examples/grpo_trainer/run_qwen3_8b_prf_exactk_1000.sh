#!/usr/bin/env bash
# run_qwen3_8b_prf_exactk_1000.sh
#
# ONE-COMMAND bring-up and launch of the 1000-step communication-efficient GRPO
# run on Qwen3-8B-Base at 16384 total context, 8x H200, MATH, PRF exact-k.
#
# Nothing is authored on the box: this script obtains the checkout, installs it,
# prepares the data, gates the science, gates the hardware, patches the base
# launcher, and execs it.
#
#   bash examples/grpo_trainer/run_qwen3_8b_prf_exactk_1000.sh
#
# Every knob below is env-overridable, and any trailing arguments are forwarded
# verbatim to Hydra (this launcher's own overrides come first, so a caller's
# duplicate key wins -- Hydra applies CLI overrides in order, last one wins).
#
# Surface, and the deltas from the project default:
#   model                Qwen2.5-Math-1.5B -> Qwen3-8B-Base   (36 layers, hidden 4096)
#   context              1024/3072 -> 1024/15360              (16384 total)
#   train / mini batch   512/256 -> 128/128                   (one on-policy tick per generation)
#   steps                100 -> 1000, val at 0/150/300/.../900
#   checkpoints          off -> every 200 steps, mirrored to R2 and deleted locally
#   codec                UNCHANGED: prf_mask, p=0.95, exact-k, constant rescale
#
# At hidden 4096 exact-k keeps exactly 205 of 4096 coordinates per token per
# boundary (3280 bits/token/boundary). With 36 layers over 8 pipeline stages the
# boundaries are decoder layers [4, 9, 14, 19, 23, 27, 31]. Both facts are
# asserted by the money gate below before a single GPU is touched.
#
# Run inside tmux. The engine redirects training to $LOG, which is the heartbeat
# log the harness registers as run.json's remote_log.
set -uo pipefail

# ---------------------------- knobs ----------------------------------------
RUN_ID="${RUN_ID:-96-qwen3-8b-prf-exactk-16k-1000}"
BRANCH="${BRANCH:-exp/96-qwen3-8b-prf-exactk-16k-1000}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
WORK="${WORK:-/workspace}"

# Money gate. There are TWO independent `${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}`
# defaults on the path to Hydra (run_qwen25_math_1p5b_rank1_relex_fsdp.sh and
# vast_comm_eff_engine_grpo.sh), so an unset MODEL_PATH does not fail -- it
# silently trains the 1.5B model for a thousand steps. Set it explicitly here.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B-Base}"

# Context and batch. These four are BARE `export NAME=value` lines in the base
# launcher (no ${VAR:-default}), so exporting them here would be overwritten;
# they have to be patched into a generated copy. See section 5.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Box gates.
EXPECT_GPUS="${EXPECT_GPUS:-8}"
MIN_RAM_GIB="${MIN_RAM_GIB:-1400}"     # 1.4 TiB; the anchor keeps a CPU clone + CPU snapshots
MIN_DISK_GIB="${MIN_DISK_GIB:-700}"    # 8B checkpoints stage locally before the R2 mirror
# SKIP_BOX_GATES=1 downgrades the RAM and disk gates to loud warnings.
SKIP_BOX_GATES="${SKIP_BOX_GATES:-0}"
# ---------------------------------------------------------------------------

RUN_DIR="$WORK/runs/$RUN_ID"
mkdir -p "$RUN_DIR" || { echo "FATAL: cannot create $RUN_DIR" >&2; exit 1; }
cd "$WORK" || { echo "FATAL: cannot cd $WORK" >&2; exit 1; }

echo "=== $RUN_ID: Qwen3-8B-Base / PRF exact-k / 16384 ctx / 1000 steps ==="

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
    || box_gate_fail "only ${DISK_GIB} GiB free on $WORK, need ${MIN_DISK_GIB} GiB for checkpoint staging" soft
else
  box_gate_fail "could not read free disk for $WORK" soft
fi

AWS_BIN="$(command -v aws 2>/dev/null)"
echo "--- preflight aws:   ${AWS_BIN:-MISSING}"
[[ -n "$AWS_BIN" ]] \
  || box_gate_fail "'aws' is not on PATH but CKPT_R2_ENABLED is on — the R2 checkpoint mirror cannot run"

# ---------------------------------------------------------------------------
# 2. Obtain the verl checkout (fast path reuse, else shallow clone).
#    ORDERING IS LOAD-BEARING: every reset/checkout happens HERE, and the money
#    gate in section 4 runs on the FINAL tree. run_layer_rotation_300.sh gated
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
# 3. Editable install (no deps) if verl is not importable yet.
# ---------------------------------------------------------------------------
python3 -c "import verl" 2>/dev/null || {
  if command -v uv >/dev/null 2>&1; then uv pip install --no-deps -e . ; else python3 -m pip install --no-deps -e . ; fi
}
python3 -c "import verl" || { echo "FATAL: verl import failed after install" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4. Money gate, on the FINAL checkout. Three claims, all cheap and all CPU:
#      a) the exact-k lever exists at all;
#      b) at Qwen3-8B-Base's hidden 4096 it keeps exactly 205 coordinates per
#         token (the 95 percent budget), so the wire budget is what we think;
#      c) the anchor clone inherits gradient checkpointing. Without that fix the
#         anchor's dense replay forward stores every activation: measured
#         +62.9 GiB at the first fire on a 1.5B model (run 90, step 20). A stale
#         checkout silently OOMs at 8B/16k, so refuse to start on one.
#    Also prints the 36-layer/8-stage boundary set for the run log.
# ---------------------------------------------------------------------------
python3 - <<'PY' || { echo "FATAL: money gate FAILED — this checkout must not be launched" >&2; exit 1; }
import inspect
import os

import torch

from verl.workers.comm_eff.activation_mask import prf_token_mask
from verl.workers.comm_eff.anchor import build_anchor_module
from verl.workers.comm_eff.boundary import decoder_boundary_indices

assert "exact_k" in inspect.signature(prf_token_mask).parameters, "prf_token_mask is missing exact_k"

H = 4096  # Qwen3-8B-Base hidden size
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
assert kept == 205, f"exact-k at hidden {H} kept {kept} coordinates, expected 205"

anchor_src = inspect.getsource(build_anchor_module)
assert "gradient_checkpointing_enable" in anchor_src, (
    "build_anchor_module does NOT enable gradient checkpointing on the anchor clone. "
    "This checkout predates that fix; at 8B/16k the first anchor fire will OOM. "
    "Refusing to launch."
)

boundaries = decoder_boundary_indices(36, 8)
assert boundaries == [4, 9, 14, 19, 23, 27, 31], f"unexpected boundary set {boundaries}"

print(f"OK: exact-k keeps {kept}/{H} coords per token per boundary ({kept * 16} bits/token/boundary)")
print("OK: anchor clone inherits gradient checkpointing")
print(f"OK: 36 layers over 8 stages -> boundaries {boundaries}")

# Per-boundary masked fractions (run 97's dense-middle lever). Gated on the
# env var so an unset run (a run 96 replay) takes a byte-identical gate. When
# set, the vector must be one float in [0, 1] per boundary, and every 0.0
# entry must provably be an identity at this run's hidden size: with
# exact_k=True, p=0.0 keeps round(1.0*H) = H channels and prf_token_mask
# early-returns all-ones, while the hook recomputes the constant-rescale gain
# per boundary as 1/(1-0.0) = 1.0.
p_by_boundary_raw = os.environ.get("COMM_EFF_MASK_P_BY_BOUNDARY", "").strip()
if p_by_boundary_raw:
    inner = p_by_boundary_raw
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    entries = [tok.strip() for tok in inner.split(",")]
    assert entries and all(entries), (
        f"COMM_EFF_MASK_P_BY_BOUNDARY={p_by_boundary_raw!r} parsed to an empty entry"
    )
    p_vec = []
    for tok in entries:
        try:
            val = float(tok)
        except ValueError as exc:
            raise AssertionError(f"COMM_EFF_MASK_P_BY_BOUNDARY entry {tok!r} is not a float") from exc
        # Strictly below 1.0: a 1.0 entry keeps round(0*H)=0 channels and the
        # hook clamps its gain to 1.0, so the run would train through an
        # all-zero boundary without any other gate tripping.
        assert 0.0 <= val < 1.0, f"COMM_EFF_MASK_P_BY_BOUNDARY entry {val} is outside [0, 1)"
        p_vec.append(val)
    # Validate the length against the pp_size this run will actually use, not
    # the hardcoded 8-stage set above: a caller overriding COMM_EFF_MASK_PP_SIZE
    # would otherwise pass here and die in ActivationMasker.register() after
    # Ray boot and the 8B model pull.
    pp_size = int(os.environ.get("COMM_EFF_MASK_PP_SIZE", "8"))
    run_boundaries = decoder_boundary_indices(36, pp_size)
    assert len(p_vec) == len(run_boundaries), (
        f"COMM_EFF_MASK_P_BY_BOUNDARY has {len(p_vec)} entries but 36 layers over "
        f"{pp_size} stages give {len(run_boundaries)} boundaries {run_boundaries}. "
        "Supply exactly one p per boundary."
    )
    for layer_idx, p_i in zip(run_boundaries, p_vec):
        if p_i == 0.0:
            dense_mask = prf_token_mask(
                sample_ids=torch.zeros(1, dtype=torch.int64),
                position_ids=torch.zeros(1, dtype=torch.int64),
                layer_idx=layer_idx,
                global_step=1,
                base_seed=0,
                hidden_size=H,
                p=0.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
                exact_k=True,
            )
            assert bool((dense_mask == 1.0).all()), (
                f"p=0.0 with exact_k=True did NOT return an all-ones mask at boundary layer "
                f"{layer_idx} (H={H}). The dense-cut identity does not hold on this checkout."
            )
    print(f"OK: p_by_boundary supplies {len(p_vec)} per-boundary fractions, cut map (layer -> p):")
    for layer_idx, p_i in zip(run_boundaries, p_vec):
        tag = f"  DENSE (identity proved at H={H})" if p_i == 0.0 else ""
        print(f"    cut after layer {layer_idx:>2}: p={p_i}{tag}")
PY

# ---------------------------------------------------------------------------
# 5. MATH parquet ($HOME/data/math), prepared with the canonical research prep.
# ---------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: MATH parquet unavailable in $DATA_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 6. Patched launcher copy. Only the four bare scalars change. Each one is
#    checked against its EXPECTED CURRENT value in the base first (so a drifted
#    base fails loud instead of being silently un-patched) and asserted again in
#    the generated copy afterwards.
# ---------------------------------------------------------------------------
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_qwen3_8b_prf_exactk_1000.gen.sh"
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
# 7. Model + codec + run controls. The codec block is the project default and is
#    IDENTICAL to run_prf_exactk_600.sh: prf_mask, p=0.95, exact-k, constant
#    rescale, masking the train forward + the old-logprob recompute + the
#    reference forward, 8 pipeline stages, no anchor-owned Q.
# ---------------------------------------------------------------------------
export MODEL_PATH
export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-prf_mask}"
export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
export COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"
export COMM_EFF_MASK_RESCALE_MODE="${COMM_EFF_MASK_RESCALE_MODE:-constant}"
export COMM_EFF_MASK_EXACT_K="${COMM_EFF_MASK_EXACT_K:-true}"
export COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"
export COMM_EFF_MASK_REFERENCE="${COMM_EFF_MASK_REFERENCE:-true}"
export COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"
export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-false}"          # prf_mask has no basis Q to own
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"

# Run schedule. 1000 steps at batch 128 needs >=10 MATH epochs, so the step cap
# is the stopping condition; 20 leaves margin after prompt filtering.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export TEST_FREQ="${TEST_FREQ:-150}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export SAVE_FREQ="${SAVE_FREQ:-200}"

# Checkpoints -> R2, deleted locally after a verified upload, so peak local disk
# stays at roughly one in-flight 8B checkpoint plus staging.
export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-true}"
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-120}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-8}"

# Token budgets. The log-prob paths default to 36864, which at 16k context and
# 8B is a much larger live activation set than the 4k protocol they were sized
# for; 18432 matches the actor's own per-GPU budget.
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-18432}"
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-18432}"
export ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_ID}"
export PROJECT_NAME="${PROJECT_NAME:-$RUN_ID}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"
export LOG="${LOG:-$RUN_DIR/train.log}"
# WandB: use it if a key is present, else fall back to offline (no crash).
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------------------------------------------------------------------
# 8. Hydra overrides this launcher owns.
#
#    max_model_len / max_num_batched_tokens are REQUIRED, not cosmetic. Neither
#    key is emitted by vast_comm_eff_engine_grpo.sh, so both fall back to their
#    rollout.yaml defaults: max_model_len=null (vLLM then uses the checkpoint's
#    max_position_embeddings, 32768 for Qwen3-8B-Base) and
#    max_num_batched_tokens=8192. With enable_chunked_prefill=False hardcoded in
#    the engine script, vLLM refuses to boot when max_num_batched_tokens is below
#    max_model_len. Both keys already exist in the schema, so they take no `+`
#    prefix, and passing them once here creates no duplicate.
#
#    "$@" goes LAST so a caller's override of the same key wins (Hydra applies
#    CLI overrides in order).
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES=(
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
)

cat <<EOF
=== launching $RUN_ID ===
    model            $MODEL_PATH
    context          $MAX_PROMPT_LENGTH prompt + $MAX_RESPONSE_LENGTH response = $TOTAL_CTX
    batch            train $TRAIN_BATCH_SIZE / mini $PPO_MINI_BATCH_SIZE (one on-policy tick per step)
    codec            $COMM_EFF_COMPRESSION_TYPE p=$COMM_EFF_MASK_P exact_k=$COMM_EFF_MASK_EXACT_K rescale=$COMM_EFF_MASK_RESCALE_MODE pp=$COMM_EFF_MASK_PP_SIZE p_by_boundary=${COMM_EFF_MASK_P_BY_BOUNDARY:-}
    schedule         $TOTAL_TRAINING_STEPS steps, test_freq $TEST_FREQ, save_freq $SAVE_FREQ, R2=$CKPT_R2_ENABLED
    rollout          gpu_mem $ROLLOUT_GPU_MEM_UTIL, max_model_len $MAX_MODEL_LEN
    log              $LOG
    hydra            ${HYDRA_OVERRIDES[*]} $*
EOF

exec bash "$PATCHED" "${HYDRA_OVERRIDES[@]}" "$@"
