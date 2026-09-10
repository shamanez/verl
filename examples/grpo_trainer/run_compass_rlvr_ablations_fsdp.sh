#!/usr/bin/env bash
# run_compass_rlvr_ablations_fsdp.sh
#
# The four ablations the COMPASS RLVR section is missing, all on Pair 1
# (Qwen2.5-Math-1.5B on MATH, 1024/2048, 128 prompts x G=8, prf_mask exact-k
# p=0.95 so k=77 of 1536 over 7 boundaries, anchor c=20 K=20, alpha=0.25,
# beta_anc=0.25, rank-one secant W=2 at full strength).
#
#   A1  compression without the anchor. The paper states outright that its
#       sweep "does not include alpha = 1, and therefore does not establish
#       the gain over a no-anchor condition". Two arms answer two different
#       questions:
#         noanchor  the anchor circuit is removed. No dense replay, no RELEX
#                   forecast, no gradient EMA, no CPU snapshots. This is what
#                   a deployment that cannot afford a dense replica gets.
#         nosign    the anchor still runs and still costs, but alpha=1.0
#                   discards its sign. Matched compute, so the difference
#                   against noanchor prices the anchor's compute separately
#                   from its signal.
#
#   A2  staleness sweep over K at fixed cadence c=20. Note that raising K at
#       fixed c also lengthens the RELEX extrapolation, because the secant
#       increment is strength * slope * horizon and horizon == K. The kNNgm
#       arms cancel that by setting strength = c/K, which separates "the
#       anchor gradient is older" from "the forecast reaches further".
#
#   A3  entropy and response-length trajectories. NOTHING TO RUN. Already in
#       research/runs/90-{dense,prf-exactk}-600/metrics/*_train.log.
#
#   A4  RL throughput. The measured half is a free by-product of these arms
#       (perf/throughput, timing_s/*); the anchor's own tax is
#       (base update_actor) minus (noanchor update_actor).
#
# EVALUATION, two tiers.
#   Tier 1, every arm, free and in-loop: MATH validation on the training path
#     at steps 0/40/80/120/160/200. Step 120 is deliberate: it is the exact
#     instrument the paper's existing anchor-ablation table already uses
#     (base 0.447, rank-one secant 0.671), so every new row is directly
#     readable against the rows already printed. This is the sweep instrument.
#   Tier 2, three checkpoints only, the full ten-benchmark suite run densely
#     through research/scripts/ood_eval/, which is the same harness that
#     produced the paper's main benchmark table. An in-loop out-of-domain
#     probe would NOT be comparable to that table; this harness is. Arms with
#     KEEP_CKPT=1 below save the checkpoints it needs.
#     A single-metric MATH sweep cannot see capability damage that leaves MATH
#     intact, and not damaging the base model is the harder constraint, so the
#     no-anchor arm and the far end of the staleness sweep are evaluated in
#     full rather than on MATH alone.
#
# ONE GPU PER ARM. Pair 1 was measured on a single H200 and the historical
# 600-step pair is single-GPU, so a 4-GPU data-parallel run would all-reduce
# G_anchor across ranks and stop being comparable to the published numbers.
# On a 4x H200 box run four arms side by side instead, one per GPU, with
# GPU=<id>. That is the same aggregate throughput and keeps every arm in the
# same family as the paper's existing Pair 1 rows.
#
# Usage, inside tmux, from /workspace:
#   ARM=base GPU=0 bash examples/grpo_trainer/run_compass_rlvr_ablations_fsdp.sh
#   ARM=noanchor GPU=1 bash examples/grpo_trainer/run_compass_rlvr_ablations_fsdp.sh
# or fan out four at once with run_compass_rlvr_ablations_fanout_fsdp.sh.
#
# Every arm validates against the real CommEffConfig before any GPU work; run
# the CPU gate first with:
#   python3 tests/workers/comm_eff/test_compass_ablation_arms.py
set -uo pipefail

# -------- knobs --------
ARM="${ARM:?set ARM=base|dense|noanchor|nosign|k10|k40|k80|k40gm|k80gm|smoke}"
GPU="${GPU:-0}"                       # single GPU index this arm owns
BRANCH="${BRANCH:-exp/compass-rlvr-ablations}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
WORK="${WORK:-/workspace}"
PROJECT="${PROJECT:-compass-rlvr-ablations}"
GPU_MEM="${GPU_MEM:-0.72}"            # proven on 1x H200 at this geometry
# Horizon. 200 steps is the default: step 120 reproduces the instrument the
# paper's Table 10 already uses, and 200 carries far enough past it to show a
# slope. TEST_FREQ=40 puts validation on 40/80/120/160/200 so 120 is a real
# measured point rather than an interpolation.
TOTAL_STEPS="${TOTAL_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-40}"
# ------------------------

# ---------------------------------------------------------------- arm table
# Each arm is a delta on the Pair 1 protocol. Anything not named here stays at
# the launcher default, which IS the paper's configuration.
unset_anchor() {
  # The validator refuses partial combinations: rank1_relex requires
  # anchor.enabled, replay_paired_batch and spectral.enabled all true, and
  # lookahead_min_snapshots != -1 requires rank1_relex active. So removing the
  # anchor is four coupled flips plus the merger, not one.
  export COMM_EFF_ANCHOR_ENABLED=false
  export COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR=false
  export COMM_EFF_ANCHOR_LOOKAHEAD_MODE=disabled
  export COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=-1
  # Not optional. Left true, the merger allocates one fp32 CPU EMA per floating
  # parameter and all-gathers every gradient every tick for a provably identity
  # transform, which would show up as fake overhead in the A4 timing.
  export COMM_EFF_SPECTRAL_ENABLED=false
}

ARM_DESC=""
KEEP_CKPT=0        # 1 = save checkpoints for the ten-benchmark suite eval
case "$ARM" in
  base)
    ARM_DESC="Pair 1 reference, full anchor at c=20 K=20 (re-run at the new val cadence)"
    KEEP_CKPT=1    # the control every ablation row is read against
    ;;
  dense)
    ARM_DESC="dense control, boundary codec off"
    export COMM_EFF_ENABLED=false
    export COMM_EFF_COMPRESSION_TYPE=dense
    export COMM_EFF_MASK_ENABLED=false
    unset_anchor
    ;;
  noanchor)
    ARM_DESC="A1: compression on, anchor circuit removed"
    unset_anchor
    KEEP_CKPT=1    # the headline ablation: needs the full capability suite
    # The one arm where a null at 120 steps would be actively misleading: the
    # anchor's whole claim is long-horizon stability, and this project has
    # three recorded arms that led at step 100-200 and then collapsed between
    # 300 and 430. Run it to the full Pair 1 horizon.
    TOTAL_STEPS="${TOTAL_STEPS_NOANCHOR:-600}"
    TEST_FREQ="${TEST_FREQ_NOANCHOR:-40}"
    ;;
  nosign)
    ARM_DESC="A1: anchor runs at full cost, its sign discarded (alpha=1)"
    export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=1.0
    ;;
  k10)  ARM_DESC="A2: K=10 (half cadence)";  export COMM_EFF_ANCHOR_DELAY_K=10 ;;
  k40)  ARM_DESC="A2: K=40 (2x cadence)";    export COMM_EFF_ANCHOR_DELAY_K=40 ;;
  k80)
    ARM_DESC="A2: K=80 (4x cadence)"
    export COMM_EFF_ANCHOR_DELAY_K=80
    KEEP_CKPT=1    # far end of the sweep: check MATH-only does not hide damage
    ;;
  k40gm)
    ARM_DESC="A2: K=40, forecast gain matched to K=20 (strength=c/K=0.5)"
    export COMM_EFF_ANCHOR_DELAY_K=40
    export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH=0.5
    ;;
  k80gm)
    ARM_DESC="A2: K=80, forecast gain matched to K=20 (strength=c/K=0.25)"
    export COMM_EFF_ANCHOR_DELAY_K=80
    export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH=0.25
    ;;
  smoke)
    ARM_DESC="25-step throughput and memory smoke, no validation"
    TOTAL_STEPS=25
    TEST_FREQ=-1
    export VAL_BEFORE_TRAIN=False
    ;;
  *) echo "FATAL: unknown ARM=$ARM" >&2; exit 1 ;;
esac

RUN_ID="${RUN_ID:-compass-$ARM}"
RUN_DIR="$WORK/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$WORK"

# 1. Checkout. BRANCH must be passed through: the sibling Pair 1 launchers
#    default to autonomous-harness-v1 and hard-reset, which would silently roll
#    this branch's arm table off the box.
if [[ -d verl/.git ]] \
   && (cd verl && git remote set-url origin "$REPO" \
       && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
       && git reset --hard FETCH_HEAD); then
  echo "=== reused checkout, reset to origin/$BRANCH ==="
else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl || {
    echo "FATAL: clone failed" >&2; exit 1; }
fi
cd verl
VERL_ROOT="$PWD"

# 2. Secrets (WANDB_API_KEY, HF_TOKEN, R2_*). The engine re-sources this too.
# shellcheck disable=SC1090
[[ -f "$HOME/.config/verl-research/secrets.env" ]] && . "$HOME/.config/verl-research/secrets.env"

# 3. MATH parquet.
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: MATH parquet unavailable in $DATA_DIR" >&2; exit 1; }

# 4. Pair 1 geometry. The base launcher hardcodes these four as bare exports,
#    so they can only be changed by patching a generated copy.
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_compass_$ARM.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"
chmod +x "$PATCHED"
grep -q '^export TRAIN_BATCH_SIZE=128$'     "$PATCHED" || { echo "FATAL: batch patch missed"      >&2; exit 1; }
grep -q '^export PPO_MINI_BATCH_SIZE=128$'  "$PATCHED" || { echo "FATAL: mini-batch patch missed" >&2; exit 1; }
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || { echo "FATAL: response patch missed"   >&2; exit 1; }
# 128/128 is exactly one optimizer tick per generation, so anchor cadence and
# delay, which are counted in ticks, equal global steps on this protocol.

# 5. Codec: the Pair 1 default, unchanged for every arm except `dense`.
export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-prf_mask}"
export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
export COMM_EFF_MASK_P=0.95
export COMM_EFF_MASK_RESCALE_MODE=constant
export COMM_EFF_MASK_EXACT_K=true
export COMM_EFF_MASK_RECOMPUTE=true
export COMM_EFF_MASK_REFERENCE=true
export COMM_EFF_MASK_PP_SIZE=8
export COMM_EFF_ANCHOR_OWNS_Q=false
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false

# 6. Run controls. One WandB project for the whole ablation family, one run
#    name per arm, so the arms are directly overlayable in the UI.
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export TOTAL_TRAINING_STEPS="$TOTAL_STEPS"
export TOTAL_EPOCHS=20                       # >=11 so the step count is the stop
export TEST_FREQ="$TEST_FREQ"
# Checkpoints only where the ten-benchmark suite needs them. Everything else
# runs SAVE_FREQ=-1, because four arms each saving a 1.5B FSDP checkpoint with
# optimizer state will fill a Vast disk long before the runs finish.
if [[ "$KEEP_CKPT" == "1" ]]; then
  export SAVE_FREQ="${SAVE_FREQ:-200}"
  export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-true}"
  export CKPT_R2_DELETE_LOCAL="${CKPT_R2_DELETE_LOCAL:-true}"
  export R2_EXPERIMENT="${R2_EXPERIMENT:-$PROJECT}"
  export R2_REGIME="${R2_REGIME:-$ARM}"
else
  export SAVE_FREQ="${SAVE_FREQ:--1}"
fi
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="$RUN_ID"
export PROJECT_NAME="$PROJECT"
export WANDB_RUN_GROUP="$PROJECT"
export LOG="$RUN_DIR/train.log"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

# 7. Pin this arm to one GPU. CUDA_VISIBLE_DEVICES alone is not enough: the
#    engine detects the GPU count with `nvidia-smi -L | wc -l` and has no env
#    override, so a shim on PATH must report a single device.
export CUDA_VISIBLE_DEVICES="$GPU"
SHIM_DIR="$RUN_DIR/bin"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/nvidia-smi" <<'SHIM'
#!/usr/bin/env bash
# Report exactly the devices this arm owns, so the engine sizes itself to one
# GPU while three sibling arms run on the other three.
REAL="$(PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "/bin$" | paste -sd: -)" command -v nvidia-smi || echo /usr/bin/nvidia-smi)"
if [[ "$#" -eq 1 && "$1" == "-L" ]]; then
  N="$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')"
  exec "$REAL" -L | head -n "$N"
fi
exec "$REAL" "$@"
SHIM
chmod +x "$SHIM_DIR/nvidia-smi"
export PATH="$SHIM_DIR:$PATH"
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == "1" ]] \
  || { echo "FATAL: nvidia-smi shim did not narrow to 1 GPU" >&2; exit 1; }

# 8. Thread budget. Four verl stacks on one box exhaust the container pids
#    cgroup long before they exhaust RAM. These caps plus ray_init.num_cpus are
#    what make the fan-out survive.
NCPU="$(nproc 2>/dev/null || echo 32)"
RAY_CPUS="${RAY_CPUS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

echo "=============================================================="
echo " COMPASS RLVR ablation  arm=$ARM  gpu=$GPU  run=$RUN_ID"
echo " $ARM_DESC"
echo " steps=$TOTAL_STEPS test_freq=$TEST_FREQ gpu_mem=$GPU_MEM ray_cpus=$RAY_CPUS"
echo " project=$PROJECT branch=$BRANCH"
echo "=============================================================="
env | grep -E '^COMM_EFF_(ENABLED|COMPRESSION|MASK_P|MASK_ENABLED|ANCHOR_|SPECTRAL_)' | sort

exec bash "$PATCHED" "ray_kwargs.ray_init.num_cpus=$RAY_CPUS"
