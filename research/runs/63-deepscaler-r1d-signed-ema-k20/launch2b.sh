#!/usr/bin/env bash
# launch.sh — issue #63: comm-eff signed_ema vs dense on DeepScaleR RLVR,
# R1-Distill-Qwen-1.5B, AIME-2024 boxed val. 4 cells, SEQUENTIAL, one box.
# dense-control runs FIRST (reward-health gate: step-0 val floor per plan inv 3).
#
# Run under tmux with stdout -> /workspace/runs/<id>/launch.log (orchestration +
# bootstrap + data-prep progress). Per-cell TRAINING output -> train.log (the
# monitor's remote_log, live global_step of the ACTIVE cell); each finished cell
# is preserved to train_<cell>.log for the analyst.
#
# NOT set -e: a single cell's non-zero exit is recorded (fail_<cell>.flag) and the
# sweep CONTINUES — semantic stops (dense floor failure) are /monitor's job.
set -uo pipefail

RUN_ID=63-deepscaler-r1d-signed-ema-k20
RUN_DIR=/workspace/runs/$RUN_ID
BASE_BRANCH=autonomous-harness-v1
VERL_DIR=/workspace/verl
DEEPSCALER_DIR=/workspace/data/deepscaler
AIME_DIR=/workspace/data/aime2024_boxed
SECRETS="$HOME/.config/verl-research/secrets.env"
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"

echo "=== [$(date -Iseconds)] launch.sh START $RUN_ID ==="

# ---------------------------------------------------------------------------
# 0. Secrets (data-prep needs HF; each cell launcher re-sources + validates).
# ---------------------------------------------------------------------------
if [[ -r "$SECRETS" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS"
  export HF_TOKEN="${HF_TOKEN:-}" \
         HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
         HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
         WANDB_API_KEY="${WANDB_API_KEY:-}"
else
  echo "FATAL: $SECRETS missing on box — push a stripped HF_TOKEN+WANDB_API_KEY copy before launch." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. verl bootstrap. The box may NOT be from the locked template (operator
#    hand-off): tolerate an absent checkout. The base IMAGE is assumed to carry
#    torch/vLLM/ray (verlai/verl:vllm020.dev1); we only ensure the verl SOURCE
#    on source_tree.base_branch + an editable install. code_change:false, so no
#    exp bundle — the AIME converter travels in the run payload instead.
# ---------------------------------------------------------------------------
cd /workspace
if [[ ! -d "$VERL_DIR/.git" ]]; then
  echo "=== verl checkout absent — cloning shamanez/verl ==="
  [[ -e "$VERL_DIR" ]] && mv "$VERL_DIR" "${VERL_DIR}.stale.$(date +%s)"
  git clone https://github.com/shamanez/verl.git "$VERL_DIR"
fi
cd "$VERL_DIR"
git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
# The onstart clone can carry a RESTRICTED fetch refspec (only its pinned branch),
# so `git fetch origin <base>` populates FETCH_HEAD but NOT the origin/<base>
# tracking ref. Check out FETCH_HEAD directly (refspec-independent) and VERIFY the
# branch — a failed checkout must NEVER silently proceed on the wrong branch.
git fetch origin "$BASE_BRANCH"
git checkout -B "$BASE_BRANCH" FETCH_HEAD
CUR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "=== verl @ $(git rev-parse --short HEAD) (branch=$CUR) ==="
[[ "$CUR" == "$BASE_BRANCH" ]] || { echo "FATAL: bootstrap left HEAD on '$CUR', expected '$BASE_BRANCH' — aborting before data prep." >&2; exit 1; }
if ! python3 -c "import verl" 2>/dev/null; then
  echo "=== verl not importable — editable install (--no-deps; image provides torch/vLLM) ==="
  (uv pip install --no-deps -e . || pip install --no-deps -e .) > /workspace/pip.log 2>&1 \
    || { echo "FATAL: verl editable install failed — see /workspace/pip.log" >&2; tail -25 /workspace/pip.log >&2; exit 1; }
fi
python3 -c "import verl, torch; print('=== verl OK; torch', torch.__version__, 'cuda', torch.cuda.is_available(), '===')" \
  || { echo "FATAL: verl/torch import failed after bootstrap — base image likely lacks the verl runtime." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Data prep (on-box; large downloads). deepscaler via the base-branch script;
#    AIME-2024 boxed val via the payload converter (committed on exp/63).
#    data_source routes: deepscaler -> DigitalLearningGmbH/MATH-lighteval,
#    AIME -> HuggingFaceH4/MATH-500 (distinct math_reward keys).
# ---------------------------------------------------------------------------
if [[ ! -f "$DEEPSCALER_DIR/train.parquet" || ! -f "$DEEPSCALER_DIR/test.parquet" ]]; then
  echo "=== [$(date -Iseconds)] prepare deepscaler -> $DEEPSCALER_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset deepscaler \
    --local_save_dir "$DEEPSCALER_DIR" --train-cap 20000 --val-size 500 --seed 42 \
    || { echo "FATAL: deepscaler prep failed" >&2; exit 1; }
fi
if [[ ! -f "$AIME_DIR/val.parquet" ]]; then
  echo "=== [$(date -Iseconds)] prepare AIME-2024 boxed val -> $AIME_DIR ==="
  python3 "$RUN_DIR/prepare_aime_boxed.py" --local_save_dir "$AIME_DIR" \
    || { echo "FATAL: AIME prep failed" >&2; exit 1; }
fi
_rows() { python3 -c "import pyarrow.parquet as p;print(p.read_table('$1').num_rows)" 2>/dev/null || echo "?"; }
echo "=== data ready: deepscaler train=$(_rows "$DEEPSCALER_DIR/train.parquet") test=$(_rows "$DEEPSCALER_DIR/test.parquet") | aime val=$(_rows "$AIME_DIR/val.parquet") ==="

# ---------------------------------------------------------------------------
# 3. Shared config — exported once, read by every cell's launcher via ${VAR:-}.
#    These are the ONLY deltas from the generic launcher defaults (which already
#    ARE the #63 substrate: powersgd r77, anchor owns Q, replay_paired_batch,
#    q_basis=act, clean_cadence=0, spectral signed_ema alpha=0.25 beta_anc=0.50,
#    snapshot/ema on cpu, disable_custom_all_reduce). anchor 20/20 + spectral
#    cadence 20 = the high-latency k-collapse regime.
# ---------------------------------------------------------------------------
# Shared surface = RLVR-Linearity reproduction (run_distill-qwen-1-5b_deepscaler.sh)
# for DeepSeek-R1-Distill-Qwen-1.5B (operator directive 2026-07-08): prompt 2048,
# response 16384, n=16 rollouts, gpu_mem_util 0.85, ppo_max_token 30000. The
# reference ran on 8 GPUs + CPU offload; on 1×H200 the offload (REF_HYDRA below)
# + the box's 1.5 TB RAM are what make it fit. anchor 20/20 + spectral cadence 20
# = the high-latency k-collapse regime (comm-eff arms only; dense = reference).
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR="$DEEPSCALER_DIR"
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export ROLLOUT_N=16
export USE_DYNAMIC_BSZ=True
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.85
export PPO_MAX_TOKEN_LEN_PER_GPU=20000   # FALLBACK 2026-07-09: offload-ON alone is marginal at the anchor refresh; lower per-GPU micro-batch token budget (linear on the ~75-80GB un-checkpointed anchor-clone activation surcharge). Pure micro-batching, NO numeric change (train_batch=128, mini_batch=64 unchanged).
export TOTAL_TRAINING_STEPS=102        # operator re-plan 2026-07-09: cap all arms at 102 steps (was 200)
export TEST_FREQ=25
export VAL_BEFORE_TRAIN=True
export COMM_EFF_ANCHOR_CADENCE=20
export COMM_EFF_ANCHOR_DELAY_K=20
export COMM_EFF_SPECTRAL_CADENCE=20
# WandB project = the RUN ID (operator directive 2026-07-08): every issue gets its
# OWN project named <issue>-<slug> — never the global verl_compression_research.
export PROJECT_NAME="$RUN_ID"
export WANDB_RUN_GROUP="$RUN_ID"
# Checkpoint policy (operator re-plan 2026-07-09): arms run to 102 steps, save at
# step 100 (SAVE_FREQ=100), mirror that checkpoint to R2. verl saves at global_step
# % save_freq == 0, so global_step_100 is written; the last 2 steps (101,102) train
# without a further save. R2 sink is dynamic — it mirrors whatever global_step_<N>/
# is saved. R2 layout: s3://$R2_BUCKET/verl-research/$R2_EXPERIMENT/$R2_REGIME/checkpoints/global_step_100/…
# R2_REGIME is set per-cell in run_cell so each arm lands in its own folder.
export SAVE_FREQ=100                    # operator re-plan 2026-07-09: checkpoint at step 100 (was 200); R2 sink mirrors global_step_100/
export CKPT_R2_ENABLED=true
export CKPT_R2_DELETE_LOCAL=true          # R2 is the home; free box disk after each verified upload
export R2_EXPERIMENT="$RUN_ID"

# Validation = AIME-2024 ONLY, avg@8 (matches the RLVR-Linearity reference
# run_distill-qwen-1-5b_deepscaler.sh: data.val_files=[aime24.parquet],
# val_kwargs.n=8). The deepscaler-test val file was DROPPED (operator 2026-07-08):
# at n=8 × 16384 resp × 500 rows every 25 steps it would cost more than training.
# The deepscaler split still yields its TRAIN reward curve (the RL signal); only
# its VAL curve is dropped. Headline capability metric = AIME avg@8.
VAL_HYDRA="data.val_files=[$AIME_DIR/val.parquet]"
DIAG_HYDRA="actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false"

# RLVR-Linearity reproduction fidelity — applied to ALL cells so dense + comm-eff
# arms share ONE RL-algorithm surface (parity comparison stays valid).
# CPU OFFLOAD REMOVED (operator 2026-07-08, 4×H200): on 4 GPUs the launcher's
# hardcoded param_offload=False/optimizer_offload=False lets FSDP shard model+
# optimizer across the 4 ranks (no CPU↔GPU transfer stalls = FAST steps). Offload
# was only needed to fit ONE H200; keep it OFF on multi-GPU. NGPUS_PER_NODE is
# auto-detected by the launcher (=4 here). NOTE: asymmetric clip 0.2/0.28 +
# seq-mean-token-mean come from the reference and DEVIATE from the project's
# vanilla-GRPO control (CLAUDE.md §1) — intentional, to reproduce the paper surface.
REF_HYDRA=(
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.rollout.max_num_batched_tokens=35840
  actor_rollout_ref.rollout.val_kwargs.n=8
  data.filter_overlong_prompts=True
  actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]
  # OFFLOAD ON (operator directive 2026-07-09 after signed-ema-b50 OOM'd at the first
  # anchor refresh, tick 20 / step 10). The launcher hardcodes actor param/optimizer
  # offload=False; these trailing overrides win (Hydra last-wins, "$@" is appended last).
  # dense-control fit at offload OFF (no compression state); the comm-eff arms add
  # PowerSGD r77 Q/P + anchor (replay_paired_batch runs an EXTRA fwd+bwd) + spectral EMA,
  # which overflowed the 143GB ceiling. Offload -> box 1.5TB RAM. Pure storage location:
  # NO numeric change, comm-eff-vs-dense comparison stays valid. dense's result unaffected.
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

# ---------------------------------------------------------------------------
# 4. Cell runner. LOG=$RUN_DIR/train.log for every cell (single live log the
#    monitor tails for the active cell's global_step); preserved to
#    train_<cell>.log after. Pre-create the launcher's hardcoded done-flag dir
#    (/workspace/verl/runs/<exp>/) so its final `touch` can't set-e-false-fail a
#    successful cell. Per-cell env deltas before `--`, per-cell Hydra after.
# ---------------------------------------------------------------------------
run_cell() {
  local cell="$1"; shift
  local envs=() hydra=() rc exp
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  hydra=("$@")
  exp="63-$cell"
  mkdir -p "$VERL_DIR/runs/$exp"
  echo "=== [$(date -Iseconds)] CELL $cell START (wandb run $exp) ==="
  env EXPERIMENT_NAME="$exp" R2_REGIME="$cell" LOG="$RUN_DIR/train.log" ${envs[@]+"${envs[@]}"} \
    bash "$LAUNCHER" "$VAL_HYDRA" "$DIAG_HYDRA" "${REF_HYDRA[@]}" ${hydra[@]+"${hydra[@]}"}
  rc=$?
  cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_${cell}.log" 2>/dev/null || true
  if [[ $rc -eq 0 ]]; then
    echo "$(date -Iseconds)" > "$RUN_DIR/done_${cell}.flag"
    echo "=== [$(date -Iseconds)] CELL $cell DONE (rc=0) ==="
  else
    echo "$(date -Iseconds) rc=$rc" > "$RUN_DIR/fail_${cell}.flag"
    echo "=== [$(date -Iseconds)] CELL $cell FAILED (rc=$rc) — continuing; /monitor classifies ==="
  fi
}

# Order top-to-bottom = plan cell table. dense-control FIRST (reward-health gate).
# dense-control ALREADY RUN to step 102 under launch.sh (operator re-plan 2026-07-09):
# curve+val kept on WandB + train_dense-control.log; no checkpoint saved (operator choice).
# run_cell dense-control     COMM_EFF_ENABLED=false
run_cell signed-ema-b50    COMM_EFF_SPECTRAL_BETA_ANC=0.50
run_cell signed-ema-b50-la COMM_EFF_SPECTRAL_BETA_ANC=0.50 -- \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=fixed_linear \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength=1.0
run_cell signed-ema-b00    COMM_EFF_SPECTRAL_BETA_ANC=0.00

echo "$(date -Iseconds) all cells attempted" > "$RUN_DIR/done.flag"
echo "=== [$(date -Iseconds)] launch.sh END $RUN_ID ==="
