#!/usr/bin/env bash
# EXP-14 master launcher — runs inside the Vast.ai container.
# ONE box, Test 1 -> Test 2 -> Test 3 chained back-to-back (strictly sequential).
# Test 4 is CONDITIONAL and is NOT run here (analyst triggers it later).
#
# The template onstart cloned shamanez/verl @ vast-ai-workload into /workspace/verl
# and pip-installed it. For code_change=true we REPLACE that tree with the
# exp/14-clean-cadence branch shipped in exp.bundle (the clean_cadence patch).
#
# Each cell:
#   - starts from the canonical comm-eff launcher (vast_comm_eff_*.sh), which
#     already sets use_orig_params=true + the no-KL objective + restored knobs,
#   - overrides ONLY the knobs that cell varies (env + Hydra "$@" passthrough),
#   - writes per-step metrics to /workspace/runs/EXP-14/metrics/<test>_<cell>.jsonl
#     (verl FileLogger via VERL_FILE_LOGGER_PATH) and stdout to logs/<test>_<cell>.log,
#   - drops a per-cell done marker. A final done.flag is written when the chain ends.
set -uo pipefail   # NOT -e: a single cell failure must not abort the whole chain
                   # (we capture each cell's rc and keep going so the analyst gets
                   #  partial metrics; FSDP-no-errors is judged per cell from logs).

RUN_DIR=/workspace/runs/EXP-14
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/logs" "$RUN_DIR/hotfix-patches"
CHAIN_LOG="$RUN_DIR/train.log"   # liveness file the runner tails; mirrors chain progress
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=================================================================="
echo "[EXP-14] master chain start $(date -Iseconds)"
echo "[EXP-14] host: $(hostname)  gpus: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=================================================================="

# Git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---------------------------------------------------------------------------
# Apply the experimental bundle (code_change=true): replace /workspace/verl with
# the exp/14-clean-cadence branch so the clean_cadence patch is in effect.
# ---------------------------------------------------------------------------
if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  echo "[EXP-14] applying exp.bundle -> exp/14-clean-cadence on /workspace/verl"
  cd /workspace
  if [[ -d verl && ! -d verl.upstream-vast-ai-workload ]]; then
    mv verl verl.upstream-vast-ai-workload      # preserve template-installed tree
  fi
  rm -rf /workspace/verl
  git clone -b exp/14-clean-cadence "$RUN_DIR/exp.bundle" /workspace/verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  echo "[EXP-14] verl now at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
  echo "[EXP-14] uv pip install --no-deps -e . (clean_cadence patch)"
  uv pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1 \
    || pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1 \
    || { echo "[EXP-14] FATAL: editable install of patched verl failed; see pip.log"; tail -20 "$RUN_DIR/pip.log"; exit 1; }
  # Hard-verify the clean_cadence knob is actually importable post-install.
  python3 -c "from verl.workers.config.comm_eff import CommEffConfig; assert hasattr(CommEffConfig(), 'clean_cadence'); print('[EXP-14] clean_cadence knob present:', CommEffConfig().clean_cadence)" \
    || { echo '[EXP-14] FATAL: clean_cadence knob missing after install'; exit 1; }
else
  echo "[EXP-14] FATAL: exp.bundle missing in $RUN_DIR — cannot run code_change cell"; exit 1
fi

cd /workspace/verl

# ---------------------------------------------------------------------------
# Dataset preprocess ONCE (cells reuse the cache). The comm-eff launcher would
# do this on first cell anyway, but doing it up front keeps the per-cell handoff
# tight and lets the first cell's liveness window be training, not preprocessing.
# ---------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "[EXP-14] preprocess GSM8K -> $DATA_DIR"
  mkdir -p "$DATA_DIR"
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR" || true
fi

COMM_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# ---------------------------------------------------------------------------
# run_cell <cell_id> <total_steps> <hydra override...>
#   Routes per-cell metrics jsonl + stdout log; isolates checkpoint/run dirs;
#   forces the restored baseline knobs (wedge=36864, mini=64, util=0.4) via the
#   launcher's env knobs; passes the with/without-comm-eff + clean_cadence deltas
#   as trailing Hydra args (last-wins over the launcher's own comm-eff defaults).
# ---------------------------------------------------------------------------
run_cell() {
  local cell="$1"; shift
  local steps="$1"; shift
  local cell_log="$RUN_DIR/logs/${cell}.log"
  local cell_jsonl="$RUN_DIR/metrics/${cell}.jsonl"

  echo ""
  echo "------------------------------------------------------------------"
  echo "[EXP-14] CELL ${cell}  steps=${steps}  start $(date -Iseconds)"
  echo "[EXP-14]   metrics -> ${cell_jsonl}"
  echo "[EXP-14]   log     -> ${cell_log}"
  echo "------------------------------------------------------------------"

  # Per-cell metrics jsonl (verl FileLogger reads this env).
  export VERL_FILE_LOGGER_PATH="$cell_jsonl"
  # Per-cell experiment name (WandB run separation) + isolated checkpoint dir.
  export EXPERIMENT_NAME="exp14-${cell}"
  export PROJECT_NAME="verl_compression_research"

  # Restored baseline batch knobs on EVERY cell (lean cells drop the clone, fit 4xH200).
  export PPO_MINI_BATCH_SIZE=64
  export PPO_MAX_TOKEN_LEN_PER_GPU=36864
  export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
  export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
  export ROLLOUT_GPU_MEM_UTIL=0.4
  # Paper-scale fixed control vars (the launcher already defaults these; explicit for audit).
  export TRAIN_BATCH_SIZE=128
  export ROLLOUT_N=8
  export MAX_PROMPT_LENGTH=1024
  export MAX_RESPONSE_LENGTH=16384
  export TOTAL_EPOCHS=2
  export VAL_BEFORE_TRAIN=True
  export TEST_FREQ=25
  export SAVE_FREQ=1000000          # effectively no mid-run checkpoint save (diagnosis, not a kept ckpt)
  export TOTAL_TRAINING_STEPS="$steps"
  # Route this cell's own train.log inside the launcher (it also tee's to LOG).
  export LOG="$cell_log"

  # The "file" logger is added on top of console+wandb via Hydra (last-wins).
  # Trailing Hydra args ($@) are the per-cell delta; they override the launcher's
  # comm-eff defaults (which assume the FULL method). set +e so a cell crash is
  # captured, not fatal to the chain.
  set +e
  bash "$COMM_LAUNCHER" \
    trainer.logger='["console","wandb","file"]' \
    trainer.total_training_steps="$steps" \
    trainer.default_local_dir="$RUN_DIR/ckpt/${cell}" \
    "$@"
  local rc=$?
  set -e 2>/dev/null || true
  set +e

  echo "[EXP-14] CELL ${cell} exit rc=${rc} at $(date -Iseconds)"
  echo "$(date -Iseconds) ${cell} rc=${rc} steps=${steps}" > "$RUN_DIR/${cell}.done"
  return 0
}

# Common Hydra deltas shared by all METHOD cells (no-KL no-entropy + use_orig_params).
# use_orig_params=true is already set by the comm-eff launcher; we re-assert it on
# the Test 1 (comm-eff OFF) cells via the trailing args for parity on every cell.
NOKL=(actor_rollout_ref.actor.use_kl_loss=False
      algorithm.use_kl_in_reward=False
      actor_rollout_ref.actor.entropy_coeff=0)
USE_ORIG=(actor_rollout_ref.actor.fsdp_config.use_orig_params=true)

# =========================== TEST 1 (GATE) ===========================
# Cell A: comm-eff OFF, WITH KL (dense reference reproduction). The ONLY with-KL cell.
run_cell test1_cellA 10 \
  actor_rollout_ref.actor.comm_eff.enabled=false \
  actor_rollout_ref.actor.comm_eff.mask.enabled=false \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  "${USE_ORIG[@]}"

# Cell B: comm-eff OFF, KL OFF (scaffold-noop). step-1 grad_norm must match Cell A and be <= 1.0.
run_cell test1_cellB 10 \
  actor_rollout_ref.actor.comm_eff.enabled=false \
  actor_rollout_ref.actor.comm_eff.mask.enabled=false \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  "${NOKL[@]}" "${USE_ORIG[@]}"

# =========================== TEST 2 (PEEL, diagnosis) ===========================
# Pure masked GRPO: mask on, anchor+spectral OFF (not even allocated), clean_cadence=0.
run_cell test2_cellA 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

run_cell test2_cellB 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=false \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=0 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

# =========================== TEST 3 (FIX, mandatory headline) ===========================
# Cell A REUSES Test 2 Cell A (identical config). Copy the jsonl + log, do NOT re-run.
echo ""
echo "[EXP-14] TEST 3 Cell A: reusing Test 2 Cell A (identical config) — copying artifacts"
if [[ -f "$RUN_DIR/metrics/test2_cellA.jsonl" ]]; then
  cp -f "$RUN_DIR/metrics/test2_cellA.jsonl" "$RUN_DIR/metrics/test3_cellA.jsonl"
fi
[[ -f "$RUN_DIR/logs/test2_cellA.log" ]] && cp -f "$RUN_DIR/logs/test2_cellA.log" "$RUN_DIR/logs/test3_cellA.log"
echo "$(date -Iseconds) test3_cellA reuse_of=test2_cellA" > "$RUN_DIR/test3_cellA.done"

# Cell B: mask-only + periodic clean step (clean_cadence=10). 100 steps. THE headline.
run_cell test3_cellB 10 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=false \
  actor_rollout_ref.actor.comm_eff.clean_cadence=10 \
  "${NOKL[@]}" "${USE_ORIG[@]}"

echo ""
echo "=================================================================="
echo "[EXP-14] master chain COMPLETE $(date -Iseconds)"
echo "[EXP-14] cells ran: test1_cellA test1_cellB test2_cellA test2_cellB test3_cellB (test3_cellA reused test2_cellA)"
echo "[EXP-14] Test 4 (conditional) NOT run — analyst triggers if needed."
echo "=================================================================="
echo "$(date -Iseconds) chain done" > "$RUN_DIR/done.flag"
