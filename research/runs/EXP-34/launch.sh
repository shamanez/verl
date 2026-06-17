#!/usr/bin/env bash
# EXP-34 — signed_ema(alpha=0.5) beta_anc sweep {0.25, 0.50, 0.75}
# ----------------------------------------------------------------------------
# Three config-only sibling cells, back-to-back on ONE operator-provided box
# (team-account instance 41292294, 4xH200). code_change=false: B2 substrate is
# pinned by the canonical wrapper (vast_comm_eff_b2_sota_*), and only the merger
# knobs are overridden via Hydra last-wins "$@" passthrough:
#     correction_mode=signed_ema  signed_ema_alpha=0.5  beta_anc=<sweep>
# val_before_train=False (val @ steps 25 & 50 only); 55 steps (50 + flush).
# W&B project = verl_compression_research_beta_sweep_signed_ema.
#
# BREAK-GLASS (attempt 2): DISABLE_CUSTOM_ALL_REDUCE=true on every cell. Attempt 1
# crashed all 3 cells at vLLM KV-cache init with the known
# `custom_all_reduce.cuh:455 'invalid argument'` CUDA-IPC failure (0 steps). The
# wrapper translates this env var into the Hydra override
# `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true`
# (NCCL all-reduce instead — greedy-val-neutral, a controlled var). See memory
# `vast-vllm-custom-allreduce-ipc-failure` + plan/34.md §Notes-for-runner.
#
# `set -u` but NOT `-e`: a crashed/early-stopped cell must NOT abort the
# remaining cells (each cell is an independent draw; a failure IS data).
# ----------------------------------------------------------------------------
set -uo pipefail

RUN=/workspace/runs/EXP-34
mkdir -p "$RUN/metrics"
cd /workspace/verl

run_cell() {
  local beta="$1" name="$2"
  echo "===== [EXP-34] CELL ${name} (signed_ema alpha=0.5 beta_anc=${beta}) START $(date -u +%FT%TZ) ====="
  mkdir -p "$RUN/${name}"
  local rc=0
  PROJECT_NAME=verl_compression_research_beta_sweep_signed_ema \
  TOTAL_TRAINING_STEPS=55 TEST_FREQ=25 EXPERIMENT_NAME="${name}" \
  DISABLE_CUSTOM_ALL_REDUCE=true \
  bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
    trainer.project_name=verl_compression_research_beta_sweep_signed_ema \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema \
    actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.5 \
    actor_rollout_ref.actor.comm_eff.spectral.beta_anc="${beta}" || rc=$?
  if (( rc != 0 )); then
    echo "===== [EXP-34] CELL ${name} exited non-zero (rc=${rc}) — recording + continuing ====="
  fi
  # Mirror the authoritative per-cell training log (val numbers + resolved
  # main_ppo command live here) into the harness run dir. Copy ONLY the log +
  # done flag — never checkpoints/wandb blobs (keep transfers small).
  cp -f "/workspace/verl/runs/${name}/train.log" "$RUN/${name}/train.log" 2>/dev/null || true
  [[ -f "/workspace/verl/runs/${name}/done.flag" ]] && cp -f "/workspace/verl/runs/${name}/done.flag" "$RUN/${name}/done.flag" 2>/dev/null || true
  [[ -f "/workspace/verl/runs/${name}/EARLY_STOP_SIGNAL" ]] && cp -f "/workspace/verl/runs/${name}/EARLY_STOP_SIGNAL" "$RUN/${name}/EARLY_STOP_SIGNAL" 2>/dev/null || true
  touch "$RUN/done_${name}.flag"
  echo "===== [EXP-34] CELL ${name} DONE $(date -u +%FT%TZ) (rc=${rc}) ====="
}

run_cell 0.25 signed_ema_b0p25
run_cell 0.50 signed_ema_b0p50
run_cell 0.75 signed_ema_b0p75

touch "$RUN/done.flag"
echo "===== [EXP-34] ALL CELLS DONE $(date -u +%FT%TZ) ====="
