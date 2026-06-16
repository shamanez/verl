#!/usr/bin/env bash
# EXP-32 cell 1: signed_ema alpha=0.5 on the CORRECTED (valid-M, #29) anchor circuit.
# Plan: research/.claude/plans/32.md. kind=experiment, code_change=FALSE (config-only).
#
# THIS IS B2 + EXACTLY ONE KNOB FLIPPED:
#   correction_mode: delayed_ef (B2 SOTA) -> signed_ema ;  signed_ema_alpha pinned 0.5.
# Both are passed via the Hydra "$@" passthrough, NOT via env vars. WHY (load-bearing):
#   vast_comm_eff_b2_sota_*.sh:52 HARD-exports COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef,
#   which CLOBBERS any pre-set env var. The b2_sota wrapper forwards "$@" to the baseline
#   launcher (line 85), which forwards it again at line 601 — AFTER it threads the env value
#   to spectral.correction_mode at line 559. So the Hydra passthrough last-wins; the env-var
#   route would SILENTLY run delayed_ef (= B2), wasting ~6-8 GPU-hr. signed_ema_alpha also
#   MUST be explicit: there is no SIGNED_EMA_ALPHA env var, and actor.yaml defaults it to 0.0
#   (the catastrophic sign-SGD / entropy-collapse mode); the experiment needs 0.5.
# Everything else (replay_paired_batch=true, snapshot_device=cpu, PowerSGD r=77, anchor
#   on/owns_q, cadence=delay_K=5, clean=0, OOM guards, locked GSM8K surface, diagnostics OFF,
#   custom_all_reduce DEFAULT) is baked into the b2_sota wrapper — do NOT re-type it.
set -euo pipefail
cd /workspace/verl

# SHA-assert the substrate is the EXP-31-closeout B2 base (f18291f or a descendant carrying
# the b2_sota launcher + signed_ema merger). It already is; this is a guard, not a fetch.
echo "=== EXP-32 code: $(git rev-parse --abbrev-ref HEAD) $(git log --oneline -1) ===" >&2
test -f examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
  || { echo "FATAL: b2_sota launcher missing — stale box" >&2; exit 3; }
grep -q '"signed_ema"' verl/workers/comm_eff/spectral_filter.py \
  || { echo "FATAL: signed_ema not an accepted correction_mode — stale code" >&2; exit 3; }

# verl importable (template onstart already pip-installed it; tolerate a re-install).
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

mkdir -p /workspace/runs/EXP-32/metrics
LOG=/workspace/runs/EXP-32/train.log
# Liveness + sync-metrics contract: /workspace/train.log IS the live cell log.
ln -sf "$LOG" /workspace/train.log

# Optional per-box vLLM CUDA-IPC workaround. DEFAULT off (B2 ran custom_all_reduce; the plan
# wants generation byte-identical to B2). The monitor re-launches with this =true ONLY if the
# EngineCore custom_all_reduce crash fires at KV-cache init. NCCL fallback is greedy-val-neutral.
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-false}"

RC=0
TOTAL_TRAINING_STEPS=50 TEST_FREQ=25 EXPERIMENT_NAME=exp32_signed_ema_a0p5_validM \
  bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
    actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema \
    actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.5 \
    > "$LOG" 2>&1 || RC=$?
echo "$(date -Iseconds) exp32_signed_ema_a0p5_validM done rc=$RC" > /workspace/runs/EXP-32/done.flag
exit "$RC"
