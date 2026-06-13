#!/usr/bin/env bash
# EXP-31 Cell D — THE HEADLINE: additive stale-anchor rank-2 sub-basis merger.
# Plan: research/.claude/plans/31.md §"Experiment sequence" Cell D; design:
# research/runs/EXP-31/cellD_design.md.
#
# Mechanism (weight-gradient realization, forward Q UNTOUCHED ⇒ Step-C avoidance
# automatic): G_corr = G_comp + λ·(δ_B2 + rank_{r_sb}(δ_B2)),  δ_B2 = M_rep − G_comp_ring,
# r_sb=2 tail. delta_subbasis_rank=0 == B2 bitwise (the off-path-parity gate).
#
# Runs on the exp/31-subbasis-merger code (applied from the shipped bundle so no
# GitHub auth is needed on the box). Everything else is Cell A's exact B2 config.
set -euo pipefail
cd /workspace/verl

# --- Apply the Cell D code from the bundle (rsync'd to the run dir by the runner) ---
#     The box's /workspace/verl is pip-installed EDITABLE (-e), so checking out the
#     branch makes the new code live on the next process start (no reinstall needed).
BUNDLE=/workspace/runs/EXP-31/exp.bundle
if [[ -f "$BUNDLE" ]]; then
  git fetch "$BUNDLE" 'exp/31-subbasis-merger:exp/31-subbasis-merger' 2>/dev/null || true
  git checkout -f exp/31-subbasis-merger
  git reset --hard exp/31-subbasis-merger
else
  echo "FATAL: $BUNDLE missing — cannot apply Cell D code" >&2; exit 3
fi
echo "=== Cell D code: $(git log --oneline -1) (branch $(git rev-parse --abbrev-ref HEAD)) ==="

# Hydra struct-mode gate: actor.yaml must DECLARE the EXP-31 sub-basis fields or the
# CLI overrides (delta_subbasis_rank=1 ...) are rejected ("not in struct"). The branch
# added them to the dataclass (CommEffSpectralConfig) but not to the YAML struct.
# Replicate the delayed_ef_lambda precedent (actor.yaml:488). Idempotent; survives the
# reset --hard above because it runs AFTER the checkout, on the fresh tree, every launch.
python3 - <<'PYEOF'
import re
f = "/workspace/verl/verl/trainer/config/actor/actor.yaml"
s = open(f).read()
if "delta_subbasis_rank" not in s:
    s2 = re.sub(
        r"(?m)^([ \t]*)delayed_ef_lambda:.*$",
        lambda m: m.group(0) + "\n" + m.group(1) + "delta_subbasis_rank: 0\n"
                  + m.group(1) + "delta_subbasis_family: tail\n" + m.group(1) + "r_delta: 0",
        s, count=1)
    assert s2 != s, "FATAL: delayed_ef_lambda anchor not found in actor.yaml — cannot patch struct"
    open(f, "w").write(s2)
    print("actor.yaml: EXP-31 sub-basis fields inserted (struct-mode gate)")
else:
    print("actor.yaml: sub-basis fields already present")
PYEOF

python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1
# Fail fast if the sub-basis knob is not in the resolved config schema (wrong code).
# (class-level hasattr — no instantiation, robust to required fields; class is CommEffSpectralConfig)
python3 -c "from verl.workers.config.comm_eff import CommEffSpectralConfig as _C; assert hasattr(_C, 'delta_subbasis_rank'), 'delta_subbasis_rank missing — wrong code checked out'; print('Cell D knobs present')"

# --- B2 substrate knobs (IDENTICAL to Cell A / resolved_params_B2.txt) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # standing OOM guard
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef
export COMM_EFF_SPECTRAL_BETA_ANC=0.0
export COMM_EFF_SPECTRAL_EF_DECAY=0.0
export COMM_EFF_SPECTRAL_EF_CLIP=0.0
export COMM_EFF_SPECTRAL_BLEND_ETA=0.3                    # inert at delayed_ef; pinned == B2
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu                   # standing OOM guard
export PPO_MAX_TOKEN_LEN_PER_GPU=18432                    # standing OOM guard
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25                                       # val at 0/25/50
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME=exp31_D_subbasis_r1_tail
export LOG=/workspace/runs/EXP-31/train_D_r1.log

mkdir -p /workspace/runs/EXP-31/metrics
ln -sf "$LOG" /workspace/train.log

# Cell D's ONLY delta vs Cell A: the sub-basis knobs (explicit trailing Hydra args,
# auditable in set -x) + the controlled-variable disable_custom_all_reduce env fix.
RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch=true \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_rank=1 \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_family=tail \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true || RC=$?
echo "$(date -Iseconds) D done rc=$RC" > /workspace/runs/EXP-31/done_D_r1.flag
exit "$RC"
