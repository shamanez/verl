#!/usr/bin/env bash
# EXP-31 Cell F — DENSE control rerun on THIS box/config (comm-eff OFF = byte-identical dense).
# Re-pins the TRUE dense bar HERE: the 0.7839 reference was a DIFFERENT box without
# disable_custom_all_reduce, and B2 shifted 0.7528→0.7400 on this config, so dense-here
# may also differ. Carries disable_custom_all_reduce (the controlled-variable fix) so the
# dense band is measured under the SAME setup as every comm-eff arm. seed via $DSEED.
# comm_eff DISABLED ⇒ the exp/31 sub-basis code is never invoked ⇒ pure dense regardless
# of the checked-out branch.
set -euo pipefail
cd /workspace/verl
DSEED="${DSEED:-0}"
echo "=== dense rerun seed=$DSEED on $(git rev-parse --short HEAD) (comm_eff OFF) ==="
python3 -c "import verl" 2>/dev/null \
  || uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
  || pip install --no-deps -e . >> /workspace/pip.log 2>&1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COMM_EFF_ENABLED=false                       # master switch OFF ⇒ byte-identical dense
export PPO_MAX_TOKEN_LEN_PER_GPU=18432
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="exp31_F_dense_seed${DSEED}"
export LOG="/workspace/runs/EXP-31/train_F_dense_s${DSEED}.log"

mkdir -p /workspace/runs/EXP-31/metrics
ln -sf "$LOG" /workspace/train.log

RC=0
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  data.seed="$DSEED" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true || RC=$?
echo "$(date -Iseconds) F_dense_seed${DSEED} done rc=$RC" > "/workspace/runs/EXP-31/done_F_dense_s${DSEED}.flag"
exit "$RC"
