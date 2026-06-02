#!/usr/bin/env bash
# C1 CLEAN relaunch — seed_anchor_cache=FALSE. The first C1 launch used the
# launcher default seed_anchor_cache=true, which seeds M_anchor with a random
# deterministic basis. For an INJECTION correction that means injecting a
# random-direction force (scaled to ||G_mask||) before the anchor fires, and the
# seed lingers via the beta_anc=0.9 EMA — contaminating the injected direction.
# seed_anchor_cache=false ⇒ M_anchor starts at ZEROS ⇒ inject_matrix is a no-op
# until the live anchor fires (step ~3), then injects the REAL stale true-gradient
# direction. Code (exp/18-anchorinject-c5d5) is already installed at /workspace/verl;
# this only re-runs training with the corrected config (no bundle re-apply).
set -uo pipefail
cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_anchorinject_c5_d5.flag
# quick sanity: inject code present
python -c "from verl.workers.comm_eff.spectral_filter import SpectralFilter; assert hasattr(SpectralFilter,'inject_matrix'); print('inject_matrix present')" || { echo "INJECT CODE MISSING — abort"; exit 3; }

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorinject_c5_d5 \
COMM_EFF_ENABLED=true \
COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True \
NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=inject \
  actor_rollout_ref.actor.comm_eff.spectral.inject_gamma=1.0 \
  > /workspace/runs/EXP-18/train_curvematch_anchorinject_c5_d5.log 2>&1
RC=$?
echo "$(date -Iseconds) rc=$RC" > /workspace/runs/EXP-18/done_curvematch_anchorinject_c5_d5.flag
echo "=== C1 clean relaunch finished rc=$RC at $(date -u +%FT%TZ) ==="
