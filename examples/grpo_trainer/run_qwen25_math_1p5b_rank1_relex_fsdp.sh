#!/usr/bin/env bash
# Current communication-efficient GRPO run on Qwen2.5-Math-1.5B and MATH.
# The 1024-token prompt plus 3072-token response is the 4096-token protocol.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VERL_ROOT="$(cd "$HERE/../.." && pwd)"

# Bare invocation uses the current project default.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "FATAL: prepared MATH train/test parquet files are required in $DATA_DIR" >&2
  exit 1
fi
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export ROLLOUT_N=8
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=512
export PPO_MINI_BATCH_SIZE=256
export ACTOR_LR=1e-6
export USE_KL_LOSS=True
export USE_KL_IN_REWARD=False
export KL_LOSS_COEF=0.001

# Match RELEX/scripts/eval.py's Qwen prompt byte-for-byte for BOTH rollout
# training and validation. The prepared parquet intentionally stays unchanged:
# this template replaces the prepared parquet's user suffix and pins RELEX's explicit
# system message, avoiding the Math base tokenizer's duplicate boxed prompt.
RELEX_QWEN_CHAT_TEMPLATE_FILE="$HERE/relex_qwen_chat_template.jinja"
if [[ ! -f "$RELEX_QWEN_CHAT_TEMPLATE_FILE" ]]; then
  echo "FATAL: missing RELEX Qwen chat template: $RELEX_QWEN_CHAT_TEMPLATE_FILE" >&2
  exit 1
fi
export RELEX_QWEN_CHAT_TEMPLATE
RELEX_QWEN_CHAT_TEMPLATE="$(<"$RELEX_QWEN_CHAT_TEMPLATE_FILE")"

# 512 prompts / 256-prompt mini-batch = exactly two optimizer ticks per
# GRPO step, preserving the documented generator ticks 1,19,39,59 -> 79.
export USE_DYNAMIC_BSZ=True
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ===========================================================================
# WHICH BOUNDARY CODEC TO USE, AND WHY EVERY ALTERNATIVE WAS REJECTED
# Settled by issue #93 (2026-07-29). Read this before changing
# COMM_EFF_COMPRESSION_TYPE.
#
# PRF exact-k is the only method to move forward with. Every other codec family
# we have run either FAILED outright or remains UNPROVEN at horizon. Stated
# strictly, arm by arm:
#
#   REJECTED, low-rank / basis-Q family (PowerSGD, and FRLR = low-rank + PRF
#   residual). This is the important one, because it is the family that keeps
#   looking attractive early.
#     - PowerSGD is the ONLY codec that has ever collapsed a run in this project.
#       Early-band weight rotation reached about 3.4x the dense control's NSS.
#     - FRLR is 2.55x better than PRF on the train-inference gap at step 100 and
#       then INVERTS: it first exceeds PRF at step 417, stays above from 424, and
#       ends 2.12x worse (31.10 vs 14.66 at step 599), while its gradient median
#       drifts 9.25x to a maximum of 176.4 against PRF's flat 1.50-1.82 and 4.645.
#     - WHY, and this is the mechanism, not a guess: a projection codec's error is
#       ALIGNMENT-DEPENDENT. It lives in the retained/discarded subspace, so it
#       points the same way step after step. That error is COHERENT, meaning
#       constant-direction, and the a1/a2 factorial plus the PowerSGD collapse
#       established that ref-KL drift is gated by bias TIMES coherence and NOT by
#       error magnitude. Worse, the subspace is fitted to activations the policy
#       keeps moving, so Q is always chasing and the error grows with the run.
#       PRF's error is ROTATION-INVARIANT: it has no preferred direction, nothing
#       accumulates, and the gap is stationary. That difference is the whole
#       result.
#     - Governance does not rescue it. We tried fast Q at cadence 1 (a7), slow Q
#       at cadence 20 (a8), anchor-owned Q so the basis moves only on the slow
#       circuit (a9), and an unbiased gain (a10). Q governance moves the gap and
#       the codec view and moves NEITHER capability NOR real drift: a7/a8/a9 have
#       identical probe/kl_dense to within 4 percent at matched step 120.
#     - a10 is the decisive one: with frlr_unbiased=true the gap goes to 14.88,
#       i.e. straight back to PRF's own operating point. FRLR's ENTIRE advantage
#       is bought by its biased gain, so there is no unbiased version of the win.
#     - A basis is also an operational cost PRF does not pay: Q must be broadcast
#       or coupled to the anchor, which is side-channel traffic on a link we are
#       trying not to use.
#
#   REJECTED, importance-ratio corrections (token-IS, with or without batch
#   normalisation). At PRF's ~14-nat operating point token overlap collapses about
#   170x, so E[rho] is about e^-14 and ESS measured 0.0006. a6 (PRF + token-IS)
#   COLLAPSED to val 0.5391 with grad_norm 608.8. a5 and a5b did not collapse but
#   did not learn: training reward 0.5895 and 0.6606 against the incumbent's
#   0.6726. The wall is the operating point, not the estimator, so every ratio
#   variant hits it.
#
#   REJECTED, biased quantization. a2 (sr_quant 1-bit round-to-nearest) was KILLED
#   at step 60: its run-MINIMUM grad_norm of 6.153 exceeds a1's 120-step MAXIMUM
#   of 0.898 by 6.9x, on the same codec with ONLY the estimator bias differing.
#
#   REJECTED on wire budget, not on stability. a1 (sr_quant 1-bit stochastic
#   rounding) has the tightest gradient behaviour ever measured here (max/min
#   1.2x, run max 0.898) but sends 2304 bits/token/boundary, 1.87x parity. It buys
#   its stability with bandwidth, so it is not a like-for-like win.
#
#   REJECTED as a null. a11 (this codec plus a fully uncompressed step every 50)
#   reduced the creep only to 0.86x, inside its pre-registered null band, and its
#   mechanism was refuted: no pull-back at the injections, -0.0024 nats across
#   eleven of them. The clean gradient is 30-35x SMALLER in norm than the
#   compressed ones it was meant to correct. Knob retained as mask.dense_every,
#   defaulting to 0 and inert.
#
#   UNPROVEN, not beaten, and the only two candidates worth a horizon run:
#   a4 (this codec plus CVC cross-entropy) and a3 (sr_quant 2-bit byte-parity
#   subset). Both matched or beat PRF exact-k on gap flatness AND gradient
#   tightness at 120 steps and were never run longer. a3 in fact holds the
#   flattest gap measured anywhere in the programme (+0.000101 over 100-120).
#   PRF exact-k's win is a HORIZON win: on that short 100-120 window it ranks
#   fourth. Do not read this block as "PRF is best at every scale".
#
# Single seed per arm throughout, so a difference smaller than run-to-run
# variation cannot be distinguished from one. Full reasoning and the corrections
# ledger: research/runs/93-long-horizon-stability/STABILITY_RANKING.md
# ===========================================================================

# Canonical pure rank-1 arm. Every default remains pinned, while explicit env
# overrides let the comparison launcher print the same values Hydra receives.
export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-prf_mask}"
# prf_mask codec (active iff COMM_EFF_COMPRESSION_TYPE=prf_mask). Anchor-independent
# PRF activation mask; mutually exclusive with PowerSGD and cannot anchor-own-Q, so a
# prf_mask arm also sets COMM_EFF_ANCHOR_OWNS_Q=false COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false.
# THE DEFAULT CODEC since issue #93 (2026-07-29). PRF exact-k won the #93
# stability programme: twelve arms were run to beat it and none did. It is the
# only codec with 600 steps of evidence that the optimizer stays in a steady
# state (gap slope +0.000848/step over 100-599, so 0.42 nats in 500 steps;
# grad-norm block median flat at 1.50-1.82 with a maximum never above 4.645),
# it is unbiased, and its mask is a PRF of seed/step/layer so there is no side
# channel and nothing to transmit. Set COMM_EFF_COMPRESSION_TYPE=powersgd to
# get the pre-#93 default back; that path is unchanged and still tested.
export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
export COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"
export COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-false}"
export COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"
export COMM_EFF_MASK_REFERENCE="${COMM_EFF_MASK_REFERENCE:-true}"
export COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"
export COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"
# issue #89 codec levers — default-off; baseline PRF stays bit-identical.
export COMM_EFF_MASK_RESCALE_MODE="${COMM_EFF_MASK_RESCALE_MODE:-constant}"
export COMM_EFF_MASK_EXACT_K="${COMM_EFF_MASK_EXACT_K:-true}"
export COMM_EFF_MASK_ANTITHETIC="${COMM_EFF_MASK_ANTITHETIC:-false}"
export COMM_EFF_MASK_P_BY_BOUNDARY="${COMM_EFF_MASK_P_BY_BOUNDARY:-}"
# FRLR (issue #89, "32+44+1"): fresh-residual low-rank codec; default off.
export COMM_EFF_MASK_FRLR="${COMM_EFF_MASK_FRLR:-false}"
export COMM_EFF_MASK_FRLR_RANK="${COMM_EFF_MASK_FRLR_RANK:-32}"
export COMM_EFF_MASK_FRLR_K="${COMM_EFF_MASK_FRLR_K:-44}"
export COMM_EFF_MASK_FRLR_UNBIASED="${COMM_EFF_MASK_FRLR_UNBIASED:-false}"
# Slow-Q lever: refresh the FRLR core Q every N global steps (1 = every step,
# the original behavior); Q stays frozen between refreshes while the
# activation sketch accumulates over the full window.
export COMM_EFF_MASK_FRLR_Q_CADENCE="${COMM_EFF_MASK_FRLR_Q_CADENCE:-1}"
# sr_quant codec (active iff COMM_EFF_COMPRESSION_TYPE=sr_quant). Dense low-bit
# stochastic-rounding boundary quantization; reuses COMM_EFF_MASK_RECOMPUTE /
# COMM_EFF_MASK_REFERENCE / COMM_EFF_MASK_SEED / COMM_EFF_MASK_PP_SIZE for
# eligibility and keying. Like prf_mask it cannot anchor-own-Q, so an sr_quant
# arm also sets COMM_EFF_ANCHOR_OWNS_Q=false.
export COMM_EFF_QUANT_BITS="${COMM_EFF_QUANT_BITS:-1}"
export COMM_EFF_QUANT_BLOCK_SIZE="${COMM_EFF_QUANT_BLOCK_SIZE:-32}"
export COMM_EFF_QUANT_ROUNDING="${COMM_EFF_QUANT_ROUNDING:-sr}"
export COMM_EFF_QUANT_SUBSET_K="${COMM_EFF_QUANT_SUBSET_K:-0}"
# Dense-view probe + adaptive KL coefficient (issue #93 I3). Defaults off:
# probe_every=0 keeps the trainer path bit-identical and ctrl_enabled=false
# keeps the loss on the static kl_loss_coef; the LR brake is log-only.
export COMM_EFF_PROBE_EVERY="${COMM_EFF_PROBE_EVERY:-0}"
export COMM_EFF_PROBE_CTRL_ENABLED="${COMM_EFF_PROBE_CTRL_ENABLED:-false}"
export COMM_EFF_PROBE_KL_TARGET_TABLE="${COMM_EFF_PROBE_KL_TARGET_TABLE:-}"
export COMM_EFF_PROBE_KL_TARGET_FLOOR="${COMM_EFF_PROBE_KL_TARGET_FLOOR:-0.005}"
export COMM_EFF_PROBE_KL_TARGET_GAIN="${COMM_EFF_PROBE_KL_TARGET_GAIN:-2.0}"
export COMM_EFF_PROBE_CTRL_KI="${COMM_EFF_PROBE_CTRL_KI:-0.3}"
export COMM_EFF_PROBE_CTRL_KP="${COMM_EFF_PROBE_CTRL_KP:-0.1}"
export COMM_EFF_PROBE_CTRL_BETA_MIN="${COMM_EFF_PROBE_CTRL_BETA_MIN:-2e-4}"
export COMM_EFF_PROBE_CTRL_BETA_MAX="${COMM_EFF_PROBE_CTRL_BETA_MAX:-0.05}"
# CVC: train the train-inference disagreement down (issue #93 I4). Defaults
# off (bit-identical): CVC_LAMBDA=0 disables the CE term, DC_ENABLED=false
# disables the DC-GRPO advantage shaping. When enabling DC, COMM_EFF_DC_TARGET
# must be set explicitly (measured step-1 static per-token floor plus slack).
export COMM_EFF_CVC_LAMBDA="${COMM_EFF_CVC_LAMBDA:-0.0}"
export COMM_EFF_CVC_WARMUP_STEPS="${COMM_EFF_CVC_WARMUP_STEPS:-20}"
export COMM_EFF_DC_ENABLED="${COMM_EFF_DC_ENABLED:-false}"
export COMM_EFF_DC_ETA="${COMM_EFF_DC_ETA:-1.0}"
export COMM_EFF_DC_TARGET="${COMM_EFF_DC_TARGET:--1.0}"
export COMM_EFF_DC_LAMBDA0="${COMM_EFF_DC_LAMBDA0:-0.05}"
export COMM_EFF_DC_LAMBDA_MAX="${COMM_EFF_DC_LAMBDA_MAX:-1.0}"
export COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"
# false under the prf_mask default: there is no PowerSGD basis to bootstrap.
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"
export COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
export COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
# false under the prf_mask default: the plain PRF mask has no basis Q for the
# anchor to own (its mask is a PRF of seed/step/layer), and the config
# validator rejects owns_q=true with compression_type=prf_mask unless frlr=true.
export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-false}"
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="${COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH:-true}"
export COMM_EFF_ANCHOR_BATCH_SCOPE="${COMM_EFF_ANCHOR_BATCH_SCOPE:-rollout_batch}"
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE="${COMM_EFF_ANCHOR_SNAPSHOT_DEVICE:-cpu}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="${COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR:-true}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_MODE:-rank1_relex}"
export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="${COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH:-1.0}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="${COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE:-auto}"
export COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS:-2}"
export COMM_EFF_ANCHOR_WARMUP_MODE="${COMM_EFF_ANCHOR_WARMUP_MODE:-stale_correct}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS:-2}"
# rank1_relex delta-base history mode. sliding_window (default) keeps the last
# `window` checkpoints (base advances); growing_fixed_base pins the seeded base
# and grows the base-relative delta history. max_snapshots caps growing_fixed_base
# retention (-1 unbounded; must stay -1 with sliding_window). The sliding_window /
# -1 defaults reproduce prior behavior byte-for-byte.
export COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE:-sliding_window}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS:--1}"
export COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
export COMM_EFF_SPECTRAL_TARGET_SCOPE="${COMM_EFF_SPECTRAL_TARGET_SCOPE:-all_floating}"
export COMM_EFF_SPECTRAL_DIAGNOSTICS="${COMM_EFF_SPECTRAL_DIAGNOSTICS:-false}"
export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
export COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.25}"
export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}"
export COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"
export COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"

# Run controls may change duration/logging, but not the fixed scientific surface.
# MATH has 14 full 512-prompt batches per epoch after prompt filtering. Eight
# epochs ensure trainer.total_training_steps=100 is the stopping condition.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-comm_eff_grpo_math_qwen25_math_1p5b}"
export LOG="${LOG:-$VERL_ROOT/runs/$EXPERIMENT_NAME/train.log}"

exec bash "$HERE/vast_comm_eff_engine_grpo.sh" \
  'actor_rollout_ref.model.custom_chat_template=${oc.env:RELEX_QWEN_CHAT_TEMPLATE}' \
  "$@"
