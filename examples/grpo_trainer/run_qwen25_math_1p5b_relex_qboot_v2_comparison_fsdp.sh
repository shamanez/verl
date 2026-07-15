#!/usr/bin/env bash
# Two-arm qboot-v2 comparison on the locked Qwen2.5-Math-1.5B / MATH surface.
#
# Both arms keep the fixed model/data/optimizer/compression surface and the new
# Q/M behavior fixed:
#   * one dense fast-actor observation bootstraps Q before the first compressed
#     old-logprob/current-policy PPO pair;
#   * stale_correct supplies dense all-floating M at the first anchor fire;
#   * W=4 retention with min_snapshots=2 gives W2, W3, then sliding W4;
#   * every unique floating gradient-bearing parameter is covered by M.
#
# The no_weight_increment arm still computes W2/W3/W4 estimates, but strength 0
# makes the applied anchor weights exactly the newest transferred checkpoint;
# stale_paired routes that checkpoint's exact paired trajectories. The
# composite arm sets strength 1, projects every floating weight tensor
# independently, and uses current trajectories on projected fires.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$HERE/../.." && pwd)"
LAUNCHER="$HERE/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"

if (( $# == 0 )); then
  ARMS=(no_weight_increment composite)
else
  ARMS=("$@")
fi

if [[ -n "${EXPERIMENT_NAME_OVERRIDE:-}" && ${#ARMS[@]} -ne 1 ]]; then
  echo "FATAL: EXPERIMENT_NAME_OVERRIDE requires exactly one selected arm" >&2
  exit 2
fi

run_arm() {
  local arm="$1"
  local experiment strength role rollout_source

  case "$arm" in
    no_weight_increment)
      experiment="relex_cmp_qboot_v2_w4min2_allfloatm_alpha0_stalepaired_math_qwen25_math_1p5b"
      strength=0.0
      role="two-circuit control: zero applied projected weight increment"
      rollout_source=stale_paired
      ;;
    composite)
      experiment="relex_cmp_qboot_v2_w4min2_allfloatm_alpha1_math_qwen25_math_1p5b"
      strength=1.0
      role="composite per-tensor W2/W3/W4 projection"
      rollout_source=auto
      ;;
    *)
      echo "FATAL: unknown qboot-v2 arm '$arm' (expected no_weight_increment or composite)" >&2
      return 2
      ;;
  esac

  # A corrected single-arm validation may use a fresh run name while reusing
  # this exact locked configuration. Multi-arm queues reject the override above
  # to prevent directory/W&B collisions.
  experiment="${EXPERIMENT_NAME_OVERRIDE:-$experiment}"

  local run_dir="$VERL_ROOT/runs/$experiment"
  if [[ -e "$run_dir" && "${ALLOW_EXISTING_RUN:-0}" != "1" ]]; then
    echo "FATAL: refusing to overwrite existing run directory: $run_dir" >&2
    return 2
  fi

  echo "=== qboot-v2 arm=$arm experiment=$experiment ==="
  echo "    role=$role"
  echo "    fast_q_bootstrap=true warmup=stale_correct M_scope=all_floating"
  echo "    mode=rank1_relex window=4 min_snapshots=2 strength=$strength rollout_source=$rollout_source"

  env \
    PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}" \
    EXPERIMENT_NAME="$experiment" \
    LOG="$run_dir/train.log" \
    MODEL_PATH="Qwen/Qwen2.5-Math-1.5B" \
    DATA_DIR="${DATA_DIR:-/workspace/data/math}" \
    TOTAL_EPOCHS=8 \
    TOTAL_TRAINING_STEPS=100 \
    TEST_FREQ=25 \
    SAVE_FREQ=-1 \
    VAL_BEFORE_TRAIN=True \
    COMM_EFF_ENABLED=true \
    COMM_EFF_COMPRESSION_TYPE=powersgd \
    COMM_EFF_POWERSGD_RANK=77 \
    COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true \
    COMM_EFF_POWERSGD_SYNC_BASIS=true \
    COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=true \
    COMM_EFF_ANCHOR_ENABLED=true \
    COMM_EFF_ANCHOR_CADENCE=20 \
    COMM_EFF_ANCHOR_DELAY_K=20 \
    COMM_EFF_ANCHOR_OWNS_Q=true \
    COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH=true \
    COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR=true \
    COMM_EFF_ANCHOR_LOOKAHEAD_MODE=rank1_relex \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=4 \
    COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=2 \
    COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="$strength" \
    COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="$rollout_source" \
    COMM_EFF_ANCHOR_WARMUP_MODE=stale_correct \
    COMM_EFF_SPECTRAL_ENABLED=true \
    COMM_EFF_SPECTRAL_TARGET_SCOPE=all_floating \
    COMM_EFF_SPECTRAL_DIAGNOSTICS=false \
    COMM_EFF_SPECTRAL_BETA_ANC=0.50 \
    COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.25 \
    COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
    COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
    bash "$LAUNCHER"

  if [[ ! -f "$run_dir/done.flag" ]]; then
    echo "FATAL: arm $arm returned success without $run_dir/done.flag" >&2
    return 1
  fi
  if [[ -f "$run_dir/EARLY_STOP_SIGNAL" ]]; then
    echo "FATAL: arm $arm produced EARLY_STOP_SIGNAL; stopping the sweep" >&2
    return 1
  fi
  echo "=== qboot-v2 arm=$arm complete ==="
}

for arm in "${ARMS[@]}"; do
  run_arm "$arm"
done
