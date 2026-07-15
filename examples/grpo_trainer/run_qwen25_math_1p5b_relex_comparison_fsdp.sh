#!/usr/bin/env bash
# Sequential Qwen2.5-Math-1.5B / MATH comparison matrix for rank1_relex.
#
# Default order: W2/secant, corrected strict-readiness W4, progressive
# W2->W3->W4 readiness, matched W2 no-increment, legacy decoder-only
# fixed_linear, strict dense GRPO. Pass arm names to run a subset, e.g.
#
#   bash run_qwen25_math_1p5b_relex_comparison_fsdp.sh w2_rank1 w4_rank1 w4_progressive
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$HERE/../.." && pwd)"
LAUNCHER="$HERE/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"

if (( $# == 0 )); then
  ARMS=(w2_rank1 w4_rank1 w4_progressive w2_no_increment fixed_linear dense)
else
  ARMS=("$@")
fi

if [[ -n "${EXPERIMENT_NAME_OVERRIDE:-}" && ${#ARMS[@]} -ne 1 ]]; then
  echo "FATAL: EXPERIMENT_NAME_OVERRIDE requires exactly one selected arm" >&2
  exit 2
fi

run_arm() {
  local arm="$1"
  local experiment mode window min_snapshots strength warmup probe_enabled comm_enabled
  local compression_type=powersgd
  local anchor_enabled=true anchor_owns_q=true replay=true lookahead=true spectral=true correction=signed_ema

  case "$arm" in
    w2_rank1)
      experiment="relex_cmp_qstagefix_v1_w2_alltensor_secant_alpha1_math_qwen25_math_1p5b"
      mode=rank1_relex window=2 min_snapshots=-1 strength=1.0 warmup=q_only probe_enabled=true comm_enabled=true
      ;;
    w4_rank1)
      # Corrected rerun of the original W=4 arm. The qstagefix label distinguishes
      # the frozen-Q PPO handoff from the legacy diagnostic run whose anchor Q
      # changed inside update_actor at cadence boundaries.
      experiment="relex_cmp_qstagefix_v1_w4_alltensor_rank1_alpha1_math_qwen25_math_1p5b"
      mode=rank1_relex window=4 min_snapshots=-1 strength=1.0 warmup=q_only probe_enabled=true comm_enabled=true
      ;;
    w4_progressive)
      # Keep W=4 as the retained target window, but start from the earliest
      # legal two-checkpoint estimate: W2 secant at global step 20, W3 rank1 at
      # step 30, then sliding W4 from step 40 onward (C=K=20, two ticks/step).
      experiment="relex_cmp_qstagefix_v1_w4_progressive_min2_alltensor_alpha1_math_qwen25_math_1p5b"
      mode=rank1_relex window=4 min_snapshots=2 strength=1.0 warmup=q_only probe_enabled=true comm_enabled=true
      ;;
    w2_no_increment)
      experiment="relex_cmp_qstagefix_v1_w2_alltensor_alpha0_math_qwen25_math_1p5b"
      mode=rank1_relex window=2 min_snapshots=-1 strength=0.0 warmup=q_only probe_enabled=true comm_enabled=true
      ;;
    fixed_linear)
      # Legacy comparator: two-source linear projection over decoder matrices
      # only. q_only is rank1-only, so this uses the canonical stale_correct
      # warmup and discloses that schedule difference in the run name/docs.
      experiment="relex_cmp_qstagefix_v1_fixed_linear_decoder_alpha1_math_qwen25_math_1p5b"
      mode=fixed_linear window=2 min_snapshots=-1 strength=1.0 warmup=stale_correct probe_enabled=false comm_enabled=true
      ;;
    dense)
      experiment="relex_cmp_dense_grpo_math_qwen25_math_1p5b"
      mode=disabled window=2 min_snapshots=-1 strength=0.0 warmup=stale_correct probe_enabled=false comm_enabled=false
      compression_type=dense
      anchor_enabled=false anchor_owns_q=false replay=false lookahead=false spectral=false correction=none
      ;;
    *)
      echo "FATAL: unknown arm '$arm' (expected w2_rank1, w4_rank1, w4_progressive, w2_no_increment, fixed_linear, dense)" >&2
      return 2
      ;;
  esac

  # A single-arm validation may supply a unique immutable run name without
  # copying the locked arm definition. Multi-arm queues reject this override
  # above so two cells can never collide in one directory/W&B run.
  experiment="${EXPERIMENT_NAME_OVERRIDE:-$experiment}"

  local run_dir="$VERL_ROOT/runs/$experiment"
  if [[ -e "$run_dir" && "${ALLOW_EXISTING_RUN:-0}" != "1" ]]; then
    echo "FATAL: refusing to overwrite existing run directory: $run_dir" >&2
    return 2
  fi

  echo "=== comparison arm=$arm experiment=$experiment ==="
  echo "    mode=$mode window=$window min_snapshots=$min_snapshots strength=$strength warmup=$warmup probe=$probe_enabled comm_eff=$comm_enabled codec=$compression_type"
  env \
    PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}" \
    EXPERIMENT_NAME="$experiment" \
    LOG="$run_dir/train.log" \
    MODEL_PATH="Qwen/Qwen2.5-Math-1.5B" \
    DATA_DIR="/workspace/data/math" \
    TOTAL_EPOCHS=8 \
    TOTAL_TRAINING_STEPS=100 \
    TEST_FREQ=25 \
    SAVE_FREQ=-1 \
    VAL_BEFORE_TRAIN=True \
    COMM_EFF_ENABLED="$comm_enabled" \
    COMM_EFF_COMPRESSION_TYPE="$compression_type" \
    COMM_EFF_ANCHOR_ENABLED="$anchor_enabled" \
    COMM_EFF_ANCHOR_OWNS_Q="$anchor_owns_q" \
    COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="$replay" \
    COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="$lookahead" \
    COMM_EFF_ANCHOR_LOOKAHEAD_MODE="$mode" \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="$window" \
    COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="$min_snapshots" \
    COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="$strength" \
    COMM_EFF_ANCHOR_WARMUP_MODE="$warmup" \
    COMM_EFF_SPECTRAL_ENABLED="$spectral" \
    COMM_EFF_SPECTRAL_CORRECTION_MODE="$correction" \
    COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED="$probe_enabled" \
    COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES="${COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES:-16}" \
    COMM_EFF_PROBE_OUT_DIR="$run_dir/rank1_projection_probe" \
    COMM_EFF_PROBE_RANK0_ONLY=true \
    bash "$LAUNCHER"

  if [[ ! -f "$run_dir/done.flag" ]]; then
    echo "FATAL: arm $arm returned success without $run_dir/done.flag" >&2
    return 1
  fi
  if [[ -f "$run_dir/EARLY_STOP_SIGNAL" ]]; then
    echo "FATAL: arm $arm produced EARLY_STOP_SIGNAL; stopping the sweep" >&2
    return 1
  fi
  echo "=== comparison arm=$arm complete ==="
}

for arm in "${ARMS[@]}"; do
  run_arm "$arm"
done
