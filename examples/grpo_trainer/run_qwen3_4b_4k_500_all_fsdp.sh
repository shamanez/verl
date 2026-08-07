#!/usr/bin/env bash
# run_qwen3_4b_4k_500_all_fsdp.sh
#
# The whole Qwen3-4B-Base 4096-context study on ONE 4x H200 box, in one command:
#
#   tmux new -s q4b
#   bash examples/grpo_trainer/run_qwen3_4b_4k_500_all_fsdp.sh
#
#   stage 1  compressed arm, 500 steps, all 4 GPUs   (runs FIRST, by design)
#   stage 2  dense control,  500 steps, all 4 GPUs
#   stage 3  in-domain + OOD eval of both arms x 5 checkpoints + the untrained base,
#            fanned over two GPU pairs
#   stage 4  the comparison page
#
# WHY THE ARMS RUN SEQUENTIALLY AT FULL WIDTH rather than 2 GPUs each in
# parallel: with rollout tensor-parallel 1 the four GPUs are four independent
# vLLM replicas, so generation, which dominates the step, scales close to
# linearly with GPU count. Two 2-GPU arms therefore finish at roughly the same
# wall clock as two sequential 4-GPU arms while doubling the chance of an OOM,
# halving the per-rank headroom the anchor's unsharded clone needs, and running
# two checkpoint streams at the same disk. Sequential is the same speed and the
# lower risk. Set PARALLEL_ARMS=1 to take the other path anyway.
#
# Every stage is resumable and skips work that already has a result, so a killed
# session can be restarted with the same command.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
RUN_ID="${RUN_ID:-qwen3-4b-4k-500}"
STAGES="${STAGES:-smoke commeff dense eval report}"
# Smoke gate. Qwen3 base models are wordier than the Qwen2.5-Math model this
# recipe was tuned on, and a response distribution pressed against the cap is the
# truncation feedback loop that ended the last long-context attempt. The
# reference 1.5B run held a 574-token mean with 2 percent clipped and fell over
# time, so a fifth of the batch clipping within the first few steps is already
# far outside anything that has trained here.
SMOKE_CLIP_MAX="${SMOKE_CLIP_MAX:-0.35}"
PARALLEL_ARMS="${PARALLEL_ARMS:-0}"
export RUN_ID

STATE="$WORK/runs/$RUN_ID"
mkdir -p "$STATE"
SUMMARY="$STATE/stages.log"

stage_done() { [[ -f "$STATE/$1.done" ]]; }
mark_done()  { touch "$STATE/$1.done"; }
say() { echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"; }

run_arm() {  # run_arm <commeff|dense>
  local arm="$1"
  if stage_done "$arm"; then say "SKIP arm $arm (already marked done)"; return 0; fi
  local exp="qwen3-4b-4k-${arm}-500"
  local flag="$WORK/verl/runs/$exp/done.flag"
  rm -f "$flag"
  say "START arm $arm"
  bash "$HERE/run_qwen3_4b_4k_500_fsdp.sh" "$arm"
  local rc=$?
  # The engine's exit status is not on its own a reliable completion signal: its
  # teardown path can return non-zero after a perfectly clean run. done.flag is
  # written only after training returns, so require BOTH to agree before
  # declaring an arm finished, and say which one disagreed when they do not.
  if [[ -f "$flag" ]] && (( rc == 0 )); then
    mark_done "$arm"; say "DONE arm $arm (rc=0, done.flag present)"
    return 0
  fi
  if [[ -f "$flag" ]]; then
    say "arm $arm wrote done.flag but exited rc=$rc — training reached the end, teardown did not."
    say "  check the tail of $WORK/runs/$exp/train.log, then: touch $STATE/$arm.done to continue"
  else
    say "FAILED arm $arm rc=$rc, no done.flag — training did not finish. Fix and rerun this script."
  fi
  return 1
}

for stage in $STAGES; do
  case "$stage" in
    smoke)
      if stage_done smoke; then say "SKIP smoke"; continue; fi
      say "START smoke (few steps, no val, no checkpoints, no R2)"
      SMOKE=1 bash "$HERE/run_qwen3_4b_4k_500_fsdp.sh" commeff
      SMOKE_LOG="$WORK/runs/qwen3-4b-4k-commeff-500/smoke.log"
      if [[ ! -s "$SMOKE_LOG" ]]; then
        say "smoke produced no log at $SMOKE_LOG — the box is not ready, stopping"
        exit 1
      fi
      SMOKE_LOG="$SMOKE_LOG" SMOKE_CLIP_MAX="$SMOKE_CLIP_MAX" python3 - <<'PY' | tee -a "$SUMMARY"
import os, re, sys
txt = open(os.environ["SMOKE_LOG"], errors="replace").read()
def series(key):
    return [float(x) for x in re.findall(re.escape(key) + r"[\"']?[: ]+(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)", txt)]
rl, clip, sec = series("response_length/mean"), series("response_length/clip_ratio"), series("timing_s/step")
mem, res = series("perf/max_memory_allocated_gb"), series("perf/max_memory_reserved_gb")
if not rl:
    print("SMOKE: no response_length in the log — training never produced a step")
    sys.exit(2)
cap = float(os.environ["SMOKE_CLIP_MAX"])
tail = lambda v: v[-3:] if v else []
print(f"SMOKE steps completed       {len(rl)}")
print(f"SMOKE response_length/mean  first={rl[0]:.0f} last={rl[-1]:.0f} max={max(rl):.0f} of 3072")
print(f"SMOKE response_length/clip  first={clip[0]:.3f} last={clip[-1]:.3f} max={max(clip):.3f}" if clip else "SMOKE clip: absent")
print(f"SMOKE timing_s/step         {[round(x) for x in tail(sec)]}")
if sec and len(sec) > 1:
    body = sorted(sec[1:])
    est = body[len(body) // 2]
    print(f"SMOKE median {est:.0f} s/step (step 1 dropped) -> {500 * est / 3600:.1f} h per arm, {2 * 500 * est / 3600:.1f} h for both")
# The anchor's replay clone does not shard, and its peak appears only at the
# first fire on step 20. The 1.5B reference read 34 GB at step 1 and 109 GB at
# step 25, so a short probe under-reports the peak by a factor of three.
if mem:
    print(f"SMOKE peak allocated        {max(mem):.1f} GB   reserved {max(res) if res else float('nan'):.1f} GB")
    if len(rl) < 21:
        print("SMOKE WARN: the probe stopped before step 20, so the first anchor fire never happened")
        print("            and this peak is NOT the peak the 500-step run will reach.")
    elif max(mem) > 125.0:
        print(f"SMOKE FAIL: {max(mem):.1f} GB allocated leaves almost nothing on a 141 GB H200.")
        print("            Ladder: ROLLOUT_GPU_MEM_UTIL 0.60 -> 0.50, then")
        print("            LOG_PROB_MAX_TOKEN_LEN_PER_GPU 24576 -> 18432, then PPO_MAX_TOKEN_LEN_PER_GPU.")
        sys.exit(4)
if clip and max(clip) > cap:
    print(f"SMOKE FAIL: clip_ratio peaked at {max(clip):.3f}, above the {cap:.2f} gate.")
    print("            Responses are pressed against the 3072 cap. Raise MAX_RESPONSE_LENGTH")
    print("            (and MAX_MODEL_LEN with it) or accept truncation feedback. Not starting 500 steps.")
    sys.exit(3)
print("SMOKE OK")
PY
      rc=${PIPESTATUS[0]}
      if (( rc != 0 )); then say "smoke gate FAILED rc=$rc — stopping before the 500-step arms"; exit 1; fi
      mark_done smoke; say "DONE smoke"
      ;;
    commeff|dense)
      if [[ "$PARALLEL_ARMS" == "1" ]]; then continue; fi
      run_arm "$stage" || exit 1
      ;;
    eval)
      if stage_done eval; then say "SKIP eval"; continue; fi
      say "START eval matrix"
      # The eval reuses the training checkout, so it needs the same working dir.
      VERL_DIR="${VERL_DIR:-$WORK/verl}" \
      OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-$WORK/runs/ood-eval-4b}" \
      CKPT_ROOT="${CKPT_ROOT:-$WORK/verl/checkpoints/$RUN_ID}" \
      PAIRS_CSV="${PAIRS_CSV:-0,1|2,3}" \
        bash "$WORK/verl/research/scripts/ood_eval/eval_qwen3_4b_4k.sh"
      if [[ -f "${OOD_EVAL_ROOT:-$WORK/runs/ood-eval-4b}/OOD_DONE" ]]; then
        mark_done eval; say "DONE eval matrix"
      else
        say "eval matrix incomplete — rerun to fill the remaining cells"
      fi
      ;;
    report)
      say "START report"
      OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-$WORK/runs/ood-eval-4b}"
      python3 "$WORK/verl/research/scripts/ood_eval/report_qwen3_4b_4k.py" \
        --results "$OOD_EVAL_ROOT/results.json" \
        --out     "$OOD_EVAL_ROOT/qwen3-4b-4k-comparison.html" \
        && say "DONE report -> $OOD_EVAL_ROOT/qwen3-4b-4k-comparison.html"
      ;;
    *) say "unknown stage '$stage'"; exit 1 ;;
  esac
done

if [[ "$PARALLEL_ARMS" == "1" ]]; then
  say "PARALLEL_ARMS=1: launch the two arms yourself on disjoint GPU sets, e.g."
  say "  CUDA_VISIBLE_DEVICES=0,1 EXPECT_GPUS=2 bash $HERE/run_qwen3_4b_4k_500_fsdp.sh commeff"
  say "  CUDA_VISIBLE_DEVICES=2,3 EXPECT_GPUS=2 bash $HERE/run_qwen3_4b_4k_500_fsdp.sh dense"
fi

say "ALL REQUESTED STAGES COMPLETE"
