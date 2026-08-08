#!/usr/bin/env bash
# run_99_both_arms.sh
#
# Drive BOTH arms of run 99 on one 4-GPU box, SEQUENTIALLY, each at the full
# GPU count:
#
#   bash examples/grpo_trainer/run_99_both_arms.sh
#
# Sequential, not two 2-GPU runs side by side, and that is a deliberate choice:
#   - both arms then train at the SAME FSDP world size, so their checkpoints
#     have the same shard layout and their per-GPU dynamic token budgets match.
#     The comparison is the deliverable, and a world-size difference between arms is
#     a confound for free.
#   - one Ray head, one vLLM stack, one set of ports and one temp dir at a time.
#     Two concurrent Ray clusters on one host is a separate engineering problem
#     and this run does not need to solve it.
#
# DENSE RUNS FIRST, by default, and that is also deliberate. It prices the
# surface before the method spends anything on it: the step-0 validation gives
# the untrained in-domain score, and step 1 gives response_length/clip_ratio,
# which at a 3072-token cap on a long-CoT model is the truncation rate. If the
# dense arm cannot learn here, the compressed arm has nothing to be compared to.
# Override with ARM_ORDER="prf dense" to lead with the method instead.
#
# Resumable: an arm whose engine wrote done.flag is skipped, so re-running after
# an interruption picks up where it stopped.
#
# NOT set -e. One arm's non-zero exit is recorded in fail_<arm>.flag and the
# sweep CONTINUES to the next arm -- a dense-arm crash should not cost the
# compressed arm its box time. Semantic stops are the operator's call.
set -uo pipefail

RUN_ID="${RUN_ID:-99-r1distill-deepscaler-600}"
WORK="${WORK:-/workspace}"
ARM_ORDER="${ARM_ORDER:-dense prf}"
RUN_DIR="$WORK/runs/$RUN_ID"
SWEEP_LOG="$RUN_DIR/sweep.log"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$RUN_DIR" || { echo "FATAL: cannot create $RUN_DIR" >&2; exit 1; }
say() { echo "$*" | tee -a "$SWEEP_LOG"; }

say ""
say "=== run 99 sweep  $(date -Iseconds) ==="
say "    run        $RUN_ID"
say "    arms       $ARM_ORDER (sequential, full GPU count each)"
say "    run dir    $RUN_DIR"

# One sweep at a time per run dir. Atomic mkdir plus a `kill -0` liveness test on
# the recorded pid: NEVER pgrep/pkill on a pattern, which in this project has
# repeatedly matched the checking command itself (and once killed the caller).
LOCK_DIR="$RUN_DIR/.sweep.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  OTHER="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$OTHER" =~ ^[0-9]+$ ]] && kill -0 "$OTHER" 2>/dev/null; then
    echo "FATAL: another run_99_both_arms.sh (pid $OTHER) is already driving $RUN_DIR." >&2
    echo "       Attach to its tmux window, or wait. Delete $LOCK_DIR only if you are sure it is dead." >&2
    exit 1
  fi
  echo "WARN: clearing a stale lock at $LOCK_DIR (recorded pid '${OTHER:-none}' is not running)" >&2
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" || { echo "FATAL: could not take the lock at $LOCK_DIR" >&2; exit 1; }
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

# ---------------------------------------------------------------------------
# Per-arm health summary, parsed from the finished arm's own train.log.
# Nothing here reads WandB: the rc=1 atexit teardown race has silently dropped
# final-step values before, and train.log is the authoritative record.
# ---------------------------------------------------------------------------
health_summary() {  # health_summary <arm>
  local arm="$1" log="$RUN_DIR/$arm/train.log"
  [[ -f "$log" ]] || { say "    no train.log for $arm"; return; }
  say ""
  say "--- health summary: $arm"
  ARM="$arm" LOG_PATH="$log" python3 - <<'PY' 2>&1 | tee -a "$SWEEP_LOG"
import os
import re

arm = os.environ["ARM"]
path = os.environ["LOG_PATH"]
with open(path, encoding="utf-8", errors="replace") as fh:
    text = fh.read()


def series(key):
    """Every (step, value) for a metric, in log order."""
    out = []
    for m in re.finditer(r"step:(\d+)[^\n]*?" + re.escape(key) + r":([0-9eE.+-]+)", text):
        try:
            out.append((int(m.group(1)), float(m.group(2))))
        except ValueError:
            pass
    return out


# The truncation rate. At a 3072-token cap on a long-CoT model this is THE
# number: it says how much of the reward signal is "ran out of tokens".
clip = series("response_length/clip_ratio")
if clip:
    steps = [v for _, v in clip]
    print(f"  response_length/clip_ratio  first={steps[0]:.3f}  last={steps[-1]:.3f}  "
          f"max={max(steps):.3f}  mean={sum(steps) / len(steps):.3f}")
    if max(steps) > 0.5:
        print("    WARN: over half the completions hit the response cap at some point.")
        print("          A large share of the zero rewards are truncations, not wrong answers.")
else:
    print("  response_length/clip_ratio  not found in the log")

for key in ("response_length/mean", "critic/score/mean", "actor/grad_norm", "actor/kl_loss"):
    s = series(key)
    if s:
        vals = [v for _, v in s]
        print(f"  {key:<28s} first={vals[0]:.4g}  last={vals[-1]:.4g}  max={max(vals):.4g}")

# In-domain validation curve, from the val-core key the trainer logs.
vals = re.findall(r"val-core/\S*?acc/mean@\d+['\"]?[: ]+([0-9.]+)", text)
if vals:
    print(f"  in-domain val ({len(vals)} points): " + " -> ".join(f"{float(v):.4f}" for v in vals))
else:
    print("  in-domain val: no val-core acc/mean@ lines in the log")

steps_seen = re.findall(r"global_step[:= ]+(\d+)", text)
if steps_seen:
    print(f"  last global_step reached: {max(int(s) for s in steps_seen)}")
PY
}

# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086
for arm in $ARM_ORDER; do
  case "$arm" in
    prf|dense) ;;
    *) say "SKIP unknown arm '$arm'"; continue ;;
  esac

  ARM_DIR="$RUN_DIR/$arm"
  mkdir -p "$ARM_DIR"

  if [[ -f "$WORK/verl/runs/$RUN_ID-$arm/done.flag" ]]; then
    say ""
    say "=== SKIP $arm: done.flag already present ==="
    health_summary "$arm"
    continue
  fi

  say ""
  say "=== ARM $arm  START  $(date -Iseconds) ==="
  ARM="$arm" RUN_ID="$RUN_ID" WORK="$WORK" \
    bash "$HERE/run_r1distill_deepscaler_600.sh" "$@"
  rc=$?
  say "=== ARM $arm  END rc=$rc  $(date -Iseconds) ==="
  if (( rc != 0 )); then
    printf 'rc=%s at %s\n' "$rc" "$(date -Iseconds)" > "$RUN_DIR/fail_$arm.flag"
    say "    recorded $RUN_DIR/fail_$arm.flag, continuing to the next arm"
  fi
  health_summary "$arm"
done

say ""
say "=== run 99 sweep finished  $(date -Iseconds) ==="
say "    Next: the step-200 capability audit reads both arms' checkpoints from R2."
say "      bash research/scripts/ood_eval/ckpt_eval.sh"
