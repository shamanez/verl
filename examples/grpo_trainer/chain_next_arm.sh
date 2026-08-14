#!/usr/bin/env bash
# chain_next_arm.sh
#
# Wait for one arm of the Qwen3-4B 4k study to FINISH CLEANLY, then launch the
# next arm on the same box.
#
#   bash chain_next_arm.sh <prev_experiment_name> <next_arm>
#   bash chain_next_arm.sh qwen3-4b-4k-freshm-500 powersgdq
#
# Run it from OUTSIDE the repo (e.g. /workspace/chain_next_arm.sh) inside its own
# tmux session. It must not live in the working tree it is about to git-reset,
# or bash will read a half-replaced script.
#
# WHY IT WAITS ON done.flag AND NOT ON THE PROCESS: the engine's teardown can
# return non-zero after a perfectly clean run, and a killed run leaves no flag at
# all. done.flag is written only after training returns, so it is the one signal
# that means "finished", not "stopped".
#
# It deliberately does NOT start the next arm when the previous one died without
# a flag. A crash, an OOM, or a collapse-triggered kill all want a human to look
# before another 14 hours of GPU time is spent, and starting anyway would also
# race the dead run's checkpoint uploads for disk and uplink.
set -uo pipefail

PREV_EXP="${1:?usage: chain_next_arm.sh <prev_experiment_name> <next_arm>}"
NEXT_ARM="${2:?usage: chain_next_arm.sh <prev_experiment_name> <next_arm>}"

WORK="${WORK:-/workspace}"
BRANCH="${BRANCH:-exp/qwen3-4b-4k-500}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
POLL="${POLL:-120}"
# The previous arm's last checkpoint keeps uploading after done.flag. Let it
# drain before the next arm starts competing for the same disk and uplink.
DRAIN_MAX="${DRAIN_MAX:-3600}"

FLAG="$WORK/verl/runs/$PREV_EXP/done.flag"
PREV_LOG="$WORK/runs/$PREV_EXP/train.log"
CHAIN_LOG="$WORK/runs/chain-${NEXT_ARM}.log"
mkdir -p "$(dirname "$CHAIN_LOG")"

say() { echo "[$(date -Iseconds)] $*" | tee -a "$CHAIN_LOG"; }

say "chain armed: waiting for $PREV_EXP to finish, then launching arm '$NEXT_ARM'"
say "  flag     $FLAG"
say "  poll     ${POLL}s"

dead_polls=0
while true; do
  if [[ -f "$FLAG" ]]; then
    say "done.flag present: $PREV_EXP finished cleanly"
    break
  fi
  # No flag yet. Is training still alive? Two consecutive dead polls with no
  # flag means it will never appear.
  if pgrep -f "[m]ain_ppo" >/dev/null 2>&1; then
    dead_polls=0
  else
    dead_polls=$((dead_polls + 1))
    say "no training process and no done.flag (${dead_polls}/2)"
    if (( dead_polls >= 2 )); then
      say "ABORT: $PREV_EXP ended WITHOUT done.flag. Not starting '$NEXT_ARM'."
      say "       Inspect the tail of $PREV_LOG, then launch manually with:"
      say "       cd $WORK/verl && bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh $NEXT_ARM"
      exit 2
    fi
  fi
  sleep "$POLL"
done

# Let the previous arm's async R2 uploads finish.
waited=0
while pgrep -f "[a]ws s3" >/dev/null 2>&1; do
  (( waited >= DRAIN_MAX )) && { say "R2 uploads still running after ${DRAIN_MAX}s; continuing anyway"; break; }
  say "waiting for $PREV_EXP checkpoint uploads to drain (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done
say "uploads drained after ${waited}s"

# Refresh the checkout ONLY NOW. Doing this while the previous arm was training
# would swap the editable-installed source under a live process.
cd "$WORK/verl" || { say "FATAL: cannot cd $WORK/verl"; exit 1; }
git remote set-url origin "$REPO"
if git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD && git reset --hard FETCH_HEAD; then
  say "checkout refreshed to $(git rev-parse --short HEAD) ($BRANCH)"
else
  say "FATAL: could not refresh the checkout to $BRANCH"; exit 1
fi

# The arm must exist in the refreshed launcher, or we would burn the wait.
if ! grep -q "\"$NEXT_ARM\"" examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh; then
  say "FATAL: arm '$NEXT_ARM' not found in the refreshed launcher"; exit 1
fi

say "launching arm '$NEXT_ARM'"
exec bash examples/grpo_trainer/run_qwen3_4b_4k_500_fsdp.sh "$NEXT_ARM"
