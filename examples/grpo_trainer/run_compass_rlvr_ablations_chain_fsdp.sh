#!/usr/bin/env bash
# run_compass_rlvr_ablations_chain_fsdp.sh
#
# Per-GPU queues for the COMPASS ablation family. Each GPU runs its arms one
# after another, so the box stays saturated without ever putting two arms on
# one device.
#
# The queues are balanced to 600 optimizer steps per GPU, which is the length
# of the longest single arm, so every GPU finishes at about the same time:
#
#   GPU 0   noanchor(600)
#   GPU 1   base(200)  -> k80gm(400)
#   GPU 2   k40(200)   -> nosign(200) -> k10(200)
#   GPU 3   k80(400)   -> k40gm(200)
#
# It does NOT fetch. The arms must all run against one tree or they are not
# code-matched, and a `git reset` under a live run would swap its code
# mid-flight because verl is installed editable. Pin the tree before starting
# and leave it alone; cherry-pick arm-script-only changes if you need them.
#
# Usage, from /workspace, inside its own tmux session:
#   tmux new -d -s chain \
#     'bash /workspace/verl/examples/grpo_trainer/run_compass_rlvr_ablations_chain_fsdp.sh'
set -uo pipefail

WORK="${WORK:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_SH="$HERE/run_compass_rlvr_ablations_fsdp.sh"
POLL="${POLL:-120}"
SESSION="${SESSION:-compass}"
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"

# GPU index -> remaining arms, in order. The arm already running on each GPU
# is deliberately NOT listed; the chain waits for it and then continues.
QUEUE_0="${QUEUE_0:-}"
QUEUE_1="${QUEUE_1:-k80gm}"
QUEUE_2="${QUEUE_2:-nosign k10}"
QUEUE_3="${QUEUE_3:-k40gm}"

PIN="$(cd "$WORK/verl" && git rev-parse --short HEAD)"
echo "=== chain starting against pinned tree $PIN, no fetch will happen ==="

gpu_busy() {
  # A GPU is busy while any compute process holds it. This is the honest
  # signal: done.flag can be missing on an engine that exited non-zero, and a
  # crashed arm frees the device just as a finished one does.
  local g="$1" n
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -c "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g" 2>/dev/null)")
  [[ "${n:-0}" -gt 0 ]]
}

run_queue() {
  local gpu="$1"; shift
  local arms=("$@")
  [[ ${#arms[@]} -eq 0 ]] && { echo "[gpu$gpu] nothing queued"; return 0; }
  for arm in "${arms[@]}"; do
    # wait for whatever is on this GPU to finish or die
    local waited=0
    while gpu_busy "$gpu"; do
      sleep "$POLL"; waited=$(( waited + POLL ))
    done
    # settle: let the previous process release memory and any R2 upload drain
    sleep 60
    while pgrep -f '[a]ws s3' >/dev/null 2>&1; do sleep 30; done
    echo "[gpu$gpu] $(date +%H:%M:%S) starting $arm after ${waited}s wait"
    mkdir -p "$WORK/runs/compass-$arm" "$WORK/verl/runs/compass-$arm"
    tmux kill-window -t "$SESSION:$arm" 2>/dev/null
    tmux new-window -d -t "$SESSION" -n "$arm" \
      "ARM=$arm GPU=$gpu WORK=$WORK SKIP_CHECKOUT=1 DATA_DIR=$DATA_DIR bash $ARM_SH 2>&1 | tee -a $WORK/runs/compass-$arm/boot.log; exec bash"
    # give it time to claim the device before the next wait loop reads it as free
    sleep 420
  done
  echo "[gpu$gpu] queue drained"
}

for g in 0 1 2 3; do
  eval "q=\$QUEUE_$g"
  # shellcheck disable=SC2086
  run_queue "$g" $q &
done
wait
echo "=== all per-GPU queues drained at $(date +%H:%M:%S) ==="
echo "Collect with: python3 research/scripts/collect_compass_ablations.py <dir> --latex"
