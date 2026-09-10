#!/usr/bin/env bash
# run_compass_rlvr_ablations_fanout_fsdp.sh
#
# Fan the COMPASS RLVR ablation arms out across the GPUs of one box, one arm
# per GPU, each in its own tmux window. On a 4x H200 box this runs the whole
# ablation family in roughly the wall-clock time of a single arm.
#
# Why one GPU per arm rather than one arm on four GPUs: Pair 1's published
# numbers come from a single-GPU run. At DP > 1 the anchor gradient is
# all-reduced across ranks, which changes the quantity under study, so a
# 4-GPU arm would not be comparable to the rows already in the paper.
#
# Usage:
#   bash examples/grpo_trainer/run_compass_rlvr_ablations_fanout_fsdp.sh WAVE1
#   bash examples/grpo_trainer/run_compass_rlvr_ablations_fanout_fsdp.sh WAVE2
#   ARMS="k10 k40" bash examples/grpo_trainer/run_compass_rlvr_ablations_fanout_fsdp.sh
#
# Then watch with:  tmux ls   /   tmux attach -t compass
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAVE="${1:-WAVE1}"
WORK="${WORK:-/workspace}"
SESSION="${SESSION:-compass}"
STAGGER="${STAGGER:-90}"          # seconds between boots, so vLLM warmups do not collide

case "${ARMS:-$WAVE}" in
  # Wave 1 answers A1 and opens A2. noanchor runs to 600 and therefore outlasts
  # the others; the wave-2 arms land on the GPUs that free up first.
  WAVE1) ARM_LIST=(noanchor base k40 k80) ;;
  WAVE2) ARM_LIST=(nosign k80gm k10 k40gm) ;;
  ALL)   ARM_LIST=(noanchor base k40 k80 nosign k80gm k10 k40gm) ;;
  *)     read -r -a ARM_LIST <<< "${ARMS:-}" ;;
esac
[[ ${#ARM_LIST[@]} -gt 0 ]] || { echo "FATAL: no arms selected" >&2; exit 1; }

# --- pre-flight, all of it cheap and all of it before any GPU work ---------
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
[[ "$NGPU" -ge "${#ARM_LIST[@]}" ]] \
  || { echo "FATAL: ${#ARM_LIST[@]} arms requested but only $NGPU GPU(s) present" >&2; exit 1; }

# Host RAM, not HBM, is what a K sweep spends. The replay ring holds
# delay_K/cadence + 1 fp32 full-model snapshots at about 6.2 GB each, plus the
# anchor's unsharded clone and the fp32 EMA, per arm.
need_gb=0
for a in "${ARM_LIST[@]}"; do
  case "$a" in
    noanchor|dense) per=8 ;;      # no snapshots, no ring, no EMA
    k40|k40gm)      per=40 ;;
    k80|k80gm)      per=55 ;;
    # The codec arms carry the standard anchor cost PLUS AQ-SGD's activation
    # buffer, which is a DETERMINISTIC cap (AQ_CAPACITY_GB, default 16) and not
    # a fraction of free RAM, so this number can be relied on.
    #
    # What the buffer actually is, since the batch-shaped estimate overstates
    # it by an order of magnitude: it holds one entry per (example, boundary),
    # not per sequence per step, and the G=8 rollouts of a prompt SHARE one
    # prefix entry because a prefix activation does not depend on the response
    # sampled after it. At Pair 1 that is n_examples * 7 boundaries * 1536 dims
    # * 2 bytes = about 0.16 GB per buffered token position, so covering one
    # epoch of ~100-token prompt prefixes needs about 16 GB. Covering full
    # 2048-token sequences would need about 350 GB, which is why aqsgd-all runs
    # under the same cap and lets the LRU evict: the resulting hit rate is the
    # measurement, and the storage requirement is part of the result.
    aqsgd|aqsgd-all|aqsgd-rn|aqsgd-payload)
                    per=$(( 32 + ${AQ_CAPACITY_GB:-16} )) ;;
    srquant)        per=32 ;;     # memoryless: no buffer at all
    *)              per=32 ;;
  esac
  need_gb=$(( need_gb + per ))
done
have_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
echo "=== host RAM: need ~${need_gb} GB for ${#ARM_LIST[@]} arms, box has ${have_gb:-unknown} GB ==="
if [[ -n "${have_gb:-}" && "$have_gb" -lt "$need_gb" ]]; then
  echo "FATAL: not enough host RAM. Drop the high-K arms to a later wave, or" >&2
  echo "       set ARMS to fewer arms." >&2
  exit 1
fi

# The config gate. Nothing launches until every arm passes the real validator.
if [[ -f "$HERE/../../tests/workers/comm_eff/test_compass_ablation_arms.py" ]]; then
  echo "=== config gate ==="
  ( cd "$HERE/../.." && PYTHONPATH=. python3 tests/workers/comm_eff/test_compass_ablation_arms.py ) \
    || { echo "FATAL: config gate failed, not launching" >&2; exit 1; }
fi

# Prepare MATH once. Four arms racing on the same parquet write would corrupt it.
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  ( cd "$WORK/verl" && python3 research/scripts/prepare_rlvr_math.py \
      --dataset math --local_save_dir "$DATA_DIR" ) 2>&1 | tail -5
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: MATH parquet unavailable in $DATA_DIR" >&2; exit 1; }

# Warm the HF cache once, so N arms do not all download the same weights.
python3 - <<'PY' || echo "WARN: model prefetch skipped"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-1.5B", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model cached")
PY

# One checkout for the whole box, before any arm starts. verl is installed
# editable, so a `git reset --hard` under a live run would swap its code
# mid-flight; the arms therefore run with SKIP_CHECKOUT=1.
BRANCH="${BRANCH:-exp/compass-rlvr-ablations}"
REPO="${REPO:-https://github.com/shamanez/verl.git}"
if [[ "${SKIP_CHECKOUT:-0}" != "1" ]]; then
  echo "=== checkout $BRANCH once for the whole box ==="
  ( cd "$WORK/verl" && git remote set-url origin "$REPO" \
      && git fetch origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
      && git reset --hard FETCH_HEAD ) \
    || { echo "FATAL: checkout failed" >&2; exit 1; }
fi
echo "=== tree: $(cd "$WORK/verl" && git rev-parse --abbrev-ref HEAD) $(cd "$WORK/verl" && git rev-parse --short HEAD) ==="

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n idle

gpu=0
for arm in "${ARM_LIST[@]}"; do
  win="$SESSION:$arm"
  tmux kill-window -t "$win" 2>/dev/null || true
  mkdir -p "$WORK/runs/compass-$arm"
  tmux new-window -d -t "$SESSION" -n "$arm" \
    "ARM=$arm GPU=$gpu WORK=$WORK SKIP_CHECKOUT=1 DATA_DIR=$DATA_DIR bash $HERE/run_compass_rlvr_ablations_fsdp.sh 2>&1 | tee -a $WORK/runs/compass-$arm/boot.log; exec bash"
  echo "launched arm=$arm on GPU $gpu  (tmux window $win)"
  gpu=$(( gpu + 1 ))
  [[ $gpu -lt ${#ARM_LIST[@]} ]] && sleep "$STAGGER"
done

echo
echo "=== $WAVE launched: ${ARM_LIST[*]} ==="
echo "watch:   tmux attach -t $SESSION"
echo "logs:    tail -f $WORK/runs/compass-<arm>/train.log"
echo "health:  grep -c global_step $WORK/runs/compass-*/train.log"
echo
echo "Per-arm falsifiers, check these once each arm is past step 20:"
echo "  base/kNN : grep -m1 'stale-replay' train.log  must print the arm's delay_K"
echo "  noanchor : NO '[comm_eff][signed_ema] enabled' line at all, and"
echo "             actor/comm_eff/anchor_backwards stays 0 while"
echo "             actor/comm_eff/mask_applications/train keeps climbing"
echo "  nosign   : the signed_ema banner reads identity=true"
