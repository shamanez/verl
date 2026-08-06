#!/usr/bin/env bash
# run_qwen3_8b_densemid_1000.sh
#
# Run 97: a rerun of run 96 (Qwen3-8B-Base / MATH / 16384 total context /
# 1000 steps / batch 128 with mini-batch 128 / PRF exact-k p=0.95 constant
# rescale / anchor 20/20 / owns_q=false) with ONE scientific change: the two
# pipeline-boundary cuts inside the middle band are left UNCOMPRESSED.
#
# THE SCIENCE. Run 96 collapsed at roughly step 180. The spark was truncation
# feedback at the 15360 response cap, but the fuel was the train-inference
# mismatch: the dense sampler and the masked trainer sat 17.9 nats apart and
# that wedge compounded over 16k-token responses. This run keeps 8 stages, the
# codec, the anchor, and every other knob identical to run 96, and only zeroes
# the masked fraction on the two middle cuts, to test whether cutting the
# wedge there removes the collapse.
#
# WHY THOSE TWO CUTS. With 36 layers over 8 stages the boundary cuts sit after
# decoder layers [4, 9, 14, 19, 23, 27, 31]. The cuts after layers 14 and 19
# are at fractional depth 0.42 and 0.56, the only two of the seven inside the
# LayerCompass active band 0.39-0.57 (arXiv 2607.01232: single-layer RL
# confined to that band matches full GRPO). With them dense, no further
# masking is applied between layers 10 and 23 (the band's entry activation
# still crosses the compressed cut after layer 9). Note two dense cuts of
# seven cut the wedge mechanically wherever they sit (mean mask ratio falls
# 0.950 to 0.679 regardless of position), so a mismatch drop here cannot by
# itself be attributed to the middle band; the attribution control is a later
# arm with the dense pair moved off-band (cuts [4,9] or [27,31]) or a
# budget-matched uniform arm (p=0.679 on all seven).
#
# MECHANISM (already in the codebase, no core change). A 0.0 entry in
# mask.p_by_boundary with exact_k=true keeps round((1-0.0)*H) = H channels, so
# prf_token_mask early-returns an all-ones mask, and the constant-rescale gain
# is recomputed per boundary as 1/(1-0.0) = 1.0. The hook then computes
# h * ones * 1.0: a bit-exact identity on the forward AND the backward. The
# money gate in the target launcher re-proves this at H=4096 before any GPU is
# touched.
#
# WIRE ACCOUNTING (why this is still a comm-eff run). 5 of the 7 links carry
# 205 of 4096 coords/token = 3280 bits/token. The 2 dense links carry the full
# bf16 hidden, 4096 x 16 = 65536 bits/token. Aggregate across the seven cuts:
# 5*3280 + 2*65536 = 147472 bits/token vs run 96's 22960 (6.4x), and the
# saving vs dense-everywhere (458752) drops from 95.0 to 67.9 percent.
# Deployment story: because pipeline cuts exist here only where masking is
# applied, this run is EQUIVALENT to 6 stages with cuts [4,9,23,27,31] all
# compressed, i.e. the middle band lives inside one (bigger, ~3x weights)
# stage and those two links never cross the internet at all. The alternative
# reading, 8 stages with a fat pipe on two middle links, prices the same
# experiment differently but is the weaker story for community hardware.
#
# SAVE_FREQ=50. Run 96 died at step 189 with save_freq 200, so zero
# checkpoints ever existed. 50 caps the loss at 49 steps, and the R2 mirror
# deletes local files after a verified upload, so disk stays bounded. NOTE:
# because run 96 never saved, step 50 is the FIRST exercise of the FSDP
# full-state gather plus the ~98 GB R2 upload at 8B/16k scale. Watch step 50
# the way step 20 is watched for the first anchor fire: host RAM, GPU
# headroom, and the heartbeat log (a stalled upload that freezes train.log
# can trip the no-heartbeat reaper).
#
# This is a THIN wrapper over run_qwen3_8b_prf_exactk_1000.sh. Everything
# else, including the money gates, the MATH parquet prep, the comm-eff env
# block and the Hydra passthrough, is inherited. Run inside tmux.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE/run_qwen3_8b_prf_exactk_1000.sh"

[[ -f "$TARGET" ]] || {
  echo "FATAL: cannot find the launcher this wraps: $TARGET" >&2
  exit 1
}

# The deltas from run 96. Each is a plain default, so an explicit value from
# the caller's environment still wins.
export RUN_ID="${RUN_ID:-97-qwen3-8b-16k-densemid-1000}"
export BRANCH="${BRANCH:-exp/97-qwen3-8b-16k-densemid-1000}"

# Hydra list literal: quoted for the shell, NO spaces inside (the engine
# script forwards it verbatim as one Hydra override). Entries are index-aligned
# with the boundary set [4,9,14,19,23,27,31], so the 0.0 entries land on the
# cuts after layers 14 and 19.
export COMM_EFF_MASK_P_BY_BOUNDARY="${COMM_EFF_MASK_P_BY_BOUNDARY:-[0.95,0.95,0.0,0.0,0.95,0.95,0.95]}"

export SAVE_FREQ="${SAVE_FREQ:-50}"

# 4x H200 box gates. MIN_RAM_GIB is NOT the 4gpu wrapper's old 768 estimate:
# run 96 MEASURED an all-time host peak of 1081 GiB on this exact 4x H200
# recipe (1368 GiB container, 122 GiB intra-step swing, one-time +113 GiB
# anchor-ring growth at fire 2). A 768 gate admits boxes that host-OOM near
# the second anchor fire. The anchor GPU peak does not shard either, so watch
# step 20 (the first anchor fire).
export EXPECT_GPUS="${EXPECT_GPUS:-4}"
export MIN_RAM_GIB="${MIN_RAM_GIB:-1200}"

echo "=== run 97 dense-mid variant ==="
echo "    RUN_ID=$RUN_ID"
echo "    BRANCH=$BRANCH"
echo "    COMM_EFF_MASK_P_BY_BOUNDARY=$COMM_EFF_MASK_P_BY_BOUNDARY"
echo "    SAVE_FREQ=$SAVE_FREQ EXPECT_GPUS=$EXPECT_GPUS MIN_RAM_GIB=$MIN_RAM_GIB"
echo "=== delegating to $(basename "$TARGET") ==="

exec bash "$TARGET" "$@"
