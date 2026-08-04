#!/usr/bin/env bash
# run_qwen3_8b_prf_exactk_1000_4gpu.sh
#
# The 4x H200 variant of run_qwen3_8b_prf_exactk_1000.sh. Same experiment, same
# science, same config: Qwen3-8B-Base / MATH / PRF exact-k at p=0.95 / 16384
# total context / 1000 steps / batch 128 with mini-batch 128.
#
# This is a THIN wrapper, deliberately. Every knob it sets is an env override the
# 8-GPU launcher already reads, so the two paths cannot silently diverge on
# anything but the three box gates below.
#
# WHY 4 GPUs IS THE RISKIER OF THE TWO, and worth testing separately:
#
#   The anchor's memory does not shard. Its replay clone is a full unsharded copy
#   on every rank, and its forward/backward packs `max_token_len_per_gpu` tokens
#   PER GPU regardless of world size (transformer_impl.py reuses the actor's
#   budget for the anchor pass). So halving the GPU count does not halve the
#   anchor peak. It only doubles the FSDP-sharded static state, from 15.3 to
#   32.8 GiB per rank.
#
#   Estimated per-rank peak at the first anchor fire, with the gradient
#   checkpointing fix in place:
#       8 ranks:  ~92 GiB against 131.3 GiB usable   (~30 percent headroom)
#       4 ranks: ~110 GiB against 131.3 GiB usable   (~15 percent headroom)
#
#   Both should fit. Only one of them has been measured, which is neither. Run
#   the 8-GPU path if you want the safer bet, this one if you want the cheaper
#   box, and watch step 20 either way because that is the first anchor fire and
#   the only step where the peak appears.
#
# Host RAM moves the OTHER way. Comm-eff state is replicated per rank, not
# sharded, so FEWER ranks need LESS host RAM: about 610 GiB here against about
# 976 GiB at 8 ranks. That is why MIN_RAM_GIB drops to 768.
#
# Cost, at the prices seen when this was written: 4x H200 lists near $10.53/hr
# against $31.06/hr for 8x. The 8-GPU box is roughly 1.9x faster in wall clock
# but 2.95x dearer per hour, so this variant is the cheaper of the two overall
# (about $285 against $519 in the good case) and the slower (about 27 h against
# 17 h).
#
# Run inside tmux. Everything else, including the money gates, the MATH parquet
# prep, the comm-eff env block and the Hydra passthrough, is inherited.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE/run_qwen3_8b_prf_exactk_1000.sh"

[[ -f "$TARGET" ]] || {
  echo "FATAL: cannot find the 8-GPU launcher this wraps: $TARGET" >&2
  exit 1
}

# The only three deltas. Each is a plain default, so an explicit value from the
# caller's environment still wins.
export EXPECT_GPUS="${EXPECT_GPUS:-4}"
export MIN_RAM_GIB="${MIN_RAM_GIB:-768}"    # 4 ranks replicate less comm-eff state than 8
export MIN_DISK_GIB="${MIN_DISK_GIB:-700}"  # unchanged, 5 checkpoints at ~98 GB each

echo "=== 4x H200 variant: EXPECT_GPUS=$EXPECT_GPUS MIN_RAM_GIB=$MIN_RAM_GIB MIN_DISK_GIB=$MIN_DISK_GIB ==="
echo "=== delegating to $(basename "$TARGET") ==="

exec bash "$TARGET" "$@"
