#!/usr/bin/env bash
# Issue #93: back-fill the LOCAL-ONLY checkpoints to R2.
#
# a5b, a6, a7 (and a8 when it lands) ran with the R2 sink OFF, so their
# checkpoints exist only on this box and die with it. This is step 1 of the
# pre-teardown list, done early for two reasons: it stops depending on a
# teardown-time race, and it frees disk. Each cell is 37G, the disk is 200G with
# 114G already used, and three more cell-saves are queued (a8, a9, a10) which
# would overflow it.
#
# Deletion is gated on a BYTE-EXACT per-cell verification against the remote
# listing. A cell whose verification does not match is left on disk untouched.
set -uo pipefail

LOG=/workspace/r2-backfill.log
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a
# aws reads the generic names; R2 creds are never echoed.
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto

BUCKET="$R2_BUCKET"
EP="$R2_ENDPOINT"
ROOT=/workspace/verl/checkpoints/93-long-horizon-stability
PREFIX_BASE=autonomous-harness-rlvr-compression/93-long-horizon-stability

say "backfill start: bucket=$BUCKET root=$ROOT"
say "disk before: $(df -h /workspace | tail -1 | tr -s ' ')"

DELETE_AFTER_VERIFY="${DELETE_AFTER_VERIFY:-yes}"

for cell in "$@"; do
  src="$ROOT/$cell"
  if [[ ! -d "$src" ]]; then say "$cell: NO local dir, skipping"; continue; fi
  dst="s3://$BUCKET/$PREFIX_BASE/$cell/checkpoints"
  say "$cell: syncing $(du -sh "$src" | cut -f1) -> $PREFIX_BASE/$cell/checkpoints"

  if ! aws s3 sync "$src" "$dst" --endpoint-url "$EP" --only-show-errors; then
    say "$cell: SYNC FAILED, leaving local copy in place"
    continue
  fi

  # Byte-exact verification: same file count and same total size, computed from
  # the remote listing rather than trusting sync's exit code alone.
  lcount=$(find "$src" -type f | wc -l | tr -d ' ')
  lbytes=$(find "$src" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  rlist=$(aws s3 ls "$dst/" --recursive --endpoint-url "$EP" 2>/dev/null)
  rcount=$(printf '%s\n' "$rlist" | grep -c . || true)
  rbytes=$(printf '%s\n' "$rlist" | awk '{s+=$3} END {print s+0}')
  say "$cell: local files=$lcount bytes=$lbytes | remote files=$rcount bytes=$rbytes"

  if [[ "$lcount" == "$rcount" && "$lbytes" == "$rbytes" ]]; then
    say "$cell: VERIFIED byte-exact in R2"
    if [[ "$DELETE_AFTER_VERIFY" == "yes" ]]; then
      rm -rf "$src"
      say "$cell: local copy removed; disk now $(df -h /workspace | tail -1 | tr -s ' ')"
    else
      say "$cell: DELETE_AFTER_VERIFY=no, local copy kept"
    fi
  else
    say "$cell: MISMATCH, local copy KEPT and NOT deleted"
  fi
done

say "backfill done. disk after: $(df -h /workspace | tail -1 | tr -s ' ')"
