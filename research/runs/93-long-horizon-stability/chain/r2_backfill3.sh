#!/usr/bin/env bash
# Issue #93: R2 checkpoint back-fill, v2 after `aws s3 sync` failed.
#
# WHY v1 FAILED. `aws s3 sync` returned InvalidPart on CompleteMultipartUpload for
# exactly the four large files per cell (11.5G optimizer, 6.62G model); every small
# file landed. At aws-cli's default 8MB chunk an 11.5G file is ~1437 parts, and R2
# rejects that many parts uploaded with the default concurrency of 10. Fixed by
# `aws configure`: max_concurrent_requests 1, multipart_chunksize 256MB
# (11.5G -> ~46 parts), multipart_threshold 256MB.
#
# v2 also drops `sync` for per-file `aws s3 cp`, which is the path r2_sink.py uses
# and which #90 already proved on this box.
#
# v3 (2026-07-26T18:30Z), after an 11.5G optimizer upload HUNG for 78 minutes:
#   * `aws s3 cp` has no internal timeout. The transfer finished (all parts on R2)
#     and the process then sat on CompleteMultipartUpload with frozen CPU. Every
#     upload is now wrapped in `timeout`, so a hang costs minutes not hours.
#   * A part can SILENTLY go missing. That upload had 46 parts where the file needs
#     47: 45 full 256MB parts plus the 1.99MB remainder, with one full middle part
#     absent. That, not part count alone, is the real cause of the original
#     InvalidPart ("one or more of the specified parts could not be found").
#   * NEVER complete a multipart by hand. Doing so on that 46-part upload produced
#     an object of 12,081,588,677 bytes for a 12,350,024,133-byte file: a corrupt
#     checkpoint short by exactly one 256MB chunk. The size check caught it and it
#     was deleted, but the lesson is that the manifest must be validated against the
#     file BEFORE completing, and there is no reason to hand-complete at all when
#     re-running `cp` is correct by construction.
#   * Stale multiparts are aborted per key before each attempt, so a retry cannot
#     inherit a gapped part set, and R2 stops billing for the abandoned parts.
#
# ORDERING IS DELIBERATE: model + config + tokenizer BEFORE the optimizer state.
# The model files are what post-hoc geometry and OOD eval need; the optimizer
# state is only needed to RESUME training. If anything fails part-way, the useful
# half is already safe.
set -uo pipefail

LOG=/workspace/r2-backfill2.log
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

set -a
# shellcheck disable=SC1091
source "$HOME/.config/verl-research/secrets.env"
set +a
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto

BUCKET="$R2_BUCKET"
EP="$R2_ENDPOINT"
ROOT=/workspace/verl/checkpoints/93-long-horizon-stability
PREFIX_BASE=autonomous-harness-rlvr-compression/93-long-horizon-stability
DELETE_AFTER_VERIFY="${DELETE_AFTER_VERIFY:-yes}"

say "backfill3 start: bucket=$BUCKET concurrency=$(aws configure get default.s3.max_concurrent_requests) chunk=$(aws configure get default.s3.multipart_chunksize)"
say "disk before: $(df -h /workspace | tail -1 | tr -s ' ')"

remote_size() {  # $1 = full key; prints bytes or empty
  timeout 120 aws s3api head-object --bucket "$BUCKET" --key "$1" --endpoint-url "$EP" \
    --query ContentLength --output text 2>/dev/null | grep -E '^[0-9]+$' || true
}

abort_stale_multiparts() {  # $1 = full key
  local key uid
  while read -r key uid; do
    [[ -z "${uid:-}" || "$uid" == "None" ]] && continue
    [[ "$key" != "$1" ]] && continue
    timeout 120 aws s3api abort-multipart-upload --bucket "$BUCKET" --key "$key" \
      --upload-id "$uid" --endpoint-url "$EP" >/dev/null 2>&1 || true
  done < <(timeout 120 aws s3api list-multipart-uploads --bucket "$BUCKET" --prefix "$1" \
      --endpoint-url "$EP" --query "Uploads[].[Key,UploadId]" --output text 2>/dev/null)
}

for cell in "$@"; do
  src="$ROOT/$cell"
  if [[ ! -d "$src" ]]; then say "$cell: NO local dir, skipping"; continue; fi
  say "$cell: starting ($(du -sh "$src" | cut -f1))"

  # Model/config first, optimizer state last.
  mapfile -t files < <( { find "$src" -type f ! -name 'optim_*'; find "$src" -type f -name 'optim_*'; } )
  ok=0; bad=0
  for f in "${files[@]}"; do
    rel="${f#"$src"/}"
    key="$PREFIX_BASE/$cell/checkpoints/$rel"
    lsz=$(stat -c %s "$f")
    rsz=$(remote_size "$key")
    if [[ "$rsz" == "$lsz" ]]; then ok=$((ok+1)); continue; fi

    done_one=no
    for attempt in 1 2 3; do
      # A retry must not inherit a gapped part set from a killed attempt.
      abort_stale_multiparts "$key"
      # UPLOAD_TIMEOUT bounds the hang: 11.5G at ~8 MB/s is ~24 min, so 45 is ample.
      if timeout "${UPLOAD_TIMEOUT:-2700}" aws s3 cp "$f" "s3://$BUCKET/$key" \
           --endpoint-url "$EP" --only-show-errors 2>>"$LOG"; then
        rsz=$(remote_size "$key")
        if [[ "$rsz" == "$lsz" ]]; then done_one=yes; break; fi
        say "$cell: $rel uploaded but size mismatch (local $lsz remote ${rsz:-none}), attempt $attempt"
      else
        rc=$?
        if [[ "$rc" == "124" ]]; then
          say "$cell: $rel attempt $attempt TIMED OUT after ${UPLOAD_TIMEOUT:-2700}s (the 78-minute hang signature)"
        else
          say "$cell: $rel upload attempt $attempt failed rc=$rc"
        fi
      fi
      sleep 15
    done
    if [[ "$done_one" == yes ]]; then
      ok=$((ok+1))
      # Log only the big ones; the small files would drown the log.
      (( lsz > 1073741824 )) && say "$cell: $rel OK ($(( lsz / 1073741824 )) GB)"
    else
      bad=$((bad+1)); say "$cell: $rel FAILED after 3 attempts"
    fi
  done

  say "$cell: verified $ok, failed $bad, of ${#files[@]} files"
  if (( bad == 0 )); then
    say "$cell: COMPLETE in R2"
    if [[ "$DELETE_AFTER_VERIFY" == "yes" ]]; then
      rm -rf "$src"
      say "$cell: local copy removed; disk now $(df -h /workspace | tail -1 | tr -s ' ')"
    else
      say "$cell: DELETE_AFTER_VERIFY=no, local copy kept"
    fi
  else
    say "$cell: INCOMPLETE, local copy KEPT"
  fi
done

say "backfill3 done. disk after: $(df -h /workspace | tail -1 | tr -s ' ')"
