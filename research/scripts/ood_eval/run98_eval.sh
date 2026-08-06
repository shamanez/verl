#!/usr/bin/env bash
# run98_eval.sh
#
# In-domain + OOD capability eval for run 98 (98-qwen3-8b-16k-best3-600):
# Qwen3-8B-Base, 1024 prompt + 15360 response, checkpoints every 50 steps.
#
# Three models, one table:
#   base      the untrained Qwen3-8B-Base anchor
#   step150   global_step_150 (also the step where training runs its own
#             in-domain val, so our in-domain number there is a CROSS-CHECK)
#   step200   global_step_200 (in-domain, we are the only source)
#
# THIS SCRIPT IS WRITTEN TO RUN ON A DIFFERENT BOX THAN THE TRAINER. It never
# assumes a local checkpoint tree: every checkpoint is pulled from R2, merged to
# a clean HF model, and the pulled FSDP shards are deleted as soon as the merge
# succeeds. The trainer box is saturated for the length of the run, so the eval
# box is a separate rental (or the same box after training ends).
#
# WHAT IT REUSES, AND WHAT IT DOES NOT
#   ood_prep.py     builds every benchmark parquet. Never reimplemented here.
#   ood_eval.sh     runs ONE model on ONE benchmark. Called directly, per pair.
#   ood_run_all.sh  its BENCHES table (bench name + sampling protocol) is PARSED
#                   out of the file so the sampling protocol has exactly one
#                   definition in the repo. Its merge() and ROSTER are hardcoded
#                   to the 1.5B dose-response matrix (run/checkpoints/ key layout,
#                   no disk discipline, no preflight), so the orchestration is
#                   re-done here rather than bent into that shape.
#
# R2 LAYOUT IS DISCOVERED, NOT ASSUMED. secrets.env on these boxes ships
# R2_BUCKET set to the PREFIX string by mistake, so R2_BUCKET is deliberately
# ignored and R2_CKPT_BUCKET is pinned below. The key layout under the prefix is
# read from a real listing (the sink writes .../<exp>/<regime>/checkpoints/
# global_step_N/actor/ in some runs and .../<exp>/global_step_N/actor/ in
# others), so both shapes work and a wrong guess cannot silently produce an
# empty download.
#
# Knobs (env):
#   EVAL_STEPS      steps to evaluate           (default "150 200")
#   VERL_DIR        verl checkout               (default /workspace/verl)
#   OOD_EVAL_ROOT   eval output root            (default /workspace/runs/run98-eval)
#   OOD_DATA_ROOT   benchmark parquets          (default /root/data/ood)
#   MATH_DATA_DIR   in-domain MATH parquet      (default $HOME/data/math)
#   CKPT_ROOT       where pulled shards land    (default $OOD_EVAL_ROOT/pulled)
#   BASE_MODEL      untrained anchor            (default Qwen/Qwen3-8B-Base)
#   PAIRS_CSV       GPU-pair pool               (default "0,1|2,3", a 4-GPU box)
#                   Two pairs means two full Ray + vLLM + FSDP stacks for an 8B
#                   model at once. On a RAM-tight box set PAIRS_CSV="0,1" and
#                   take the serial path instead of discovering the ceiling
#                   halfway through the suite.
#   R2_PREFIX       key prefix under the bucket (default run 98's)
#   R2_CKPT_BUCKET  bucket                      (pinned to shamane-pluralis)
#   R2_MIN_MB_S     slowest download rate to size the aws timeout (default 15)
#   MAX_RESPONSE_LENGTH  generation cap         (default 15360, matches training)
#   MIN_GPU_GIB     hard floor per GPU          (default 40; warn below 70)
#   MERGE_RAM_GIB   host RAM the merge needs    (default 100)
#   MERGE_MIN_GIB   smallest believable merged model (default 12)
#   DRY_RUN=1       run every preflight gate and stop before the first download
#
# Credentials come from ~/.config/verl-research/secrets.env (off-repo, chmod 600).
# The engine this driver ends up calling reads the SAME file and hard-requires
# HF_TOKEN and WANDB_API_KEY, and refuses to start if VAST_API_KEY is present, so
# all three are checked here rather than 33 times at the far end of a download.
# Nothing secret is printed or written by this script.
#
# Run inside tmux. Every log is appended, never truncated, so an interrupted run
# resumes into the same files.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID="98-qwen3-8b-16k-best3-600"
EVAL_STEPS="${EVAL_STEPS:-150 200}"

VERL_DIR="${VERL_DIR:-/workspace/verl}"
OOD_EVAL_ROOT="${OOD_EVAL_ROOT:-/workspace/runs/run98-eval}"
OOD_DATA_ROOT="${OOD_DATA_ROOT:-/root/data/ood}"
MATH_DATA_DIR="${MATH_DATA_DIR:-$HOME/data/math}"
CKPT_ROOT="${CKPT_ROOT:-$OOD_EVAL_ROOT/pulled}"
MERGED_ROOT="${MERGED_ROOT:-$OOD_EVAL_ROOT/merged}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
PAIRS_CSV="${PAIRS_CSV:-0,1|2,3}"

# R2. R2_BUCKET is NOT read: on these boxes it holds the prefix, not a bucket.
R2_CKPT_BUCKET="shamane-pluralis"
R2_PREFIX="${R2_PREFIX:-autonomous-harness-rlvr-compression/$RUN_ID}"
R2_MIN_MB_S="${R2_MIN_MB_S:-15}"
[[ "$R2_MIN_MB_S" =~ ^[0-9]+$ ]] && (( R2_MIN_MB_S > 0 )) || {
  echo "FATAL: R2_MIN_MB_S must be a positive integer (it divides the download timeout)" >&2; exit 1; }
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"

# Generation surface. Identical to training so the in-domain number at step 150
# is comparable to the in-training val at step 150 rather than merely similar.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"

# Size model for the disk gate, printed with its derivation. Only the per-pull
# number is measured from R2. These four are the 8B constants, and are
# env-overridable so a different model size does not need a code edit.
# The 91 GiB is fp32 weights (~30) plus the two fp32 AdamW moments (~60): verl
# keeps the FSDP actor master copy in fp32. The merger casts every shard to
# bf16, so the MERGED model is half the weight size, not half of 91.
CKPT_GIB_NOMINAL="${CKPT_GIB_NOMINAL:-91}"   # full actor dir of one 8B FSDP checkpoint in R2
MERGE_GIB="${MERGE_GIB:-16}"                 # one merged bf16 HF model
MERGE_MIN_GIB="${MERGE_MIN_GIB:-12}"         # a merge holding less than this is truncated, not finished
BASE_CACHE_GIB="${BASE_CACHE_GIB:-16}"       # the HF download of the untrained anchor
DISK_SLACK_GIB="${DISK_SLACK_GIB:-20}"       # logs, vLLM scratch, tokenizer caches
# Host RAM for the merge. It loads every fp32 rank shard (~30 GiB), holds the
# bf16 copies while it concatenates, then materialises an empty bf16 model on
# CPU to save through. Measured shape is roughly 60-70 GiB resident for 8B.
MERGE_RAM_GIB="${MERGE_RAM_GIB:-100}"
MIN_GPU_GIB="${MIN_GPU_GIB:-40}"             # hard floor per GPU
WARN_GPU_GIB="${WARN_GPU_GIB:-70}"           # below this the KV cache gets uncomfortable at 16k

IN_DOMAIN_BENCH="math_indomain"
BASE_INDOMAIN_REF="0.7214"   # measured in-training at step 0 of this run

mkdir -p "$OOD_EVAL_ROOT" "$MERGED_ROOT" "$CKPT_ROOT" || {
  echo "FATAL: cannot create $OOD_EVAL_ROOT" >&2; exit 1; }
DRIVER_LOG="$OOD_EVAL_ROOT/run98_eval.log"

say() { echo "$*" | tee -a "$DRIVER_LOG"; }
die() { echo "FATAL: $*" | tee -a "$DRIVER_LOG" >&2; exit 1; }

file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }

# One driver at a time per eval root. A second copy would put a second pair of
# vLLM stacks on the same GPUs and would rewrite train.log files the first copy
# is still filling, which reads afterwards as a mysteriously failed benchmark.
# The lock is an atomic mkdir plus a `kill -0` liveness test on the recorded
# pid: NEVER pgrep/pkill on a pattern, which in this project has repeatedly
# matched the checking command itself (and once killed the caller).
LOCK_DIR="$OOD_EVAL_ROOT/.run98_eval.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  OTHER="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$OTHER" =~ ^[0-9]+$ ]] && kill -0 "$OTHER" 2>/dev/null; then
    echo "FATAL: another run98_eval.sh (pid $OTHER) is already driving $OOD_EVAL_ROOT." >&2
    echo "       Attach to its tmux window, or wait for it. Delete $LOCK_DIR only if you are sure it is dead." >&2
    exit 1
  fi
  echo "WARN: clearing a stale lock at $LOCK_DIR (recorded pid '${OTHER:-none}' is not running)" >&2
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" || { echo "FATAL: could not take the lock at $LOCK_DIR" >&2; exit 1; }
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

say ""
say "=== run 98 eval driver  $(date -Iseconds) ==="
say "    run            $RUN_ID"
say "    steps          $EVAL_STEPS"
say "    base model     $BASE_MODEL"
say "    context        $MAX_PROMPT_LENGTH prompt + $MAX_RESPONSE_LENGTH response = $MAX_MODEL_LEN"
say "    eval root      $OOD_EVAL_ROOT"
say "    r2             s3://$R2_CKPT_BUCKET/$R2_PREFIX/"

# ===========================================================================
# 1. Local preflight. Nothing here touches the network.
# ===========================================================================
say ""
say "--- preflight: local"

[[ -d "$VERL_DIR/.git" || -d "$VERL_DIR/verl" ]] || die "no verl checkout at VERL_DIR=$VERL_DIR"
for f in ood_eval.sh ood_prep.py ood_run_all.sh; do
  [[ -f "$SCRIPT_DIR/$f" ]] || die "missing sibling harness file $SCRIPT_DIR/$f"
done
BASE_LAUNCHER="$VERL_DIR/examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
[[ -f "$BASE_LAUNCHER" ]] || die "base launcher not found: $BASE_LAUNCHER"
[[ -f "$VERL_DIR/examples/grpo_trainer/relex_qwen_chat_template.jinja" ]] \
  || die "the base launcher's RELEX chat template is missing from $VERL_DIR/examples/grpo_trainer/. The eval prompt would not match training."
command -v python3 >/dev/null 2>&1 || die "python3 is not on PATH"
( cd "$VERL_DIR" && python3 -c "import verl" ) >/dev/null 2>&1 \
  || die "verl is not importable from $VERL_DIR (install it: pip install --no-deps -e .)"

# The four bare scalars this driver has to patch into a copy of the base
# launcher. The SHAPE CHECK lives here, not next to the sed, so DRY_RUN=1
# actually exercises the one thing most likely to have drifted: the base
# launcher's own defaults. Section 6 re-uses this table verbatim.
PATCH_SPEC=(
  "MAX_PROMPT_LENGTH|1024|$MAX_PROMPT_LENGTH"
  "MAX_RESPONSE_LENGTH|3072|$MAX_RESPONSE_LENGTH"
  "TRAIN_BATCH_SIZE|512|$TRAIN_BATCH_SIZE"
  "PPO_MINI_BATCH_SIZE|256|$PPO_MINI_BATCH_SIZE"
)
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom _pto <<< "$spec"
  grep -q "^export ${pname}=${pfrom}\$" "$BASE_LAUNCHER" \
    || die "base launcher shape drifted: expected 'export ${pname}=${pfrom}' in $BASE_LAUNCHER. Re-derive this patch table before running."
done
say "    patch table    4/4 bare scalars found in the base launcher"

# GPUs. The real count, before any shim is put on PATH.
NVIDIA_SMI_REAL="$(command -v nvidia-smi 2>/dev/null || true)"
[[ -n "$NVIDIA_SMI_REAL" ]] || die "nvidia-smi is not on PATH; this driver needs GPUs"
GPU_COUNT="$("$NVIDIA_SMI_REAL" -L 2>/dev/null | wc -l | tr -d ' ')"
say "    GPUs detected  ${GPU_COUNT:-0}"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= 2 )) \
  || die "need at least 2 GPUs (one pair); detected ${GPU_COUNT:-0}"

IFS='|' read -r -a PAIRS <<< "$PAIRS_CSV"
(( ${#PAIRS[@]} >= 1 )) || die "PAIRS_CSV=$PAIRS_CSV parsed to zero pairs"
for pair in "${PAIRS[@]}"; do
  [[ "$pair" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "PAIRS_CSV entry '$pair' is not a comma-separated GPU list"
  IFS=',' read -r -a _ids <<< "$pair"
  for gid in "${_ids[@]}"; do
    (( gid < GPU_COUNT )) || die "PAIRS_CSV references GPU $gid but only $GPU_COUNT GPUs exist"
  done
done
say "    GPU pairs      ${#PAIRS[@]} (${PAIRS[*]})"

# Per-GPU memory. ROLLOUT_TP is a BARE `export ROLLOUT_TP=1` in the base
# launcher, exactly as run 98's training launcher left it, so each GPU of a pair
# runs its OWN complete vLLM replica: ~16 GiB of bf16 8B weights per GPU, inside
# a budget of ROLLOUT_GPU_MEM_UTIL x total, with the actor's FSDP shard (~8 GiB
# per GPU of a 2-GPU pair, fp32 master weights) already resident when vLLM
# profiles. At 16384 context the remainder is the KV cache and it is what sets
# how many sequences run at once.
GPU_MIN_MIB="$("$NVIDIA_SMI_REAL" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                | tr -d ' ' | sort -n | head -1)"
if [[ "$GPU_MIN_MIB" =~ ^[0-9]+$ ]]; then
  GPU_MIN_TOTAL_GIB=$(( GPU_MIN_MIB / 1024 ))
  say "    GPU memory     ${GPU_MIN_TOTAL_GIB} GiB on the smallest GPU (vLLM budget ${ROLLOUT_GPU_MEM_UTIL} of it)"
  (( GPU_MIN_TOTAL_GIB >= MIN_GPU_GIB )) \
    || die "an 8B model at 16384 context needs more than ${MIN_GPU_GIB} GiB per GPU; the smallest here is ${GPU_MIN_TOTAL_GIB} GiB. Override MIN_GPU_GIB only if you have measured it."
  (( GPU_MIN_TOTAL_GIB >= WARN_GPU_GIB )) \
    || say "    WARN: below ${WARN_GPU_GIB} GiB/GPU the KV cache left after the 16 GiB of weights is small, so the long-response benchmarks will run at low concurrency and may take many hours. Consider ROLLOUT_GPU_MEM_UTIL=0.85 and PAIRS_CSV with a single pair."
else
  say "    WARN: could not read GPU memory from nvidia-smi; skipping the memory gate"
fi

AWS_BIN="$(command -v aws 2>/dev/null || true)"
[[ -n "$AWS_BIN" ]] || die "'aws' is not on PATH. Install the awscli v2 self-contained zip (it bundles its own python and cannot perturb the pinned torch/vllm stack), then re-run."
say "    aws            $AWS_BIN"

TIMEOUT_BIN="$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)"
[[ -n "$TIMEOUT_BIN" ]] || say "    WARN: no 'timeout' binary; aws downloads will run unbounded (a stalled multipart can hang for hours)"

# ===========================================================================
# 2. Benchmark table. Parsed out of ood_run_all.sh so the sampling protocol
#    (which sets are greedy mean@1 and which are avg@8 at temp 0.7 / top_p 0.8)
#    has exactly one definition in this repo. A parse failure is fatal: copying
#    the table here would be the duplication this is avoiding.
# ===========================================================================
BENCH_TABLE="$(awk '/^BENCHES=\(/{f=1;next} f&&/^\)/{exit} f' "$SCRIPT_DIR/ood_run_all.sh" \
  | sed -e 's/#.*$//' -e 's/^[[:space:]]*"//' -e 's/"[[:space:]]*$//' -e 's/[[:space:]]*$//' \
  | grep -v '^[[:space:]]*$')"
[[ -n "$BENCH_TABLE" ]] || die "could not parse the BENCHES table out of $SCRIPT_DIR/ood_run_all.sh"

BENCH_SPECS=()
BENCH_NAMES=()
while IFS= read -r line; do
  read -r _b _n _t _p _extra <<<"$line"
  [[ -n "$_b" && -n "$_n" && -n "$_t" && -n "$_p" && -z "${_extra:-}" ]] \
    || die "BENCHES row '$line' in ood_run_all.sh does not have exactly 4 fields (name n temp top_p)"
  [[ "$_n" =~ ^[0-9]+$ ]] || die "BENCHES row '$line' has a non-integer n"
  BENCH_SPECS+=("$_b $_n $_t $_p")
  BENCH_NAMES+=("$_b")
done <<< "$BENCH_TABLE"
(( ${#BENCH_SPECS[@]} >= 8 )) || die "parsed only ${#BENCH_SPECS[@]} benchmarks from ood_run_all.sh, expected the full suite"

# In-domain goes FIRST and is not part of that table: it is the training val set
# itself (the MATH test split), scored with verl's own validation sampling
# defaults (n=1, temperature 0, top_p 1.0), which is what makes the step-150
# number a cross-check of the in-training val rather than a second opinion.
ALL_SPECS=("$IN_DOMAIN_BENCH 1 0 1.0" "${BENCH_SPECS[@]}")
ALL_NAMES=("$IN_DOMAIN_BENCH" "${BENCH_NAMES[@]}")
say "    benchmarks     ${#ALL_SPECS[@]} (${ALL_NAMES[*]})"

# Model roster. The unquoted expansion of EVAL_STEPS is the intended word split
# (this file is bash, not the zsh the operator's shell runs, where it would not
# split and the whole string would become one bogus step). A tag is literally
# "step" plus its number, so the step is recovered as ${tag#step} and no
# associative array is needed.
TAGS=("base")
# shellcheck disable=SC2086
for s in $EVAL_STEPS; do
  [[ "$s" =~ ^[0-9]+$ ]] || die "EVAL_STEPS entry '$s' is not an integer"
  TAGS+=("step$s")
done
(( ${#TAGS[@]} >= 2 )) || die "EVAL_STEPS is empty"
say "    models         ${TAGS[*]}"

# ===========================================================================
# 3. R2 preflight: credentials, listing, key layout, step existence.
#    Everything here happens BEFORE a single byte is downloaded.
# ===========================================================================
say ""
say "--- preflight: R2"

[[ -f "$SECRETS_FILE" ]] || die "secrets file not found: $SECRETS_FILE (needs R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT)"
# shellcheck disable=SC1090
{ set -a; source "$SECRETS_FILE"; set +a; } || die "could not source $SECRETS_FILE"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing from $SECRETS_FILE}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing from $SECRETS_FILE}"
: "${R2_ENDPOINT:?R2_ENDPOINT missing from $SECRETS_FILE}"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

# The same secrets file is what the ENGINE reads, and the engine's own gate is
# stricter than ours: vast_comm_eff_engine_grpo.sh hard-fails on a missing
# HF_TOKEN or WANDB_API_KEY and hard-fails again if VAST_API_KEY is present.
# Every one of the eval jobs goes through that gate. Check it here, before the
# download, so a stripped-wrong secrets file costs a second instead of a 30 GiB
# pull followed by 33 jobs that each die in under a second.
: "${HF_TOKEN:?HF_TOKEN missing from $SECRETS_FILE. The engine requires it, and Qwen3-8B-Base is pulled from the hub for the base column.}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing from $SECRETS_FILE. The engine requires the variable to exist even when WANDB_MODE=offline.}"
[[ -z "${VAST_API_KEY:-}" ]] \
  || die "VAST_API_KEY is set (probably a verbatim copy of the laptop secrets file). The engine refuses to start with it present. Re-strip $SECRETS_FILE on the laptop."
say "    engine creds   HF_TOKEN + WANDB_API_KEY present, no VAST_API_KEY"

# R2_BUCKET from secrets.env is a prefix string on these boxes. Pinned above.
say "    bucket         $R2_CKPT_BUCKET (pinned; R2_BUCKET from secrets.env is deliberately ignored)"

# 256MB parts. At the 8MB default an 11.5G shard is ~1437 parts and R2 rejects
# that many, and a silently-missing part is the failure mode that has cost this
# project hours. Same setting on the read side keeps ranged GETs sane.
if [[ "${AWS_CONFIGURE_CHUNKS:-1}" == "1" ]]; then
  "$AWS_BIN" configure set default.s3.multipart_chunksize 256MB 2>/dev/null || true
  "$AWS_BIN" configure set default.s3.multipart_threshold 256MB 2>/dev/null || true
fi

R2_LISTING="$OOD_EVAL_ROOT/r2_listing.txt"
say "    listing        s3://$R2_CKPT_BUCKET/$R2_PREFIX/ ..."
if ! "$AWS_BIN" s3api list-objects-v2 \
      --bucket "$R2_CKPT_BUCKET" --prefix "$R2_PREFIX/" \
      --endpoint-url "$R2_ENDPOINT" \
      --query 'Contents[].[Key,Size]' --output text > "$R2_LISTING" 2>"$OOD_EVAL_ROOT/r2_listing.err"; then
  say "    listing FAILED. aws said:"
  sed -n 1,20p "$OOD_EVAL_ROOT/r2_listing.err" | tee -a "$DRIVER_LOG"
  die "cannot list s3://$R2_CKPT_BUCKET/$R2_PREFIX/ (credentials, endpoint, or bucket wrong)"
fi
[[ -s "$R2_LISTING" ]] && ! grep -qx "None" "$R2_LISTING" \
  || die "s3://$R2_CKPT_BUCKET/$R2_PREFIX/ listed ZERO objects. Wrong prefix, or the run has not saved yet."

MANIFEST_DIR="$OOD_EVAL_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"
R2_PLAN="$OOD_EVAL_ROOT/r2_plan.tsv"
R2_LISTING="$R2_LISTING" MANIFEST_DIR="$MANIFEST_DIR" EVAL_STEPS="$EVAL_STEPS" \
python3 - > "$R2_PLAN" <<'PY'
"""Derive, per requested step, the real actor/ key prefix and what to download.

Reads the listing rather than hardcoding a layout, so both the
<exp>/<regime>/checkpoints/global_step_N/actor/ shape and the flatter
<exp>/global_step_N/actor/ shape resolve. Writes one download manifest per step
(relative-path TAB size, optimizer shards excluded) plus a plan line for bash.
"""

import os
import re
import sys

listing = os.environ["R2_LISTING"]
mdir = os.environ["MANIFEST_DIR"]
steps = os.environ["EVAL_STEPS"].split()

pat = re.compile(r"^(?P<pre>.*?/global_step_(?P<step>\d+)/actor/)(?P<rest>.+)$")
shard_pat = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")
found = {}
for raw in open(listing, encoding="utf-8"):
    line = raw.rstrip("\n")
    if not line or line == "None":
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        parts = line.rsplit(None, 1)
    if len(parts) != 2:
        continue
    key, size_s = parts[0], parts[1]
    try:
        size = int(size_s)
    except ValueError:
        continue
    m = pat.match(key)
    if not m:
        continue
    st = str(int(m.group("step")))
    found.setdefault(st, {}).setdefault(m.group("pre"), []).append((m.group("rest"), size))

print("AVAILABLE\t" + " ".join(sorted(found, key=int)))

bad = 0
for st in steps:
    st = str(int(st))
    if st not in found:
        print(f"MISSING\t{st}")
        bad += 1
        continue
    prefixes = sorted(found[st])
    if len(prefixes) > 1:
        print("AMBIGUOUS\t" + st + "\t" + " ".join(prefixes))
        bad += 1
        continue
    pre = prefixes[0]
    objs = found[st][pre]
    total = sum(sz for _, sz in objs)
    keep = [(r, sz) for r, sz in objs if not os.path.basename(r).startswith("optim")]
    dl = sum(sz for _, sz in keep)

    # SHARD COMPLETENESS. "at least one model_*.pt" is not enough: this run
    # mirrors to R2 WHILE it trains, so a step can be listed with only some of
    # its rank shards uploaded. The merger reads world_size from
    # fsdp_config.json and then opens model_world_size_<W>_rank_<i>.pt for every
    # i, so a half-uploaded step passes a naive check, costs a 30 GiB download,
    # and dies in the merge. Recover W from the shard NAMES (which is exactly
    # the set of files the merger will open), require one W and all of 0..W-1.
    # Note W is 4 for this run's 4-GPU box; the 1.5B reference runs were 8, and
    # nothing here assumes either.
    shard_ranks = {}
    for r, _sz in keep:
        m2 = shard_pat.match(r)
        if m2:
            shard_ranks.setdefault(int(m2.group(1)), set()).add(int(m2.group(2)))
    ws = 0
    shards_ok = 0
    if len(shard_ranks) == 1:
        ws = next(iter(shard_ranks))
        got = shard_ranks[ws]
        shards_ok = int(got == set(range(ws)))
        if not shards_ok:
            missing_ranks = sorted(set(range(ws)) - got)
            print("SHARDS\t{st}\t{ws}\t{miss}".format(st=st, ws=ws, miss=",".join(map(str, missing_ranks))))
    elif len(shard_ranks) > 1:
        print("SHARDS\t{st}\t0\tmixed world sizes {w}".format(st=st, w=sorted(shard_ranks)))

    # The merger calls AutoConfig/AutoTokenizer on <actor>/huggingface, so those
    # two files specifically are what "the huggingface dir is here" has to mean.
    rels = {r for r, _ in keep}
    has_model = int(ws > 0)
    has_hf = int("huggingface/config.json" in rels and "huggingface/tokenizer_config.json" in rels)
    has_fsdp = int("fsdp_config.json" in rels)
    with open(os.path.join(mdir, f"step_{st}.tsv"), "w", encoding="utf-8") as fh:
        for r, sz in sorted(keep):
            fh.write(f"{r}\t{sz}\n")
    print(
        "STEP\t{st}\t{pre}\t{n}\t{total}\t{dl}\t{nk}\t{hm}\t{hh}\t{hf}\t{ws}\t{sh}".format(
            st=st, pre=pre, n=len(objs), total=total, dl=dl, nk=len(keep),
            hm=has_model, hh=has_hf, hf=has_fsdp, ws=ws, sh=shards_ok,
        )
    )
    if not (has_model and has_hf and has_fsdp and shards_ok):
        bad += 1

# Also expose the first key of the first resolved step for the head-object probe.
for st in steps:
    st = str(int(st))
    if st in found:
        pre = sorted(found[st])[0]
        rest = sorted(found[st][pre])[0][0]
        print(f"PROBEKEY\t{pre}{rest}")
        break

sys.exit(1 if bad else 0)
PY
PLAN_RC=$?
tee -a "$DRIVER_LOG" < "$R2_PLAN" > /dev/null

AVAILABLE_STEPS="$(awk -F'\t' '$1=="AVAILABLE"{print $2}' "$R2_PLAN")"
say "    steps in R2    ${AVAILABLE_STEPS:-none}"

if (( PLAN_RC != 0 )); then
  while IFS=$'\t' read -r kind a b c; do
    case "$kind" in
      MISSING)   say "    MISSING: global_step_$a is NOT in R2 under $R2_PREFIX" ;;
      AMBIGUOUS) say "    AMBIGUOUS: global_step_$a resolves to more than one actor prefix: $b" ;;
      SHARDS)    say "    STILL UPLOADING: global_step_$a is world_size $b and is missing rank shard(s) $c" ;;
    esac
  done < "$R2_PLAN"
  while IFS=$'\t' read -r kind st _pre _n _tot _dl _nk hm hh hf _ws sh; do
    [[ "$kind" == "STEP" ]] || continue
    (( hm )) || say "    INCOMPLETE: global_step_$st has no model_world_size_<W>_rank_<i>.pt shard"
    (( hh )) || say "    INCOMPLETE: global_step_$st is missing huggingface/config.json or huggingface/tokenizer_config.json (the merger opens both)"
    (( hf )) || say "    INCOMPLETE: global_step_$st has no fsdp_config.json"
    (( sh )) || say "    INCOMPLETE: global_step_$st does not have every rank shard yet (see STILL UPLOADING above)"
  done < "$R2_PLAN"
  say "    If a step is merely still uploading, wait for the trainer to finish that"
  say "    checkpoint and re-run; nothing has been downloaded."
  die "requested steps [$EVAL_STEPS] are not all present and complete in R2. Available: ${AVAILABLE_STEPS:-none}"
fi

# Real head-object probe on a key we know exists, so a listing-only permission
# cannot pass for a working read credential.
PROBE_KEY="$(awk -F'\t' '$1=="PROBEKEY"{print $2}' "$R2_PLAN" | head -1)"
[[ -n "$PROBE_KEY" ]] || die "no probe key resolved from the listing"
if "$AWS_BIN" s3api head-object --bucket "$R2_CKPT_BUCKET" --key "$PROBE_KEY" \
     --endpoint-url "$R2_ENDPOINT" --query 'ContentLength' --output text >/dev/null 2>&1; then
  say "    head-object    OK on a known key"
else
  die "head-object failed on a key the listing returned. The credential can list but not read; fix R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY."
fi

# ===========================================================================
# 4. Disk gate, on measured sizes rather than a guess.
#    Peak is ONE pulled checkpoint plus ALL merges, because each pull is deleted
#    the moment its merge succeeds.
# ===========================================================================
say ""
say "--- preflight: disk"

MAX_DL_GIB=0
TOTAL_CKPT_GIB=0
N_STEPS=0
while IFS=$'\t' read -r kind st _pre nobj tot dl _nk _hm _hh _hf ws _sh; do
  [[ "$kind" == "STEP" ]] || continue
  g_tot=$(( (tot + 1073741823) / 1073741824 ))
  g_dl=$(( (dl + 1073741823) / 1073741824 ))
  say "    global_step_$st: $nobj objects, world_size $ws (all shards present), ${g_tot} GiB listed, ${g_dl} GiB to pull (optimizer shards excluded)"
  (( g_dl > MAX_DL_GIB )) && MAX_DL_GIB=$g_dl
  TOTAL_CKPT_GIB=$(( TOTAL_CKPT_GIB + g_tot ))
  N_STEPS=$(( N_STEPS + 1 ))
done < "$R2_PLAN"
(( N_STEPS > 0 )) || die "no steps resolved from the R2 plan"

MERGE_TOTAL_GIB=$(( MERGE_GIB * N_STEPS ))
NEED_GIB=$(( MAX_DL_GIB + MERGE_TOTAL_GIB + BASE_CACHE_GIB + DISK_SLACK_GIB ))
NAIVE_GIB=$(( TOTAL_CKPT_GIB + MERGE_TOTAL_GIB + BASE_CACHE_GIB + DISK_SLACK_GIB ))
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

say ""
say "    A full 8B FSDP checkpoint lists at roughly ${CKPT_GIB_NOMINAL} GiB and a merged HF"
say "    model at roughly ${MERGE_GIB} GiB. Pulling all ${N_STEPS} requested checkpoints whole and"
say "    keeping them would need about ${NAIVE_GIB} GiB. This driver excludes the optimizer"
say "    shards on download and DELETES each pulled checkpoint as soon as its merge"
say "    succeeds, so the peak is one pull plus all merges:"
say "        ${MAX_DL_GIB} GiB  largest single pull        -> $CKPT_ROOT"
say "      + ${MERGE_TOTAL_GIB} GiB  ${N_STEPS} merged HF models, kept   -> $MERGED_ROOT"
say "      + ${BASE_CACHE_GIB} GiB  HF cache for $BASE_MODEL -> $HF_CACHE_DIR"
say "      + ${DISK_SLACK_GIB} GiB  logs, vLLM scratch, caches -> $OOD_EVAL_ROOT"
say "      = ${NEED_GIB} GiB total, charged per filesystem below"

# Charge each requirement to the filesystem that actually holds it, so an
# overridden CKPT_ROOT or HF_HOME on a small volume cannot pass a gate that only
# ever looked at OOD_EVAL_ROOT. Grouping is an awk pass over device<TAB>gib<TAB>path
# lines rather than an associative array, so this runs on any bash.
DISK_CLAIMS="$OOD_EVAL_ROOT/.disk_claims.tsv"
: > "$DISK_CLAIMS"
claim() {  # claim <path> <gib>
  local p="$1" g="$2" dev
  mkdir -p "$p" 2>/dev/null || true
  dev="$(df -Pk "$p" 2>/dev/null | awk 'NR==2{print $1}')"
  [[ -n "$dev" ]] || die "could not read the filesystem for $p"
  printf '%s\t%s\t%s\n' "$dev" "$g" "$p" >> "$DISK_CLAIMS"
}
claim "$CKPT_ROOT"     "$MAX_DL_GIB"
claim "$MERGED_ROOT"   "$MERGE_TOTAL_GIB"
claim "$HF_CACHE_DIR"  "$BASE_CACHE_GIB"
claim "$OOD_EVAL_ROOT" "$DISK_SLACK_GIB"

DISK_FAIL=0
while IFS=$'\t' read -r dev need where; do
  [[ -n "$dev" ]] || continue
  free_gib="$(df -Pk "$where" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')"
  say "    $dev (via $where): ${free_gib:-unknown} GiB free, ${need} GiB required"
  if ! [[ "$free_gib" =~ ^[0-9]+$ ]] || (( free_gib < need )); then
    say "    INSUFFICIENT DISK on $dev"
    DISK_FAIL=1
  fi
done < <(awk -F'\t' '{n[$1]+=$2; if(!($1 in p)) p[$1]=$3}
                     END{for (d in n) printf "%s\t%s\t%s\n", d, n[d], p[d]}' "$DISK_CLAIMS")
(( DISK_FAIL == 0 )) \
  || die "not enough free disk. Free space, or point CKPT_ROOT / MERGED_ROOT / OOD_EVAL_ROOT / HF_HOME at a bigger volume."

# Host RAM. The merge is a CPU job and it is the memory-hungriest thing this
# driver does: verl.model_merger loads every fp32 rank shard into RAM at once,
# casts each to bf16 while concatenating, and then materialises an empty bf16
# model on CPU to save through. That is roughly 60-70 GiB resident for an 8B
# checkpoint. There is no gate for it anywhere else in the chain, and it fires
# AFTER the download, which is the expensive thing to have to redo.
say ""
RAM_GIB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ "$RAM_GIB" =~ ^[0-9]+$ ]]; then
  say "    host RAM       ${RAM_GIB} GiB, merge needs about ${MERGE_RAM_GIB} GiB"
  (( RAM_GIB >= MERGE_RAM_GIB )) \
    || die "host RAM ${RAM_GIB} GiB is below the ${MERGE_RAM_GIB} GiB the 8B merge needs. Merging would OOM after the download. Use a bigger box, or override MERGE_RAM_GIB if you have measured this one."
else
  say "    WARN: could not read host RAM from 'free -g'; the ${MERGE_RAM_GIB} GiB merge gate is unchecked"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say ""
  say "=== DRY_RUN=1: every preflight gate passed, stopping before the first download ==="
  exit 0
fi

# ===========================================================================
# 5. Benchmark data. ood_prep.py owns every dataset definition. We only ask it
#    for the sets we do not already have, and we hand-build the in-domain dir
#    (the MATH test split is not one of ood_prep.py's benchmarks, it is the
#    training val set).
# ===========================================================================
say ""
say "--- data"

if [[ ! -f "$MATH_DATA_DIR/test.parquet" || ! -f "$MATH_DATA_DIR/train.parquet" ]]; then
  say "    preparing MATH parquet in $MATH_DATA_DIR"
  ( cd "$VERL_DIR" && python3 research/scripts/prepare_rlvr_math.py \
      --dataset math --local_save_dir "$MATH_DATA_DIR" ) >> "$OOD_EVAL_ROOT/prep.log" 2>&1
fi
[[ -f "$MATH_DATA_DIR/test.parquet" && -f "$MATH_DATA_DIR/train.parquet" ]] \
  || die "MATH parquet unavailable in $MATH_DATA_DIR (see $OOD_EVAL_ROOT/prep.log)"

IN_DOMAIN_DIR="$OOD_DATA_ROOT/$IN_DOMAIN_BENCH"
mkdir -p "$IN_DOMAIN_DIR" || die "could not create $IN_DOMAIN_DIR (is OOD_DATA_ROOT=$OOD_DATA_ROOT writable?)"
# -f -n, not a bare ln: a dangling symlink left by an earlier run on a wiped
# MATH_DATA_DIR fails an -e test and then fails 'ln -s' with "File exists".
for f in test.parquet train.parquet; do
  ln -sfn "$MATH_DATA_DIR/$f" "$IN_DOMAIN_DIR/$f" \
    || die "could not link $MATH_DATA_DIR/$f into $IN_DOMAIN_DIR"
  [[ -f "$IN_DOMAIN_DIR/$f" ]] || die "$IN_DOMAIN_DIR/$f does not resolve to a real file"
done
say "    in-domain      $IN_DOMAIN_DIR (MATH test split, the training val set)"

MISSING_BENCHES=()
for b in "${BENCH_NAMES[@]}"; do
  [[ -f "$OOD_DATA_ROOT/$b/test.parquet" ]] || MISSING_BENCHES+=("$b")
done
if (( ${#MISSING_BENCHES[@]} > 0 )); then
  say "    building ${#MISSING_BENCHES[@]} benchmark parquets via ood_prep.py: ${MISSING_BENCHES[*]}"
  ( cd "$VERL_DIR" && OOD_ROOT="$OOD_DATA_ROOT" MATH_TRAIN="$MATH_DATA_DIR/train.parquet" \
      python3 "$SCRIPT_DIR/ood_prep.py" --only "${MISSING_BENCHES[@]}" ) 2>&1 | tee -a "$OOD_EVAL_ROOT/prep.log"
fi
STILL_MISSING=()
for b in "${BENCH_NAMES[@]}"; do
  [[ -f "$OOD_DATA_ROOT/$b/test.parquet" ]] || STILL_MISSING+=("$b")
done
(( ${#STILL_MISSING[@]} == 0 )) \
  || die "ood_prep.py did not produce: ${STILL_MISSING[*]} (see $OOD_EVAL_ROOT/prep.log)"
say "    benchmarks     all ${#BENCH_NAMES[@]} parquets present under $OOD_DATA_ROOT"

# ===========================================================================
# 6. Launcher. Two generated files, both regenerated every run:
#      a) a patched copy of the base launcher (four BARE exports that no env var
#         can override, so they have to be patched, exactly as the run-98
#         training launcher does it), living next to the engine so its $HERE
#         still resolves,
#      b) a thin wrapper that adds the Hydra keys ood_eval.sh has no way to
#         forward. Wrapper keys go FIRST so ood_eval.sh's own overrides win.
#    Plus the nvidia-smi -L shim: the engine hard-detects NGPUS_PER_NODE from
#    nvidia-smi -L, which ignores CUDA_VISIBLE_DEVICES, so without the shim a
#    2-GPU slice on a 4-GPU box asks Ray for 4 GPUs and dies.
# ===========================================================================
say ""
say "--- launcher"

PATCHED="$VERL_DIR/examples/grpo_trainer/run98_eval.gen.sh"
cp "$BASE_LAUNCHER" "$PATCHED" || die "could not copy $BASE_LAUNCHER"
# PATCH_SPEC and its shape check are in section 1 so DRY_RUN=1 covers them.
for spec in "${PATCH_SPEC[@]}"; do
  IFS='|' read -r pname pfrom pto <<< "$spec"
  sed -i.bak -e "s/^export ${pname}=${pfrom}\$/export ${pname}=${pto}/" "$PATCHED" \
    || die "sed failed for ${pname}"
  rm -f "$PATCHED.bak"
  grep -q "^export ${pname}=${pto}\$" "$PATCHED" || die "${pname} patch missed (wanted ${pto})"
  say "    patched ${pname}: ${pfrom} -> ${pto}"
done
chmod +x "$PATCHED"

LAUNCHER="$OOD_EVAL_ROOT/run98_eval_launcher.gen.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Generated by run98_eval.sh. Do not edit, it is overwritten every run.
set -uo pipefail
export COMM_EFF_ENABLED=false
export CKPT_R2_ENABLED=false
export ROLLOUT_GPU_MEM_UTIL="\${ROLLOUT_GPU_MEM_UTIL:-$ROLLOUT_GPU_MEM_UTIL}"
exec bash "$PATCHED" \\
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \\
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \\
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \\
  "\$@"
EOF
chmod +x "$LAUNCHER"
say "    launcher       $LAUNCHER (max_model_len $MAX_MODEL_LEN)"

SHIM_DIR="$OOD_EVAL_ROOT/shim"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/nvidia-smi" <<EOF
#!/usr/bin/env bash
# Generated by run98_eval.sh. Makes a bare 'nvidia-smi -L' report only the GPUs
# in CUDA_VISIBLE_DEVICES, which the real binary ignores. Everything else is
# passed through to the real binary by absolute path, so this cannot recurse.
REAL="$NVIDIA_SMI_REAL"
if [[ "\$#" -eq 1 && "\$1" == "-L" && -n "\${CUDA_VISIBLE_DEVICES:-}" ]]; then
  n=\$(awk -F, '{print NF}' <<< "\$CUDA_VISIBLE_DEVICES")
  # NOT 'exec ... | head': in a pipeline exec only replaces the left subshell,
  # so the script would fall through and print the unfiltered list as well.
  "\$REAL" -L | head -n "\$n"
  exit 0
fi
exec "\$REAL" "\$@"
EOF
chmod +x "$SHIM_DIR/nvidia-smi"
say "    shim           $SHIM_DIR/nvidia-smi (real binary: $NVIDIA_SMI_REAL)"

# Offline by default even though WANDB_API_KEY exists (the engine demands the
# variable, it does not demand a live connection). Every score this driver reads
# comes from train.log, so the online path buys nothing and costs something: the
# rc=1 atexit teardown race has silently dropped final-step values before. Set
# WANDB_MODE explicitly to opt back in.
export WANDB_MODE="${WANDB_MODE:-offline}"
say "    wandb          WANDB_MODE=$WANDB_MODE (scores are read from train.log, not WandB)"

export VERL_DIR OOD_EVAL_ROOT OOD_DATA_ROOT LAUNCHER SHIM_DIR

# ===========================================================================
# 7. Model preparation: pull, size-verify, merge, delete shards.
#    The result is handed back in the global PREPARED_MODEL rather than on
#    stdout: a 91 GiB pull takes long enough that its progress lines have to
#    reach the terminal live, and a command substitution would swallow them.
# ===========================================================================
PREPARED_MODEL=""

# merge_is_complete <dir>: true only for a merged model vLLM can actually load,
# and it stamps .merge_ok so the next run of this driver can trust it. Checks
# the config, a tokenizer, at least one weight file, and that the weights add up
# to something the size of an 8B bf16 model -- an interrupted save_pretrained
# leaves a valid-looking config and a short or absent shard set.
merge_is_complete() {
  local out="$1" wbytes wgib nfiles
  [[ -f "$out/config.json" ]] || { say "    merge check: no config.json in $out"; return 1; }
  [[ -f "$out/tokenizer_config.json" ]] || { say "    merge check: no tokenizer in $out"; return 1; }
  nfiles="$(find "$out" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l | tr -d ' ')"
  (( nfiles > 0 )) || { say "    merge check: no weight files in $out"; return 1; }
  wbytes=0
  local f fsz
  while IFS= read -r f; do
    fsz="$(file_size "$f")"
    [[ "$fsz" =~ ^[0-9]+$ ]] || fsz=0
    wbytes=$(( wbytes + fsz ))
  done < <(find "$out" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null)
  wgib=$(( wbytes / 1073741824 ))
  if (( wgib < MERGE_MIN_GIB )); then
    say "    merge check: $out holds only ${wgib} GiB of weights across $nfiles files, expected at least ${MERGE_MIN_GIB}"
    return 1
  fi
  printf 'merged %s  %s GiB across %s files\n' "$(date -Iseconds)" "$wgib" "$nfiles" > "$out/.merge_ok"
  say "    merge check: $nfiles weight files, ${wgib} GiB, tokenizer present"
  return 0
}

pull_and_merge() {  # pull_and_merge <step> <tag>  -> sets PREPARED_MODEL
  local step="$1" tag="$2"
  local out="$MERGED_ROOT/$tag"
  local ck="$CKPT_ROOT/global_step_$step/actor"
  local manifest="$MANIFEST_DIR/step_$step.tsv"
  PREPARED_MODEL=""

  # RESUME ON SUCCESS ONLY. config.json is NOT a success marker: transformers'
  # save_pretrained writes the config BEFORE the weight shards, and the merger
  # creates target_dir at argument-parse time, so a merge killed part way (or a
  # host-OOM, which is the realistic failure at 8B) leaves a directory holding a
  # config and no weights. Keying resume on that hands a broken model to vLLM.
  # .merge_ok is written by merge_is_complete below, after the weights are on
  # disk and their combined size has been checked.
  if [[ -f "$out/.merge_ok" ]]; then
    say "    merged model already present and verified for $tag: $out"
    PREPARED_MODEL="$out"; return 0
  fi
  if [[ -f "$out/config.json" ]]; then
    say "    $out has a config but no .merge_ok: a previous merge did not finish. Redoing it."
    rm -rf "$out"
  fi

  local pre dl_bytes
  pre="$(awk -F'\t' -v s="$step" '$1=="STEP" && $2==s {print $3}' "$R2_PLAN")"
  dl_bytes="$(awk -F'\t' -v s="$step" '$1=="STEP" && $2==s {print $6}' "$R2_PLAN")"
  [[ -n "$pre" ]] || { say "    NO PLAN for step $step"; return 1; }

  local secs=$(( dl_bytes / (R2_MIN_MB_S * 1048576) + 600 ))
  (( secs < 1800 )) && secs=1800

  local attempt
  for attempt in 1 2; do
    mkdir -p "$ck"
    say "    pull  s3://$R2_CKPT_BUCKET/$pre -> $ck (attempt $attempt, timeout ${secs}s)"
    local -a cmd=("$AWS_BIN" s3 cp "s3://$R2_CKPT_BUCKET/$pre" "$ck/"
                  --recursive --exclude "optim*" --only-show-errors
                  --endpoint-url "$R2_ENDPOINT")
    if [[ -n "$TIMEOUT_BIN" ]]; then
      "$TIMEOUT_BIN" "$secs" "${cmd[@]}" >> "$OOD_EVAL_ROOT/pull.log" 2>&1
    else
      "${cmd[@]}" >> "$OOD_EVAL_ROOT/pull.log" 2>&1
    fi
    local rc=$?
    if (( rc == 124 )); then
      say "    pull TIMED OUT after ${secs}s. Sample list-parts and 'ps -o time=' together:"
      say "    both flat means a hang, both climbing means merely slow (raise R2_MIN_MB_S)."
    elif (( rc != 0 )); then
      say "    aws s3 cp exited $rc (see $OOD_EVAL_ROOT/pull.log)"
    fi

    # Size-verify every expected object. A part can go missing silently, so a
    # zero exit from aws is not evidence that the bytes are all here.
    local bad=0 rel sz got
    while IFS=$'\t' read -r rel sz; do
      [[ -n "$rel" ]] || continue
      if [[ ! -f "$ck/$rel" ]]; then
        say "    MISSING after pull: $rel"; bad=$(( bad + 1 )); continue
      fi
      got="$(file_size "$ck/$rel")"
      if [[ "$got" != "$sz" ]]; then
        say "    SIZE MISMATCH $rel: local $got, R2 $sz"; bad=$(( bad + 1 ))
      fi
    done < "$manifest"

    if (( bad == 0 )); then
      say "    pull verified: $(wc -l < "$manifest" | tr -d ' ') objects match R2 byte for byte"
      break
    fi
    say "    pull incomplete ($bad objects wrong) on attempt $attempt"
    (( attempt == 2 )) && { say "    GIVING UP on step $step"; return 1; }
  done

  [[ -f "$ck/fsdp_config.json" ]] || { say "    no fsdp_config.json in $ck"; return 1; }
  say "    merge global_step_$step -> $out"
  ( cd "$VERL_DIR" && OMP_NUM_THREADS=8 python3 -m verl.model_merger merge \
      --backend fsdp --local_dir "$ck" --target_dir "$out" ) >> "$OOD_EVAL_ROOT/merge.log" 2>&1

  # The merger writes the tokenizer from <ck>/huggingface. Belt and braces: if it
  # did not, copy it before the shards go away, because vLLM loads the tokenizer
  # from MODEL_PATH and a missing one fails at boot on every bench.
  if [[ ! -f "$out/tokenizer_config.json" && -d "$ck/huggingface" ]]; then
    cp -n "$ck/huggingface/"* "$out/" 2>/dev/null || true
    say "    copied tokenizer/config files from $ck/huggingface"
  fi

  if ! merge_is_complete "$out"; then
    say "    MERGE FAILED for step $step (see $OOD_EVAL_ROOT/merge.log); pulled shards kept at $ck for inspection"
    return 1
  fi

  say "    merge OK, deleting pulled shards $ck (peak disk stays at one checkpoint)"
  rm -rf "$ck"
  rmdir "$CKPT_ROOT/global_step_$step" 2>/dev/null || true
  PREPARED_MODEL="$out"
  return 0
}

# ===========================================================================
# 8. Eval. Per (model, bench), skipped when a result already exists.
# ===========================================================================

has_result() {  # has_result <tag> <bench>
  # -a because train.log carries carriage returns and progress-bar bytes that
  # can make grep decide the file is binary and skip the match.
  local f="$OOD_EVAL_ROOT/$1/$2/train.log"
  [[ -f "$f" ]] && grep -qa "acc/mean@" "$f" 2>/dev/null
}

run_tag() {  # run_tag <tag> <model_path>
  local tag="$1" model="$2" i=0
  local -a pids=()
  if [[ "$model" != "$BASE_MODEL" && ! -f "$model/.merge_ok" ]]; then
    say "    SKIP $tag: no verified merged model at '$model'"; return 1
  fi
  local spec b n t p gpus
  for spec in "${ALL_SPECS[@]}"; do
    read -r b n t p <<<"$spec"
    if has_result "$tag" "$b"; then
      say "    skip $tag/$b (result already on disk)"
      continue
    fi
    gpus="${PAIRS[$(( i % ${#PAIRS[@]} ))]}"
    say "    start $tag/$b on GPUs $gpus (n=$n temp=$t top_p=$p)"
    bash "$SCRIPT_DIR/ood_eval.sh" "$model" "$b" "$tag" "$gpus" "$n" "$t" "$p" &
    pids+=("$!")
    i=$(( i + 1 ))
    if (( ${#pids[@]} >= ${#PAIRS[@]} )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait "${pids[@]}"
  fi
  local ok=0
  for spec in "${ALL_SPECS[@]}"; do
    read -r b _n _t _p <<<"$spec"
    has_result "$tag" "$b" && ok=$(( ok + 1 ))
  done
  if (( ok == ${#ALL_SPECS[@]} )); then
    say "    DONE $tag: $ok/${#ALL_SPECS[@]} benchmarks"
  else
    say "    PARTIAL $tag: $ok/${#ALL_SPECS[@]} benchmarks (re-run this script to resume)"
  fi
}

for tag in "${TAGS[@]}"; do
  say ""
  say "=== model $tag  $(date -Iseconds) ==="
  if [[ "$tag" == "base" ]]; then
    model="$BASE_MODEL"
  else
    step="${tag#step}"
    # Do not pay for a 91 GiB pull if every bench for this tag is already done.
    pending=0
    for spec in "${ALL_SPECS[@]}"; do
      read -r b _n _t _p <<<"$spec"
      has_result "$tag" "$b" || pending=$(( pending + 1 ))
    done
    if (( pending == 0 )); then
      say "    all ${#ALL_SPECS[@]} benchmarks already done for $tag, no checkpoint pull needed"
      continue
    fi
    if pull_and_merge "$step" "$tag"; then
      model="$PREPARED_MODEL"
    else
      say "    SKIP $tag: checkpoint could not be prepared"
      continue
    fi
  fi
  run_tag "$tag" "$model"
done

# ===========================================================================
# 9. Table.
# ===========================================================================
RESULTS="$OOD_EVAL_ROOT/RESULTS_run98.txt"
say ""
OOD_EVAL_ROOT="$OOD_EVAL_ROOT" \
TAGS_CSV="$(IFS=,; echo "${TAGS[*]}")" \
BENCH_CSV="$(IFS=,; echo "${ALL_NAMES[*]}")" \
IN_DOMAIN_BENCH="$IN_DOMAIN_BENCH" \
BASE_INDOMAIN_REF="$BASE_INDOMAIN_REF" \
RUN_ID="$RUN_ID" \
python3 - <<'PY' | tee "$RESULTS" | tee -a "$DRIVER_LOG"
import os
import re

root = os.environ["OOD_EVAL_ROOT"]
tags = os.environ["TAGS_CSV"].split(",")
benches = os.environ["BENCH_CSV"].split(",")
indomain = os.environ["IN_DOMAIN_BENCH"]
ref = os.environ["BASE_INDOMAIN_REF"]
run_id = os.environ["RUN_ID"]

# Prefer the val-core key, which is what ood_eval.sh itself reports, so a stray
# acc/mean@ line elsewhere in the log cannot be picked up as the score. Fall
# back to the loose pattern only if the tight one finds nothing.
tight = re.compile(r"val-core/\S*?acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")
loose = re.compile(r"acc/mean@[0-9]+['\"]?[: ]+([0-9.]+)")
tbl = {}
for tag in tags:
    for b in benches:
        f = os.path.join(root, tag, b, "train.log")
        acc = None
        if os.path.exists(f):
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            m = tight.findall(text) or loose.findall(text)
            if m:
                acc = float(m[-1])
        tbl[(tag, b)] = acc

trained = [t for t in tags if t != "base"]
w = 11
head = f"{'benchmark':<14s}" + "".join(f"{t:>{w}s}" for t in tags)
head += "".join(f"{t + '-base':>{w + 3}s}" for t in trained)
print(f"run {run_id}: in-domain + OOD capability audit")
print("")
print(head)
print("-" * len(head))


def cell(v):
    return f"{v:{w}.4f}" if v is not None else f"{'.':>{w}s}"


for b in benches:
    line = f"{b:<14s}" + "".join(cell(tbl[(t, b)]) for t in tags)
    for t in trained:
        a, z = tbl[(t, b)], tbl[("base", b)]
        d = (a - z) if (a is not None and z is not None) else None
        line += f"{('%+.4f' % d) if d is not None else 'n/a':>{w + 3}s}"
    print(line)
    if b == indomain:
        print("-" * len(head))

print("")
base_id = tbl[("base", indomain)]
print(f"in-domain ({indomain}) is the MATH test split, scored with verl's own")
print("validation sampling (n=1, temperature 0), so the step-150 column is a")
print("cross-check of the val this run logged in-training at step 150, and the")
print("step-200 column is the only source at 200.")
print("")
if base_id is None:
    print(f"base in-domain here: not measured. In-training step-0 val was {ref}.")
else:
    print(f"base in-domain here {base_id:.4f} against the in-training step-0 val {ref}")
    print(f"(delta {float(base_id) - float(ref):+.4f}). A large gap there means the eval")
    print("surface drifted from the training surface, not that the model changed.")
print("")
print("A dot means no result on disk for that cell. Re-running the driver resumes it.")
PY

say ""
say "=== run 98 eval finished  $(date -Iseconds) ==="
say "    table   $RESULTS"
say "    log     $DRIVER_LOG"

# The completion marker is SUCCESS-ONLY. An unconditional touch here would tell
# a watcher the audit is done while cells are still dots, which is exactly the
# mistake the per-cell resume logic is built to avoid.
CELLS_DONE=0
CELLS_TOTAL=0
for tag in "${TAGS[@]}"; do
  for spec in "${ALL_SPECS[@]}"; do
    read -r b _n _t _p <<<"$spec"
    CELLS_TOTAL=$(( CELLS_TOTAL + 1 ))
    has_result "$tag" "$b" && CELLS_DONE=$(( CELLS_DONE + 1 ))
  done
done
rm -f "$OOD_EVAL_ROOT/RUN98_EVAL_DONE" "$OOD_EVAL_ROOT/RUN98_EVAL_PARTIAL"
if (( CELLS_DONE == CELLS_TOTAL )); then
  say "    cells   $CELLS_DONE/$CELLS_TOTAL, complete"
  touch "$OOD_EVAL_ROOT/RUN98_EVAL_DONE"
else
  say "    cells   $CELLS_DONE/$CELLS_TOTAL, INCOMPLETE. Re-run this script to resume the rest."
  touch "$OOD_EVAL_ROOT/RUN98_EVAL_PARTIAL"
fi
